"""Encode images with a SubCellSpace CVAE, in process.

Why this module exists
----------------------
Every other model in this harness is either a lookup (SubCellSpace's latents
were computed once at data-processing time and live in ``obsm['latent']``) or an
out-of-process adapter (PointNet, ELLA). A *variant* CVAE is neither: it has the
same architecture and the same environment as the shipped model, so it needs no
adapter, but it has no stored latents either, so every artifact has to be
encoded from scratch.

The recipe is ``make_paper_figures/src/shuffle.py:64-79``, copied exactly:

    gene_img  = render_image(rows, cols)            # pixels -> 1, gaussian 1.5
    gene_t    = cvae.prepare_new_input(gene_img)
    dapi_t    = cvae.prepare_new_input(dapi_img)
    input_img = torch.cat((gene_t, dapi_t), dim=1)  # 3-channel: (..., cell_t)
    z, mu, logvar = cvae.embed(input_img)

Two things about that recipe decide whether a comparison means anything.

``embed`` is stochastic and unseeded
------------------------------------
``z = mu + exp(logvar/2) * randn_like(std)`` draws from the global torch RNG and
nothing in the published call path seeds it, so encoding one image twice gives
two different latents. The published pipeline stores the *sampled* ``z``. This
module makes the choice explicit and records it:

``mode="mu"``  deterministic; the posterior mean. Use for anything new.
``mode="z"``   seeded sample. Use only to reproduce stored latents.

Not every SubCellSpace model has two channels
---------------------------------------------
The shipped model takes ``(gene, DAPI)``. The ``supervised_with_dapi_with_cell_mask``
zoo takes ``(gene, DAPI, cell_mask)`` -- ``CVAE_hard_coded(n_channels=3, ...)``
in ``~/CVAE/train_with_dapi_and_cell.py:71``, assembled as
``torch.cat((x, y, z), dim=0)`` over gene / DAPI / cell mask in
``~/CVAE/utils.py:691-694``. The arity is read off
``encoder_conv.0.weight.shape[1]`` and never assumed; a 3-channel model asked to
encode without cell masks raises rather than silently building a 2-channel
batch it cannot consume.

The mask channels are not binary
--------------------------------
``obsm/dapi_mask`` and ``obsm/cell_mask`` are float64 with graded values, and
``prepare_new_input`` min-max scales whatever it is handed. Read them with
``H5adMasks(..).iter_masks(cells, raw=True)``; the default boolean masks are for
the geometry path and would feed the CVAE channels it was never trained on.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

import numpy as np

from .controls import RENDER_SHAPE, RENDER_SIGMA, ControlSet, render_image
from .config import SEED

MODES = ("mu", "z")


# --- the model file ----------------------------------------------------------

def load_model(path, device: Optional[str] = None) -> Dict[str, Any]:
    """``torch.load`` a SubCellSpace checkpoint and sanity-check it.

    The file is a *dict*, not a module. ``ld`` is read from it and never
    assumed: the zoo holds 5, 10, 15, 20 and 25-dimensional models whose
    filenames are the only other record of which is which.
    """
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.load(str(path), map_location=dev)
    except ModuleNotFoundError as e:
        if e.name == "CVAE":
            raise ModuleNotFoundError(
                "unpickling this checkpoint needs the repo-root CVAE.py importable as "
                "'CVAE' -- add the SubCellSpace repo root to sys.path before loading. "
                "The file stores a CVAE_hard_coded *instance*, not just weights."
            ) from e
        raise
    if not isinstance(model, dict) or "cvae" not in model:
        raise ValueError("{}: expected a dict with a 'cvae' key, got {}"
                         .format(path, type(model).__name__))
    model["device"] = dev
    model["cvae"].to(dev)
    model["cvae"].eval()
    if "ld" not in model:
        model["ld"] = int(model["cvae"].enc_mu.out_features)
    model["n_channels"] = int(model["cvae_state_dict"]["encoder_conv.0.weight"].shape[1])
    if model["n_channels"] not in (2, 3):
        raise ValueError("{}: unsupported channel count {}"
                         .format(path, model["n_channels"]))
    return model


def model_summary(model: Dict[str, Any], path=None) -> Dict[str, Any]:
    """The provenance worth stamping into results ``params``."""
    keep = ("ld", "beta", "lr", "class_lr", "n_classes", "epochs", "reduction")
    out = {k: model.get(k) for k in keep if k in model}
    out["latent_dim"] = int(model["ld"])
    out["n_channels"] = int(model.get("n_channels", 2))
    if path is not None:
        out["model_path"] = str(path)
        out["model_name"] = Path(path).name
    return out


# --- preprocessing -----------------------------------------------------------

def prepare_batch(imgs: np.ndarray) -> "torch.Tensor":
    """Batched equivalent of ``CVAE_hard_coded.prepare_new_input``.

    That method is ``ToTensor`` -> per-image ``Normalize(mean, std)`` ->
    per-image ``custom_scale(0, 1)``, and ``custom_scale`` is plain min-max. A
    z-score is a monotone affine map, so min-max applied after it lands in
    exactly the same place as min-max applied to the raw image, and the whole
    function collapses to per-image min-max. That equivalence is *checked*, not
    argued: see ``check_prepare_batch``.

    The two differ on one input: a constant image. The original raises
    ``ValueError("std evaluated to zero...")`` inside ``transforms.Normalize``;
    this one divides by zero and yields NaN. Both refuse the image, just at
    different points -- ``encode_images`` rejects NaN latents rather than
    letting them reach a benchmark. Measured over 100 non-constant images
    (sparse renders, graded DAPI-like, binary, and 1e-7..1e3 scales) the two
    paths agree to a max absolute difference of 1.2e-07, i.e. float32 rounding.
    """
    import torch
    x = imgs if torch.is_tensor(imgs) else torch.as_tensor(np.asarray(imgs, dtype=np.float32))
    if x.ndim == 2:
        x = x[None]
    squeeze = x.ndim == 4          # already (N,1,H,W) from render_batch
    if squeeze:
        x = x[:, 0]
    flat = x.reshape(x.shape[0], -1)
    vmin = flat.min(dim=1).values[:, None, None]
    vmax = flat.max(dim=1).values[:, None, None]
    return ((x - vmin) / (vmax - vmin))[:, None]


def gaussian_kernel1d(sigma: float = RENDER_SIGMA, truncate: float = 4.0):
    """The exact 1-D weights ``scipy.ndimage.gaussian_filter`` uses.

    ``skimage.filters.gaussian`` delegates to it, so reproducing these weights
    (radius ``int(truncate*sigma + 0.5)``, normalised to sum 1) is what lets the
    GPU renderer stand in for the published CPU one.
    """
    lw = int(truncate * sigma + 0.5)
    x = np.arange(-lw, lw + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), lw


def blur_batch(x, sigma: float = RENDER_SIGMA):
    """Separable Gaussian over ``(N,1,H,W)`` with ``mode='nearest'`` padding."""
    import torch
    import torch.nn.functional as F
    k, lw = gaussian_kernel1d(sigma)
    kt = torch.as_tensor(k, dtype=x.dtype, device=x.device)
    x = F.conv2d(F.pad(x, (lw, lw, 0, 0), mode="replicate"), kt.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, lw, lw), mode="replicate"), kt.view(1, 1, -1, 1))
    return x


def render_batch(rowcols, device=None, sigma: float = RENDER_SIGMA,
                 shape: Tuple[int, int] = RENDER_SHAPE, dtype=None):
    """Render many point sets to blurred images in one GPU pass.

    ``rowcols`` is a sequence of ``(rows, cols)`` pairs. Scatter-then-blur, with
    the pixel *set* to 1 rather than accumulated, so duplicate points collapse
    exactly as ``controls.render_image`` makes them.

    This exists because rendering was ~37% of encode time once the grouping bug
    was fixed, and at full APEX scale that is hours of one core.
    """
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = dtype or torch.float64
    n, (H, W) = len(rowcols), shape
    img = torch.zeros((n, 1, H, W), dtype=dt, device=dev)
    if n:
        bi, fi = [], []
        for i, (r, c) in enumerate(rowcols):
            r = np.asarray(r, dtype=np.int64)
            c = np.asarray(c, dtype=np.int64)
            ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            r, c = r[ok], c[ok]
            bi.append(np.full(r.shape, i, dtype=np.int64))
            fi.append(r * W + c)
        if bi:
            b = torch.as_tensor(np.concatenate(bi), device=dev)
            f = torch.as_tensor(np.concatenate(fi), device=dev)
            img.view(n, -1)[b, f] = 1
    return blur_batch(img, sigma=sigma)


def check_render_batch(rowcols, atol: float = 1e-6, device=None) -> Dict[str, Any]:
    """Prove the GPU renderer equals ``controls.render_image`` point for point."""
    ref = np.stack([render_image(r, c) for r, c in rowcols])
    got = render_batch(rowcols, device=device)[:, 0].cpu().numpy()
    diff = float(np.abs(ref - got).max())
    return {"max_abs_diff": diff, "n": len(rowcols), "passes": diff < atol}


def check_prepare_batch(cvae, imgs: Sequence[np.ndarray], atol: float = 1e-6
                        ) -> Dict[str, Any]:
    """V1: prove the batched path equals the published per-image one.

    Returns the max absolute difference and whether NaNs land in the same
    places. Call it before trusting ``prepare_batch`` on real data.
    """
    import torch
    imgs = [np.asarray(im) for im in imgs]
    const = {i for i, im in enumerate(imgs) if im.min() == im.max()}
    usable = [im for i, im in enumerate(imgs) if i not in const]
    if not usable:
        raise ValueError("every image is constant; nothing to compare")
    ref = torch.cat([cvae.prepare_new_input(im.astype(np.float64))
                     for im in usable], dim=0)
    got = prepare_batch(np.stack([im.astype(np.float32) for im in usable]))
    diff = float((ref - got).abs().max())
    return {"max_abs_diff": diff, "n_images": len(usable),
            "n_constant_skipped": len(const),
            "any_nan": bool(torch.isnan(ref).any() or torch.isnan(got).any()),
            "passes": diff < atol}


# --- encoding ----------------------------------------------------------------

def encode_images(model: Dict[str, Any], gene_imgs=None, dapi_imgs=None, cell_imgs=None,
                  mode: str = "mu", seed: int = SEED, batch_size: int = 512,
                  progress: bool = False, gene_points=None,
                  render_dtype=None) -> np.ndarray:
    """Encode paired channel images to latents.

    ``gene_imgs``/``dapi_imgs`` (and ``cell_imgs`` for a 3-channel model) are
    equal-length sequences of 100x100 arrays. Channel order is
    ``(gene, DAPI, cell_mask)``, matching the training assembly. Returns
    ``(n, ld)`` float32.
    """
    import torch
    if mode not in MODES:
        raise ValueError("mode must be one of {}".format(MODES))
    nch = int(model.get("n_channels", 2))
    if (gene_imgs is None) == (gene_points is None):
        raise ValueError("pass exactly one of gene_imgs or gene_points")
    if gene_points is not None:
        gene_points = list(gene_points)
        gene_imgs = [None] * len(gene_points)   # length only; rendered per batch
    else:
        gene_imgs = list(gene_imgs)
    dapi_imgs = list(dapi_imgs)
    cell_imgs = list(cell_imgs) if cell_imgs is not None else None
    if len(gene_imgs) != len(dapi_imgs):
        raise ValueError("{} gene images but {} DAPI images"
                         .format(len(gene_imgs), len(dapi_imgs)))
    if nch == 3:
        if cell_imgs is None:
            raise ValueError(
                "this model has 3 input channels (gene, DAPI, cell_mask) but no "
                "cell masks were supplied; encoding it as 2 channels would feed "
                "it something it was never trained on")
        if len(cell_imgs) != len(gene_imgs):
            raise ValueError("{} gene images but {} cell masks"
                             .format(len(gene_imgs), len(cell_imgs)))
    elif cell_imgs is not None:
        raise ValueError("cell masks supplied but this model takes {} channels"
                         .format(nch))
    if not gene_imgs:
        return np.zeros((0, int(model["ld"])), dtype=np.float32)

    cvae, dev = model["cvae"], model["device"]
    out = np.empty((len(gene_imgs), int(model["ld"])), dtype=np.float32)
    if mode == "z":
        torch.manual_seed(seed)

    with torch.no_grad():
        for start in range(0, len(gene_imgs), batch_size):
            stop = min(start + batch_size, len(gene_imgs))
            if gene_points is not None:
                g = prepare_batch(render_batch(gene_points[start:stop], device=dev,
                                               dtype=render_dtype or torch.float32))
            else:
                g = prepare_batch(np.stack(gene_imgs[start:stop])).to(dev)
            chans = [g.float(), prepare_batch(np.stack(dapi_imgs[start:stop])).to(dev).float()]
            if nch == 3:
                chans.append(prepare_batch(np.stack(cell_imgs[start:stop])).to(dev).float())
            z, mu, _ = cvae.embed(torch.cat(chans, dim=1))
            out[start:stop] = (mu if mode == "mu" else z).cpu().numpy()
            if progress and (start // batch_size) % 20 == 0:
                print("  encoded %d/%d" % (stop, len(gene_imgs)), flush=True)

    bad = int(np.isnan(out).any(axis=1).sum())
    if bad:
        raise ValueError("{} of {} latents are NaN -- a constant input image "
                         "min-max scales to NaN. Drop those observations before "
                         "encoding.".format(bad, len(out)))
    return out


def encode_control_set(model: Dict[str, Any], cs: ControlSet, dapi: Dict[str, np.ndarray],
                       cell: Optional[Dict[str, np.ndarray]] = None,
                       mode: str = "mu", seed: int = SEED, batch_size: int = 512,
                       sigma: float = RENDER_SIGMA, obs_keys: Optional[Sequence[str]] = None,
                       progress: bool = False) -> Tuple[np.ndarray, List[str]]:
    """Render and encode every observation of a ``ControlSet``.

    ``dapi`` (and ``cell`` for a 3-channel model) map the mask-carrying entity to
    its image. For real data an ``obs_key`` is ``"<cell>__<gene>"`` and the masks
    belong to the *cell*, so lookup falls back to the key's cell part -- one mask
    read per cell, reused across that cell's genes. For simulated data the key is
    the image stem and the maps are keyed by stem directly.

    An observation whose masks are missing is skipped, so the returned key list
    is what was actually encoded and may be shorter than ``obs_keys``.
    """
    nch = int(model.get("n_channels", 2))
    if nch == 3 and cell is None:
        raise ValueError("this model needs cell masks; pass cell=")
    keys = ([str(k) for k in obs_keys] if obs_keys is not None
            else [str(k) for k in cs.points["obs_key"].unique()])
    # Subset the point table to the requested observations *before* grouping.
    # A ControlSet holds every observation in the dataset (514,512 for APEX at
    # 2,031 cells), so grouping the whole table to encode a few hundred of them
    # costs seconds per call and dominates everything else.
    wanted = set(keys)
    pts = cs.points
    if len(wanted) < pts["obs_key"].nunique():
        pts = pts[pts["obs_key"].isin(wanted)]
    by_obs = {k: g for k, g in pts.groupby("obs_key", sort=False)}

    def lookup(store, key):
        if store is None:
            return None
        v = store.get(key)
        return v if v is not None else store.get(key.rsplit("__", 1)[0])

    genes, dapis, cells, used = [], [], [], []
    for key in keys:
        grp = by_obs.get(key)
        if grp is None:
            continue
        d = lookup(dapi, key)
        c = lookup(cell, key) if nch == 3 else None
        if d is None or (nch == 3 and c is None):
            continue
        genes.append((grp["row"].values, grp["col"].values))
        dapis.append(np.asarray(d, dtype=np.float32).reshape(RENDER_SHAPE))
        if nch == 3:
            cells.append(np.asarray(c, dtype=np.float32).reshape(RENDER_SHAPE))
        used.append(key)

    X = encode_images(model, dapi_imgs=dapis, cell_imgs=cells if nch == 3 else None,
                      gene_points=genes, mode=mode, seed=seed,
                      batch_size=batch_size, progress=progress)
    return X, used


def encode_image_paths(model: Dict[str, Any], gene_paths, dapi_paths, cell_paths=None,
                       mode: str = "mu", seed: int = SEED, batch_size: int = 512,
                       io_workers: int = 16, progress: bool = False) -> np.ndarray:
    """Encode from rendered images on disk, streaming batch by batch.

    For *simulated* data this is the faithful source. Those coordinate trees were
    recovered from the blurred images by peak detection, so re-rendering them
    reproduces the stored latent only to cosine ~0.9995 (min 0.82 over a
    400-row probe); reading the original TIFF the model was trained on
    reproduces it at median 1.000000, min 0.999996. APEX is the other way round --
    its stored integer coordinates *are* the source, and rendering them is exact.

    Images are loaded one batch ahead on a thread pool rather than up front:
    44,706 rows x 3 channels x 100x100 float32 is 5.4 GB if materialised.
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from skimage import io

    if mode not in MODES:
        raise ValueError("mode must be one of {}".format(MODES))
    nch = int(model.get("n_channels", 2))
    gene_paths = [str(x) for x in gene_paths]
    dapi_paths = [str(x) for x in dapi_paths]
    cell_paths = [str(x) for x in cell_paths] if cell_paths is not None else None
    if len(dapi_paths) != len(gene_paths):
        raise ValueError("{} gene paths but {} DAPI paths"
                         .format(len(gene_paths), len(dapi_paths)))
    if nch == 3 and cell_paths is None:
        raise ValueError("this model has 3 input channels but no cell masks were given")
    if nch == 3 and len(cell_paths) != len(gene_paths):
        raise ValueError("{} gene paths but {} cell paths"
                         .format(len(gene_paths), len(cell_paths)))
    if not gene_paths:
        return np.zeros((0, int(model["ld"])), dtype=np.float32)

    def read(paths):
        return np.stack([np.asarray(im, dtype=np.float32)
                         for im in pool.map(io.imread, paths)])

    cvae, dev = model["cvae"], model["device"]
    out = np.empty((len(gene_paths), int(model["ld"])), dtype=np.float32)
    if mode == "z":
        torch.manual_seed(seed)

    with ThreadPoolExecutor(max_workers=io_workers) as pool, torch.no_grad():
        for start in range(0, len(gene_paths), batch_size):
            stop = min(start + batch_size, len(gene_paths))
            chans = [prepare_batch(read(gene_paths[start:stop])).to(dev).float(),
                     prepare_batch(read(dapi_paths[start:stop])).to(dev).float()]
            if nch == 3:
                chans.append(prepare_batch(read(cell_paths[start:stop])).to(dev).float())
            z, mu, _ = cvae.embed(torch.cat(chans, dim=1))
            out[start:stop] = (mu if mode == "mu" else z).cpu().numpy()
            if progress and (start // batch_size) % 20 == 0:
                print("  encoded %d/%d" % (stop, len(gene_paths)), flush=True)

    bad = int(np.isnan(out).any(axis=1).sum())
    if bad:
        raise ValueError("{} of {} latents are NaN".format(bad, len(out)))
    return out


def tree_paths(tree, stems: Sequence[str], need_cell: bool = False):
    """``(gene, dapi, cell, kept_stems)`` paths inside a simulated tree.

    Layout: ``<tree>/<stem>.tif``, ``<tree>/dapi_zoomed/<stem>_DAPI.tif``,
    ``<tree>/cell_mask_zoomed/<stem>_cell-mask.tif``. Paths are constructed and
    then existence-checked, never listed -- these directories hold 805,549
    entries and walking one costs more than the reads.
    """
    tree = Path(tree)
    g, d, c, kept = [], [], [], []
    for stem in stems:
        stem = str(stem)
        gp = tree / (stem + ".tif")
        dp = tree / "dapi_zoomed" / (stem + "_DAPI.tif")
        cp = tree / "cell_mask_zoomed" / (stem + "_cell-mask.tif")
        if not (gp.is_file() and dp.is_file()):
            continue
        if need_cell and not cp.is_file():
            continue
        g.append(gp); d.append(dp); c.append(cp); kept.append(stem)
    return g, d, c, kept


def encode_tree_stems(model: Dict[str, Any], tree, stems: Sequence[str],
                      mode: str = "mu", seed: int = SEED, batch_size: int = 512,
                      io_workers: int = 16, progress: bool = False
                      ) -> Tuple[np.ndarray, List[str]]:
    """Encode simulated observations by image stem, from the rendered TIFFs."""
    nch = int(model.get("n_channels", 2))
    g, d, c, kept = tree_paths(tree, stems, need_cell=(nch == 3))
    X = encode_image_paths(model, g, d, c if nch == 3 else None, mode=mode, seed=seed,
                           batch_size=batch_size, io_workers=io_workers,
                           progress=progress)
    return X, kept


def masks_by_cell(h5ad_path, cells: Sequence[str]
                  ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """``({cell: dapi}, {cell: cell_mask})`` read block-wise, stored float values."""
    from .ella_io import H5adMasks
    dapi, cell_mask = {}, {}
    for cell, cm, d in H5adMasks(h5ad_path).iter_masks(cells, raw=True):
        if d is not None:
            dapi[str(cell)] = np.asarray(d, dtype=np.float32)
        if cm is not None:
            cell_mask[str(cell)] = np.asarray(cm, dtype=np.float32)
    return dapi, cell_mask


def dapi_by_cell(h5ad_path, cells: Sequence[str]) -> Dict[str, np.ndarray]:
    """``{cell: dapi_image}`` read block-wise, with the stored float values."""
    return masks_by_cell(h5ad_path, cells)[0]


def masks_by_stem(tree, stems: Sequence[str], dapi_dir: str = "dapi_zoomed",
                  cell_dir: str = "cell_mask_zoomed"
                  ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """``({stem: dapi}, {stem: cell_mask})`` from a simulated coordinate tree.

    Paths are constructed, not listed: these directories hold 805,549 entries.
    """
    from skimage import io
    droot, croot = Path(tree) / dapi_dir, Path(tree) / cell_dir
    dapi, cell_mask = {}, {}
    for stem in stems:
        dp = droot / (str(stem) + "_DAPI.tif")
        if dp.is_file():
            dapi[str(stem)] = np.asarray(io.imread(dp), dtype=np.float32)
        cp = croot / (str(stem) + "_cell-mask.tif")
        if cp.is_file():
            cell_mask[str(stem)] = np.asarray(io.imread(cp), dtype=np.float32)
    return dapi, cell_mask


def dapi_by_stem(tree, stems: Sequence[str], dapi_dir: str = "dapi_zoomed"
                 ) -> Dict[str, np.ndarray]:
    """``{stem: dapi_image}`` from a simulated coordinate tree."""
    return masks_by_stem(tree, stems, dapi_dir=dapi_dir)[0]


# --- dumps -------------------------------------------------------------------

def write_encoded(path, X: np.ndarray, keys: Sequence[str], index_kind: str,
                  method: str, dataset: str, model: Dict[str, Any],
                  model_path=None, mode: str = "mu", seed: int = SEED,
                  extra: Optional[Dict[str, Any]] = None):
    """``write_dump`` with the encode convention recorded alongside.

    ``encode_mode`` and ``seed`` belong in the dump because a ``mu`` dump and a
    ``z`` dump of the same model over the same images are different data, and
    nothing else on disk distinguishes them.
    """
    from .representations import write_dump
    meta = dict(model_summary(model, model_path))
    meta.update({"encode_mode": mode, "encode_seed": int(seed),
                 "render_sigma": RENDER_SIGMA})
    meta.update(extra or {})
    return write_dump(path, X, keys, index_kind=index_kind, method=method,
                      dataset=dataset, extra=meta)
