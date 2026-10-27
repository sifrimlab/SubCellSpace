"""Derive a method's earth-mover threshold the way the paper derived its own.

``config.DISTANCE_THRESHOLD`` (0.06) and ``MEDIUM_DISTANCE_THRESHOLD`` (0.03)
were not chosen; they were *read off* a figure. This module reproduces that
reading so any method can be given the same pair of landmarks on the same
simulated data, blind to the real datasets' ground-truth panels.

Why this exists
---------------
``detection.sweep.threshold_sweep`` also reports a threshold -- the Youden-J
optimum -- but that one is fitted on the *evaluation* labels. It cannot be
computed for a new dataset, and comparing a method at its label-fitted optimum
against another at a blind, simulation-derived cut penalises the one that played
fair. Youden stays, relabelled as a *headroom* reading; this is what a
deployment threshold has to be derived from.

The procedure, from ``make_paper_figures/fig4_...ipynb`` cells 26-32 (the
paper's Fig 4D)
---------------------------------------------------------------------------
1. Take the all-proportions test pair -- the same two files
   ``detection.classifier.load_training_pair`` reads, i.e. artifact A -- and keep
   the high-proportion rows (``prop >= 0.8``).
2. Build a **random baseline curve**: ``n_steps + 1`` independent pairs of
   disjoint 1,000-row samples drawn from the *shuffled* half, each pair scored
   against the other. Its **maximum** is the control ceiling: the largest earth
   mover's distance this method produces between two samples that differ by
   nothing but the draw. Published value for the shipped latent: 0.0176.
3. Per pattern subtype, build **mixtures** at proportions ``step / n_steps``,
   mixing ``int(max_obs * proportion)`` patterned rows with random ones, and
   score each against a fixed full-random baseline sample of the same subtype.
   That gives an EMD-versus-proportion curve per subtype.
4. Read two landmarks off those curves:

   **lenient** -- ``find_threshold_x`` returns, per subtype, the first EMD from
   which the curve never falls back to the ceiling. The maximum across subtypes
   is the EMD above which *every* subtype is detectable. Published: 0.03.

   **stringent** -- ``find_stable_x`` on the **cell-edge** curve: the first EMD
   from which the curve stops decreasing as more clearly patterned data enters
   the mixture. Published: 0.06.

A note on the two landmark definitions
--------------------------------------
The plan that commissioned this described *lenient* as "the midpoint between the
max random EMD and the departure point". The published cell does no such
midpoint: ``max_diff`` is exactly ``max(find_threshold_x(...))`` over the
subtypes, plotted as ``max_diff at y=...``. The code is what produced the
figure, so the code is what is reproduced here -- and the check that this is the
right reading is that SubCellSpace comes back at the published 0.03 / 0.06.

Why ``protrusion`` is excluded from the lenient max
---------------------------------------------------
Measured, not assumed. Running this procedure on artifact A with the shipped
classifier over five independent draws:

===========================  ========  ========  ==========
landmark                     mean      sd        published
===========================  ========  ========  ==========
control ceiling              0.01701   0.00049   0.017625
stringent (cell-edge)        0.06457   0.00380   **0.06**
lenient, all six subtypes    0.04874   0.00752   --
lenient, excluding protrusion 0.03080  0.00156   **0.03**
===========================  ========  ========  ==========

``protrusion`` is the argmax in **every** draw, and the all-subtype reading never
comes near 0.03. So the published 0.03 is the max over the other five, and this
is the only reading that reproduces it.

The mechanism is grid resolution on the steepest curve, not a property of the
weakest one. ``find_threshold_x`` returns the y *after* the crossing, so a curve
that leaves the ceiling behind in a single 1/18 step overshoots. Protrusion is
the most detectable subtype by a wide margin -- it climbs to EMD 0.738 against
cell-edge's 0.075, and jumps 0.0149 -> 0.0521 across the crossing -- so its
"departure" value is an artifact of where the grid happens to fall, and it
dominates a max. (``prop`` is not the cause: it spans 0.1-0.99 with median 0.54
for every subtype in this file, protrusion included.)

Both readings are emitted. ``lenient`` is the operative one; ``lenient_all``
carries the verbatim all-subtype value so the choice is visible rather than
buried, and switching between them moves *every* method's threshold together, so
the comparison stays like-for-like under either.

What is deliberately *not* parameterised
----------------------------------------
``find_stable_x`` is read off ``cell-edge`` because that is the subtype the
published cell hardcodes, not because cell-edge is provably weakest for every
method. Holding it fixed is what makes the derivation protocol-matched across
methods; letting each method pick its own worst subtype would be a different
procedure, and a more flattering one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import anndata as ad
import numpy as np

from ..config import SEED
from .genes import rep_matrix

#: Published constants. ``n_steps``/``max_obs`` are commented out in fig4 cell 32
#: and carried in from cell 31's ``n_samples = 18`` / ``sample_size = 1000``,
#: which the same cell's own assertion uses.
N_STEPS = 18
MAX_OBS = 1000

#: ``high_prop_data`` is cut at 0.8; the per-subtype slice is then cut again at
#: 0.5, which is a no-op after the first. Kept because it is what the cell does.
MIN_PROP = 0.8
MIX_PROP = 0.5

#: fig4 cell 32 seeds the per-subtype pool shuffles with 42. The random baseline
#: curve there uses the ambient global RNG, whose state at that point in the
#: notebook is not recoverable; SEED is used instead so the ceiling is
#: reproducible. Both are recorded in the emitted JSON.
MIX_SEED = 42

#: The subtype whose curve supplies the stringent landmark. See the module
#: docstring on why this is fixed rather than chosen per method.
STABLE_PATTERN = "cell-edge"

#: Subtypes kept out of the lenient max. See the module docstring: protrusion's
#: curve crosses the ceiling in one 1/18 grid step, so `find_threshold_x`'s
#: after-the-crossing value overshoots and dominates the max, and only the
#: five-subtype reading reproduces the published 0.03. Applied identically to
#: every method, so it cannot favour one.
LENIENT_EXCLUDE = ("protrusion",)

#: ``find_stable_x``'s tolerance for "no longer decreasing", verbatim from fig4.
STABLE_EPSILON = 1e-4


# --- the two landmark readers, verbatim from fig4 cells 26 and 27 -------------

def find_threshold_x(x, y, threshold):
    """First y from which every later y stays >= threshold. None if never.

    Verbatim from ``fig4`` cell 26 (whose signature defaults ``threshold`` to a
    hardcoded 0.0156). Note it returns the **y** value, not the x -- the figure
    draws it as a horizontal line, so the landmark is an EMD, not a proportion.
    """
    n = len(y)
    for i in range(n):
        if all(val >= threshold for val in y[i:]):
            return y[i]
    return None  # if no such point exists


def find_stable_x(x, y, epsilon=STABLE_EPSILON):
    """First (x, y) from which the curve never decreases by more than epsilon.

    Verbatim from ``fig4`` cell 27. The caller takes ``[1]``, the y value.

    **Degenerate on a monotone curve**, which is why ``find_stable_above`` below
    exists and is what the derivation actually uses. See its docstring.
    """
    n = len(y)
    for i in range(n):
        stable = True
        for j in range(i + 1, n):
            if y[j] < y[j - 1] - epsilon:
                stable = False
                break
        if stable:
            return x[i], y[i]
    return None  # no stable point found


def find_stable_above(x, y, ceiling, epsilon=STABLE_EPSILON):
    """First (x, y) that is stable **and** clear of the control ceiling.

    Both conditions come from the same figure cell -- this is
    ``find_stable_x``'s "no longer decreasing" AND ``find_threshold_x``'s "stays
    at or above the ceiling", required at the same index.

    Why the second condition has to be there
    ----------------------------------------
    ``find_stable_x`` alone answers "where does the curve stop going down", and
    on a curve that never goes down the answer is index 0 -- the pure-random
    mixture. Measured on artifact A, that is exactly what happens for three of
    the four methods:

    ==================  ==================  =================  ==============
    method              find_stable_x       its own ceiling    verdict
    ==================  ==================  =================  ==============
    subcellspace        0.061799            0.016987           above -- fine
    pointnet_vae_64     0.005035            0.008435           **below**
    bento_features      0.006078            0.007134           **below**
    cnn_baseline        0.003092            0.007672           **below**
    ==================  ==================  =================  ==============

    A threshold below the ceiling is a threshold two *random* samples clear, so
    it would call essentially every gene patterned and the comparison would say
    nothing. The published latent never needed the guard because its cell-edge
    curve happens to wobble at low proportion, so ``find_stable_x`` was already
    forced past the ceiling; the three cleanly monotone curves expose that the
    guard was implicit rather than absent.

    It changes nothing for SubCellSpace: the binding constraint there is
    stability at index 15, while the ceiling is already cleared from index 5, so
    this returns the same 0.061799 -- i.e. the published 0.06.
    """
    n = len(y)
    for i in range(n):
        if any(v < ceiling for v in y[i:]):
            continue
        if all(y[j] >= y[j - 1] - epsilon for j in range(i + 1, n)):
            return x[i], y[i]
    return None


# --- the statistic ------------------------------------------------------------

def _as_adata(X: np.ndarray, tag: str, gene: str) -> ad.AnnData:
    """A bare AnnData ``quantify_gene_patterness`` can consume.

    ``obs_names`` are synthesised per call rather than carried through from the
    source rows, for a reason that is easy to get wrong: the published function
    looks the control up as ``random_test_data[pattern.obs_names]`` and only
    falls back to positional matching when that raises ``KeyError``. Both halves
    of artifact A use positional integer names, so a real name could land in the
    baseline by coincidence and silently select the wrong rows. Giving the
    baseline the *same* synthetic names as the mixture makes the lookup exact --
    the same idiom fig4's own random arm uses
    (``subsampled_random2.obs_names = subsampled_random1.obs_names``) -- and the
    statistic is unaffected either way, since Wasserstein and KS both read the
    two score vectors as sets.
    """
    import pandas as pd
    obs = pd.DataFrame({"gene_name": [gene] * X.shape[0]},
                       index=["%s%06d" % (tag, i) for i in range(X.shape[0])])
    return ad.AnnData(X=np.asarray(X, dtype=np.float64), obs=obs)


def _emd(Xa: np.ndarray, Xb: np.ndarray, classifier, gene: str = "mix") -> float:
    """Earth mover's distance between the two arms' pattern-probabilities.

    Delegates to the published ``quantify_gene_patterness`` rather than
    recomputing, so the number here is the number the paper's own statistic
    produces. ``max_number_of_observations_to_consider`` equals the arm size, so
    its internal subsample never fires and the call is deterministic.
    """
    import sys
    from ..config import PUBLISHED_SRC
    src = str(PUBLISHED_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from classifiers import quantify_gene_patterness

    if Xa.shape[0] != Xb.shape[0]:
        raise ValueError("arms must be the same size, got %d and %d"
                         % (Xa.shape[0], Xb.shape[0]))
    a = _as_adata(Xa, "mix_", gene)
    b = _as_adata(Xb, "mix_", gene)          # same names: an exact lookup
    dist, pval, pat, ctl = quantify_gene_patterness(
        a, gene, classifier, b,
        max_number_of_observations_to_consider=Xa.shape[0], random_seed=SEED)
    if pat.n_obs != Xa.shape[0]:
        raise RuntimeError("quantify_gene_patterness kept %d of %d rows -- its "
                           "gene selection or subsample fired unexpectedly"
                           % (pat.n_obs, Xa.shape[0]))
    return float(dist), float(pval)


# --- the curves ---------------------------------------------------------------

def _slice(obs, mask) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def mixture_curves(real, ctrl, classifier, rep: Optional[str] = None,
                   n_steps: int = N_STEPS, max_obs: int = MAX_OBS,
                   min_prop: float = MIN_PROP, mix_prop: float = MIX_PROP,
                   mix_seed: int = MIX_SEED, baseline_seed: int = SEED,
                   patterns: Optional[Sequence[str]] = None,
                   verbose: bool = True) -> Dict[str, Any]:
    """EMD-versus-proportion per subtype, plus the pure-random baseline curve.

    ``real`` and ``ctrl`` are artifact A's two halves as a method's
    ``train_pair()`` returns them, so this works for a latent stored in
    ``obsm[rep]`` and for a dump carried in ``.X`` alike. Both need ``prop`` and
    ``pattern`` in ``.obs``; every registered method's pair supplies them, either
    from the h5ad or parsed back from the filename stems.
    """
    Xr = rep_matrix(real, rep)
    Xc = rep_matrix(ctrl, rep)
    pr = real.obs["prop"].to_numpy(dtype=float)
    pc = ctrl.obs["prop"].to_numpy(dtype=float)
    lr = real.obs["pattern"].astype(str).to_numpy()
    lc = ctrl.obs["pattern"].astype(str).to_numpy()

    hi_r = pr >= min_prop
    hi_c = pc >= min_prop
    if verbose:
        print("  high-proportion rows (prop >= %.2f): real %d of %d | random %d of %d"
              % (min_prop, int(hi_r.sum()), len(pr), int(hi_c.sum()), len(pc)))

    out: Dict[str, Any] = {"curves": {}, "pool_sizes": {}, "departure": {}}

    # --- the random baseline curve, and the ceiling it defines ----------------
    # Disjoint 1,000-row draws from the shuffled half, scored against each other:
    # two samples that differ by nothing but the draw. Its maximum is the largest
    # EMD this method produces from noise alone.
    idx_c_hi = _slice(ctrl.obs, hi_c)
    rng = np.random.default_rng(baseline_seed)
    perm = rng.permutation(idx_c_hi.size)
    need = (n_steps + 1) * 2 * max_obs
    if idx_c_hi.size < need:
        raise ValueError("the random baseline needs %d high-proportion shuffled "
                         "rows and only %d are available" % (need, idx_c_hi.size))
    bx, by = [], []
    for k in range(n_steps + 1):
        a = idx_c_hi[perm[(2 * k) * max_obs:(2 * k + 1) * max_obs]]
        b = idx_c_hi[perm[(2 * k + 1) * max_obs:(2 * k + 2) * max_obs]]
        d, _ = _emd(Xc[a], Xc[b], classifier, gene="random_%d" % k)
        bx.append(k / n_steps)
        by.append(d)
    out["curves"]["random"] = {"x": bx, "y": by}
    out["ceiling"] = float(np.max(by))
    if verbose:
        print("  random baseline: %d points, max (the ceiling) %.6f, median %.6f"
              % (len(by), out["ceiling"], float(np.median(by))))

    # --- one mixture curve per subtype ---------------------------------------
    names = list(patterns) if patterns is not None else sorted(set(lr) & set(lc))
    for name in names:
        m_pat = hi_r & (pr >= mix_prop) & (lr == name)
        m_rnd = hi_c & (lc == name)
        idx_pat = _slice(real.obs, m_pat)
        idx_rnd = _slice(ctrl.obs, m_rnd)
        out["pool_sizes"][name] = {"patterned": int(idx_pat.size),
                                   "random": int(idx_rnd.size)}

        rng = np.random.default_rng(mix_seed)
        p_perm = rng.permutation(idx_pat.size)
        r_perm = rng.permutation(idx_rnd.size)

        # Pointers advance so no row is reused across steps, and the baseline is
        # taken from the far end of the random permutation so it never overlaps
        # the rows mixed in. Both are fig4's own construction.
        need_p = sum(int(max_obs * (s / n_steps)) for s in range(n_steps + 1))
        need_r = sum(max_obs - int(max_obs * (s / n_steps)) for s in range(n_steps + 1))
        if idx_pat.size < need_p or idx_rnd.size < need_r + max_obs:
            raise ValueError(
                "%s needs %d patterned and %d random high-proportion rows "
                "(mixtures %d + baseline %d) but has %d and %d -- rows would be "
                "reused across steps and the curve would not be independent"
                % (name, need_p, need_r + max_obs, need_r, max_obs,
                   idx_pat.size, idx_rnd.size))

        base = idx_rnd[r_perm[-max_obs:]]
        pp = rr = 0
        cx, cy, cp = [], [], []
        for step in range(n_steps + 1):
            proportion = step / n_steps
            n_pat = int(max_obs * proportion)
            n_rnd = max_obs - n_pat
            take_p = idx_pat[p_perm[pp:pp + n_pat]]
            take_r = idx_rnd[r_perm[rr:rr + n_rnd]]
            pp += n_pat
            rr += n_rnd
            if n_pat and n_rnd:
                Xm = np.vstack([Xr[take_p], Xc[take_r]])
            elif n_pat:
                Xm = Xr[take_p]
            else:
                Xm = Xc[take_r]
            d, pv = _emd(Xm, Xc[base], classifier, gene="%s_%d" % (name, step))
            cx.append(proportion)
            cy.append(d)
            cp.append(pv)
        out["curves"][name] = {"x": cx, "y": cy, "pval": cp}
        dep = find_threshold_x(np.array(cx), np.array(cy), out["ceiling"])
        out["departure"][name] = None if dep is None else float(dep)
        if verbose:
            print("    %-13s pools %6d/%6d | EMD %.6f -> %.6f | departs at %s"
                  % (name, idx_pat.size, idx_rnd.size, cy[0], cy[-1],
                     "never" if dep is None else "%.6f" % dep))
    return out


def derive_thresholds(real, ctrl, classifier, rep: Optional[str] = None,
                      method: str = "", stable_pattern: str = STABLE_PATTERN,
                      lenient_exclude: Sequence[str] = LENIENT_EXCLUDE,
                      verbose: bool = True, **kw) -> Dict[str, Any]:
    """The full Fig-4D reading: the curves, the ceiling, and the two landmarks.

    Returns a JSON-serialisable dict. ``lenient`` and ``stringent`` are the two
    thresholds; ``None`` for either means the reading failed for this method,
    which is a finding about the method and must not be silently replaced by a
    default -- ``resolve`` refuses to.
    """
    res = mixture_curves(real, ctrl, classifier, rep=rep, verbose=verbose, **kw)

    dep = res["departure"]
    never = sorted(k for k, v in dep.items() if v is None)
    res["never_departs"] = never
    # `max_diff` in fig4: the EMD above which EVERY subtype has departed from the
    # control ceiling. A subtype that never departs is excluded from the max and
    # named, because averaging it away would understate the lenient cut.
    all_deps = [v for v in dep.values() if v is not None]
    res["lenient_all"] = float(max(all_deps)) if all_deps else None
    res["lenient_argmax_all"] = (max((k for k, v in dep.items() if v is not None),
                                     key=lambda k: dep[k]) if all_deps else None)
    keep = [v for k, v in dep.items() if v is not None and k not in set(lenient_exclude)]
    res["lenient"] = float(max(keep)) if keep else None
    res["lenient_exclude"] = list(lenient_exclude)

    cur = res["curves"].get(stable_pattern)
    st = vb = None
    if cur is not None:
        cx, cy = np.array(cur["x"]), np.array(cur["y"])
        st = find_stable_above(cx, cy, res["ceiling"])
        vb = find_stable_x(cx, cy)                    # the verbatim reading
    res["stringent"] = None if st is None else float(st[1])
    res["stringent_at_proportion"] = None if st is None else float(st[0])
    res["stringent_verbatim"] = None if vb is None else float(vb[1])
    # Flagged rather than silently repaired: True means the verbatim landmark
    # fell below this method's own noise floor and the ceiling guard is what is
    # holding the threshold up. See `find_stable_above`.
    res["stringent_verbatim_below_ceiling"] = bool(
        vb is not None and vb[1] < res["ceiling"])
    res["stable_pattern"] = stable_pattern

    res["method"] = method
    res["latent_dim"] = int(rep_matrix(real, rep).shape[1])
    res["params"] = {"n_steps": kw.get("n_steps", N_STEPS),
                     "max_obs": kw.get("max_obs", MAX_OBS),
                     "min_prop": kw.get("min_prop", MIN_PROP),
                     "mix_prop": kw.get("mix_prop", MIX_PROP),
                     "mix_seed": kw.get("mix_seed", MIX_SEED),
                     "baseline_seed": kw.get("baseline_seed", SEED),
                     "stable_epsilon": STABLE_EPSILON,
                     "stable_pattern": stable_pattern,
                     "lenient_exclude": list(lenient_exclude),
                     "source": "fig4_cells_26_32"}
    if verbose:
        def _f(v):
            return "none" if v is None else "%.6f" % v
        print("  ceiling %.6f | lenient %s (all subtypes %s, argmax %s) | stringent %s%s"
              % (res["ceiling"], _f(res["lenient"]), _f(res["lenient_all"]),
                 res["lenient_argmax_all"], _f(res["stringent"]),
                 "" if not never else "   (never departs: %s)" % ", ".join(never)))
    return res


# --- persistence and use ------------------------------------------------------

def write(res: Dict[str, Any], path) -> Path:
    """Emit one method's derivation, curves included, so it is auditable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1, sort_keys=True))
    tmp.replace(path)
    return path


def read(path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_all(dirpath) -> Dict[str, Dict[str, Any]]:
    """{method: derivation} for every JSON under ``dirpath``."""
    out = {}
    for p in sorted(Path(dirpath).glob("*.json")):
        d = read(p)
        out[d.get("method") or p.stem] = d
    return out


#: Which landmark each dataset's published call uses. The paper does not share
#: one threshold: ileum was run at the lenient cut and APEX at the stringent one
#: (`config.DATASET_THRESHOLDS`, verified there against the saved gene tables).
#: A derived method inherits that choice rather than making its own, which is
#: what makes the comparison per-dataset like-for-like.
DATASET_LANDMARK = {"ileum": "lenient", "apex_hek": "stringent"}


def resolve(derivations: Dict[str, Dict[str, Any]], dataset: str,
            published: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """{method: threshold} for one dataset, from the derived landmarks.

    ``published`` names the methods that keep their published cut instead --
    SubCellSpace, the reference the others are being compared against. Every
    other method takes the landmark this dataset's published call used.
    """
    landmark = DATASET_LANDMARK[dataset]
    out = dict(published or {})
    for name, d in derivations.items():
        if name in out:
            continue
        v = d.get(landmark)
        if v is None:
            raise ValueError(
                "%s has no %s landmark: its cell-edge curve never %s, so a "
                "threshold cannot be read off it. Inspect "
                "artifacts/thresholds/%s.json before scoring it at any cut."
                % (name, landmark, "stabilises" if landmark == "stringent"
                   else "departs from the control ceiling", name))
        out[name] = float(v)
    return out
