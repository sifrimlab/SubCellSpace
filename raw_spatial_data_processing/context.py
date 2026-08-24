"""Runtime state shared across pipeline steps.

The old script threaded a ``tmp`` variable (sometimes a bare ``grid``, sometimes
a ``(grid, adata)`` tuple) from one function to the next, which made the data
flow hard to follow and easy to break. Instead, steps now read what they need
from a single :class:`PipelineContext`, which loads heavy objects *lazily* and
only once:

    * ``ctx.device``  -- the torch device
    * ``ctx.model``   -- the autoencoder, loaded on first use
    * ``ctx.grid``    -- the tile grid, loaded from its pickle on first use
    * ``ctx.adata``   -- the AnnData object, loaded from disk on first use

Because loading is lazy, a step can resume in the middle of the pipeline and the
context will reconstruct whatever upstream state it needs straight from the
checkpoint files on disk.
"""

from __future__ import annotations

import logging
import pickle

import torch
import anndata as ad

from .config import PipelineConfig


class PipelineContext:
    def __init__(self, config: PipelineConfig, logger: logging.Logger | None = None):
        self.config = config
        self.paths = config.paths
        self.log = logger or logging.getLogger("pipeline")

        self._device = None
        self._use_gpu = None
        self._model = None
        self._grid = None
        self._adata = None

    @property
    def use_gpu(self) -> bool:
        """Whether to run cellpose and the encoder on a GPU.

        Auto-detected unless the config says otherwise, so the pipeline runs on a
        CPU-only machine (most macOS/Windows installs) without any config change.
        """
        if self._use_gpu is None:
            want = self.config.use_gpu
            available = torch.cuda.is_available()
            if want is None:
                self._use_gpu = available
                self.log.info(
                    "CUDA %s -- running on %s",
                    "available" if available else "not available",
                    "GPU" if available else "CPU",
                )
            else:
                self._use_gpu = bool(want) and available
                if want and not available:
                    self.log.warning(
                        "use_gpu was requested but no CUDA device is visible; "
                        "falling back to CPU (this will be slow)."
                    )
        return self._use_gpu

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = (
                torch.device(self.config.gpu_id) if self.use_gpu else torch.device("cpu")
            )
        return self._device

    @property
    def model(self):
        if self._model is None:
            self.log.info("Loading model from %s", self.config.model_path)
            # weights_only is left at the torch<2.6 default on purpose: this is a
            # fully pickled model object (with a "cvae" entry), not a state dict.
            self._model = torch.load(self.config.model_path, map_location=self.device)
        return self._model

    @property
    def grid_available(self) -> bool:
        """True if the tile grid can be loaded.

        `is_complete()` runs before the pipeline knows how far it got -- including for
        `--status` on a brand new basedir -- so any step whose completion check counts
        per-tile artifacts must gate on this rather than touching `grid` and raising.
        """
        return self._grid is not None or self.paths.tile_grid_file.is_file()

    @property
    def grid(self):
        if self._grid is None:
            path = self.paths.tile_grid_file
            if not path.is_file():
                raise FileNotFoundError(
                    f"Tile grid not found at {path}. Run the 'build_tile_grid' step first."
                )
            self.log.info("Loading tile grid from %s", path)
            with open(path, "rb") as f:
                self._grid = pickle.load(f)
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    @property
    def adata(self):
        if self._adata is None:
            path = self.paths.adata_file
            if not path.is_file():
                raise FileNotFoundError(
                    f"AnnData not found at {path}. Run the 'build_adata' step first."
                )
            self.log.info("Loading AnnData from %s", path)
            self._adata = ad.read_h5ad(path)
        return self._adata

    @adata.setter
    def adata(self, value):
        self._adata = value
