"""The Step abstraction and the resume-aware pipeline runner.

The old script decided where to resume with a parallel ``order_bool_list`` of
``Path.is_file()`` checks plus some index arithmetic, and then re-derived
``grid``/``adata`` with ad-hoc ``if to_do_index > N`` guards. That logic was
correct but extremely easy to get out of sync when adding a step.

Here, each step owns its own completion check. A step is a small object with:

    * ``name``        -- stable identifier (used by ``--from`` / ``--status``)
    * ``description`` -- one-line human summary
    * ``is_complete`` -- True when its checkpoint already exists on disk
    * ``run``         -- do the work (idempotent: every step also skips work it
                         has already done at the per-tile level)

The :class:`Pipeline` figures out the furthest point reached on disk and runs
everything after it -- the same "resume from furthest progress" semantics as the
original, but derived from the steps themselves rather than a separate list.
"""

from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .context import PipelineContext

try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:  # psutil is only needed for the optional resource profiling
    _HAVE_PSUTIL = False


def _format_duration(seconds: float) -> str:
    """Human-readable duration: '4.2s', '3m07s', '1h02m30s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _format_bytes(n) -> str:
    """Human-readable byte count; 'n/a' for None."""
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _dir_size(path) -> int:
    """Total size in bytes of all files under ``path`` (0 if it doesn't exist)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class _ResourceSampler:
    """Sample process-tree RSS in a background thread while a step runs.

    Sums the parent and all child processes (so the joblib workers in the
    parallel steps are counted). A no-op when disabled or psutil is missing.
    """

    def __init__(self, enabled: bool = True, interval: float = 0.5):
        self.enabled = enabled and _HAVE_PSUTIL
        self.interval = interval
        self._samples: List[int] = []
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _tree_rss() -> int:
        proc = psutil.Process()
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                pass
        return rss

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self._samples.append(self._tree_rss())
            except Exception:
                pass

    def __enter__(self):
        if self.enabled:
            try:
                self._samples.append(self._tree_rss())
            except Exception:
                pass
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return False

    @property
    def avg(self):
        return sum(self._samples) / len(self._samples) if self._samples else None

    @property
    def peak(self):
        return max(self._samples) if self._samples else None


class PipelineStep(ABC):
    #: stable identifier, e.g. "segment"
    name: str = ""
    #: one-line human-readable summary
    description: str = ""

    @abstractmethod
    def is_complete(self, ctx: PipelineContext) -> bool:
        """Return True if this step's output checkpoint already exists."""

    @abstractmethod
    def run(self, ctx: PipelineContext) -> None:
        """Perform the work. Must be safe to re-run."""


class Pipeline:
    def __init__(self, steps: List[PipelineStep]):
        self.steps = list(steps)
        self.current_step: Optional[str] = None  # name of the step in flight

    def index_of(self, name: str) -> int:
        for i, step in enumerate(self.steps):
            if step.name == name:
                return i
        raise KeyError(f"Unknown step {name!r}. Known steps: {[s.name for s in self.steps]}")

    def resume_index(self, ctx: PipelineContext) -> int:
        """Index of the first step to run: just after the furthest completed step."""
        completed = [i for i, s in enumerate(self.steps) if s.is_complete(ctx)]
        return max(completed) + 1 if completed else 0

    def plan(
        self,
        ctx: PipelineContext,
        *,
        force: bool = False,
        start_from: Optional[str] = None,
    ) -> Tuple[int, List[PipelineStep]]:
        if force:
            start = 0
        elif start_from is not None:
            start = self.index_of(start_from)
        else:
            start = self.resume_index(ctx)
        return start, self.steps[start:]

    def status(self, ctx: PipelineContext) -> None:
        self.log_overview(ctx, start=self.resume_index(ctx))

    def log_overview(self, ctx: PipelineContext, start: int) -> None:
        ctx.log.info("Pipeline steps (resuming at #%d):", start + 1)
        for i, step in enumerate(self.steps):
            if i < start:
                marker = "skip"
            elif i == start:
                marker = "NEXT"
            else:
                marker = "todo"
            ctx.log.info("  [%-4s] %2d. %-22s %s", marker, i + 1, step.name, step.description)

    def run(
        self,
        ctx: PipelineContext,
        *,
        force: bool = False,
        start_from: Optional[str] = None,
    ) -> None:
        start, to_run = self.plan(ctx, force=force, start_from=start_from)
        self.log_overview(ctx, start=start)

        if not to_run:
            ctx.log.info("Nothing to do -- pipeline already complete.")
            return

        profile = getattr(ctx.config, "profile_resources", False)
        out_dir = ctx.paths.out_dir

        records = []  # (name, dt, avg_mem, peak_mem, storage_delta)
        overall_t0 = time.perf_counter()
        for step in to_run:
            self.current_step = step.name
            ctx.log.info("=== START  %s -- %s", step.name, step.description)
            size_before = _dir_size(out_dir) if profile else 0
            t0 = time.perf_counter()
            with _ResourceSampler(enabled=profile) as sampler:
                step.run(ctx)
            dt = time.perf_counter() - t0
            storage_delta = (_dir_size(out_dir) - size_before) if profile else None
            records.append((step.name, dt, sampler.avg, sampler.peak, storage_delta))
            ctx.log.info("=== DONE   %s (%s)", step.name, _format_duration(dt))
        total = time.perf_counter() - overall_t0

        self.current_step = None
        ctx.log.info("All done! Timing summary:")
        for name, dt, *_ in records:
            ctx.log.info("  %-22s %8s", name, _format_duration(dt))
        ctx.log.info("  %-22s %8s", "TOTAL (end-to-end)", _format_duration(total))

        if profile:
            self._log_resource_summary(ctx, records, out_dir)

    def _log_resource_summary(self, ctx, records, out_dir) -> None:
        ctx.log.info("Resource summary (process-tree memory; storage added to %s):", out_dir.name)
        if not _HAVE_PSUTIL:
            ctx.log.info("  (memory profiling unavailable -- psutil not installed)")
        for name, _dt, avg, peak, storage in records:
            ctx.log.info("  %-22s avg %9s  peak %9s  +storage %9s",
                         name, _format_bytes(avg), _format_bytes(peak), _format_bytes(storage))
        peaks = [r[3] for r in records if r[3] is not None]
        ctx.log.info("  %-22s peak %9s  total    %9s", "OVERALL",
                     _format_bytes(max(peaks) if peaks else None), _format_bytes(_dir_size(out_dir)))
