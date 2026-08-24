"""Command-line entry point for the Xenium -> AnnData pipeline.

Usage (run from the repository root so ``src/`` is importable):

    python -m restructured.run configs/IMU027.yaml            # resume / run
    python -m restructured.run configs/IMU027.yaml --status    # show progress
    python -m restructured.run configs/IMU027.yaml --from segment
    python -m restructured.run configs/IMU027.yaml --force      # rerun everything

By default the pipeline resumes from just after the furthest completed step,
exactly like the original script -- but the resume point is computed from each
step's own checkpoint check, and every step additionally skips per-tile work it
has already finished, so an interrupted run picks up tile-by-tile.

If the run quits due to an error, the traceback (and the failing step) is written
to a separate ``<basedir>_error_<timestamp>.log`` next to the main log, and the
error is re-raised so the process still exits non-zero.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import datetime

from .config import PipelineConfig
from .context import PipelineContext
from .pipeline import Pipeline
from .steps import build_steps, _cleanup_tiled_images


def setup_logging(config: PipelineConfig) -> logging.Logger:
    log_file = f"{config.basedir.name}_log_{datetime.now().strftime('%d:%m:%y')}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    logger.info("Logging to %s", log_file)
    return logger


def write_error_log(config: PipelineConfig, logger: logging.Logger, step: str | None) -> str:
    """Write the active exception's traceback to a separate, timestamped error
    log next to the main log, and point the main log/console at it."""
    err_file = f"{config.basedir.name}_error_{datetime.now().strftime('%d:%m:%y_%H:%M:%S')}.log"
    with open(err_file, "w") as f:
        f.write(f"Pipeline run failed at {datetime.now().isoformat()}\n")
        f.write(f"Base directory: {config.basedir}\n")
        f.write(f"Failing step:   {step or 'unknown'}\n\n")
        f.write(traceback.format_exc())
    logger.error("Run FAILED in step '%s' -- traceback written to %s", step or "unknown", err_file)
    return err_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to the experiment YAML config file")
    parser.add_argument("--status", action="store_true",
                        help="Print which steps are complete and exit")
    parser.add_argument("--from", dest="start_from", metavar="STEP",
                        help="Force the run to (re)start from this step name")
    parser.add_argument("--force", action="store_true",
                        help="Rerun every step from the beginning")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    config = PipelineConfig.from_yaml(args.config)
    logger = setup_logging(config)
    ctx = PipelineContext(config, logger)
    pipeline = Pipeline(build_steps())

    if args.status:
        pipeline.status(ctx)
        return

    try:
        pipeline.run(ctx, force=args.force, start_from=args.start_from)

        # Free the regenerable tiled images once the pipeline is actually complete
        # (unless the user wants to keep them for diagnostics / visualization).
        if not config.keep_tiled_images and pipeline.steps[-1].is_complete(ctx):
            _cleanup_tiled_images(ctx)
    except Exception:
        # Record the failure in its own log, then re-raise so the process still
        # exits non-zero and the traceback is visible on the console.
        write_error_log(config, logger, pipeline.current_step)
        raise


if __name__ == "__main__":
    main()
