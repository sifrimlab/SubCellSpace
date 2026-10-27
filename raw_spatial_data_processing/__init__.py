"""Restructured Xenium -> AnnData (+ subcellular embedding) pipeline.

This package is a clean, resumable reimplementation of the old
``processXeniumIntoSingleGeneImages_template.py`` monolith. The behaviour of
each processing step is preserved; only the structure changed:

    config.py    parameters for one experiment + all derived paths
    context.py   lazily-loaded runtime state (tile grid, AnnData, model, device)
    pipeline.py  the Step abstraction and the resume-aware runner
    steps.py     the eleven concrete pipeline steps
    run.py       command-line entry point

The original ``src/`` helper modules are reused unchanged, so we make sure the
repository root is importable no matter where the pipeline is launched from.
"""

import sys
from pathlib import Path

# The trained model is a fully pickled nn.Module, so unpickling it requires the
# top-level `CVAE` module (repo root) to be importable. `src` is now a subpackage,
# so this is the only reason the shim remains.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
