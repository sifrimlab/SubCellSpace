# Where the cached inputs live and where figures are written.
# Point DATA_ROOT at the downloaded data bundle and nothing else needs editing.
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HERE = REPO / "additional_analyses"

# Override the locations with the ADDITIONAL_ANALYSIS_DATA / ADDITIONAL_ANALYSIS_FIGURES env vars.
DATA_ROOT = Path(os.environ.get("ADDITIONAL_ANALYSIS_DATA",
                                "/mnt/data/david/subcellspace_data/additional_analysis"))
FIGURES = Path(os.environ.get("ADDITIONAL_ANALYSIS_FIGURES", HERE / "figures"))

MODEL = REPO / "SubCellSpace"                 # the published 2-channel checkpoint

# revbench (encoding, metadata parsing, the random-forest probe) ships with the benchmark
BENCHMARK = REPO / "benchmark"
for p in (str(REPO), str(BENCHMARK), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def data(*parts):
    return DATA_ROOT.joinpath(*parts)


def figures(topic):
    out = FIGURES / topic
    out.mkdir(parents=True, exist_ok=True)
    return out
