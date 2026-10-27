"""revbench -- benchmarking harness for SubCellSpace.

    import revbench as rb
    rep  = rb.from_adata(adata, "obsm:latent", name="subcellspace")
    df   = rb.run_suite(adata, [rep], dataset="simulated", smoke=True)
    rb.summarize(rb.load_all())
"""
from .controls import (ControlSet, check_control_consistency,
                       extract_from_coord_tree, extract_from_h5ad,
                       load_control, render_image, simulated_metadata,
                       to_pointcloud_parquet)
from .config import (DISTANCE_THRESHOLD, MIN_OBS_PER_GENE, SEED,
                     derive_metadata_columns)
from .registry import BENCHMARKS, applicable, register
from .representations import (Representation, attach, common_valid, from_adata,
                              from_dump, read_dump, standalone, write_dump)
from .results import (MetricRow, compact, load_all, summarize)
from .suite import BenchmarkContext, run_suite

__all__ = [
    "ControlSet", "extract_from_h5ad", "extract_from_coord_tree", "load_control",
    "render_image", "to_pointcloud_parquet", "check_control_consistency",
    "simulated_metadata",
    "SEED", "DISTANCE_THRESHOLD", "MIN_OBS_PER_GENE", "derive_metadata_columns",
    "Representation", "from_adata", "from_dump", "standalone", "attach",
    "common_valid", "write_dump", "read_dump",
    "MetricRow", "load_all", "summarize", "compact",
    "BENCHMARKS", "register", "applicable",
    "BenchmarkContext", "run_suite",
]
