# Everything the benchmark needs to find, and the methods it compares.
# Point DATA_ROOT at the downloaded data bundle and nothing else needs editing.

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The data bundle (see README): https://doi.org/10.5281/zenodo.22897067
# together with the four large files from 10.5281/zenodo.19708995.
# Override the location with the BENCHMARK_DATA env var.
DATA_ROOT = Path(os.environ.get("BENCHMARK_DATA", "/mnt/data/david/benchmarks/benchmark_data"))

DATASETS = DATA_ROOT / "datasets"
EMBEDDINGS = DATA_ROOT / "embeddings"
GENE_TABLES = DATA_ROOT / "gene_tables"
MODELS = DATA_ROOT / "models"
CONTROLS = DATA_ROOT / "controls"
THRESHOLDS = DATA_ROOT / "thresholds"

# written by the notebooks
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
INFO_FILES = HERE / "info_files"

RESULTS_FILE = RESULTS / "benchmark_results.csv.gz"

seed = 101
min_obs_per_gene = 11          # from the power analysis; genes below this are not tested

# The two real datasets, and the shuffled control each is scored against.
REAL_DATASETS = {
    "ileum": DATASETS / "ileum.h5ad",
    "apex_hek": DATASETS / "apex_hek.h5ad",
}

SIMULATED_VALIDATION = DATASETS / "simulated_validation.h5ad"
SIMULATED_PATTERNED = DATASETS / "simulated_patterned.h5ad"      # artifact A, real arm
SIMULATED_SHUFFLED = DATASETS / "simulated_shuffled.h5ad"        # artifact A, shuffled arm

SUBCELLSPACE_MODEL = MODELS / "SubCellSpace"
PATTERN_CLASSIFIER = MODELS / "rf_pattern_nonpattern.pkl"
SUBTYPE_CLASSIFIER = MODELS / "multiclass_rf.pkl"

# Methods, left to right as they appear in every table and figure. Two blocks:
# the comparison against other people's tools first, then our own alternative
# implementations. Point-VAE is ours too -- a completely different model, but our
# creation; had it come out ahead we would have published that instead.
METHODS = [
    dict(key="subcellspace", name="SubCellSpace", block="reference",
         note="convolutional VAE, trained jointly against a reconstruction loss "
              "and two auxiliary pattern classifiers"),
    dict(key="bento_features", name="Bento", block="tool",
         note="13 spatial features fed to the published RNAforest classifier"),
    dict(key="ella", name="ELLA", block="tool",
         note="radial non-homogeneous Poisson test, per gene, so it has no embedding"),
    dict(key="pointnet_vae_64", name="Point-VAE", block="ours",
         note="VAE over transcript coordinates as a point cloud, 64-d"),
    dict(key="cnn_baseline", name="CNN baseline", block="ours",
         note="same pipeline, convolutional encoder, no variational term"),
    dict(key="image_transformer_small", name="Image transformer (small)", block="ours",
         note="same pipeline, ViT encoder trained from scratch, 3.8M parameters"),
    dict(key="image_transformer_large", name="Image transformer (large)", block="ours",
         note="same pipeline, pretrained ViT-B/16 encoder, 86M parameters"),
]

method_order = [m["key"] for m in METHODS]
method_names = {m["key"]: m["name"] for m in METHODS}
method_block = {m["key"]: m["block"] for m in METHODS}

# where the table splits: everything from here on is one of our own alternatives
first_alternative = next(m["key"] for m in METHODS if m["block"] == "ours")
competitors = [m["key"] for m in METHODS if m["block"] in ("reference", "tool")]
alternatives = [m["key"] for m in METHODS if m["block"] == "ours"]

# one colour per method, fixed so it never moves between figures
palette = {
    "subcellspace": "#2a78d6",
    "bento_features": "#eda100",
    "ella": "#e87ba4",
    "pointnet_vae_64": "#eb6834",
    "cnn_baseline": "#1baf7a",
    "image_transformer_small": "#4a3aa7",
    "image_transformer_large": "#e34948",
}

# ELLA reports a p-value per gene rather than an embedding, so it only appears
# where a gene-level test makes sense.
def applies_to(method_key, dataset):
    if method_key == "ella":
        return dataset in ("ileum", "apex_hek")
    return True


# Files that live at the top level of the SubCellSpace data record
# (zenodo.org/records/19708995) under a different name, rather than inside the
# benchmark's archive. Whoever assembles the bundle copies them in; the bundle
# builder can leave them out with --upload-only when preparing that archive.
ALREADY_DEPOSITED = {
    "datasets/apex_hek.h5ad":
        "hek293T_Xenium_adata_subcell_including_randomized_embeddings.h5ad",
    "datasets/ileum.h5ad":
        "merfish_SI_adata_subcell_including_randomized_embeddings.h5ad",
    "datasets/simulated_shuffled.h5ad":
        "simulated_adata_all_proportions_only_test_randomized.h5ad",
    "datasets/simulated_validation.h5ad":
        "simulated_adata_unbalanced_traintestsplit_with_umap.h5ad",
    # added alongside the benchmark: the record held only the randomized half
    "datasets/simulated_patterned.h5ad":
        "simulated_adata_all_proportions_only_test.h5ad",
}


def embedding_file(method_key, arm):
    # arm is one of: patterned, shuffled, validation,
    #                ileum_real, ileum_control, apex_hek_real, apex_hek_control
    return EMBEDDINGS / method_key / f"{arm}.npz"


def gene_table(method_key, dataset, arm):
    return GENE_TABLES / method_key / f"{dataset}_{arm}.csv"
