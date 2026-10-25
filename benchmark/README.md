# Benchmarking SubCellSpace

This directory holds the benchmark reported in the paper: SubCellSpace against six alternatives, on
the same workflow, in two stages.

- **Stage 1** — how well does a trained model perform on simulated data, where the pattern of every
  observation is known by construction.
- **Stage 2** — how well does it perform on real data, against published ground truth: a MERFISH
  small intestine section and APEX-seq HEK293.

| notebook | what it does | needs |
|---|---|---|
| `3_summary.ipynb` | every table and figure in the paper's benchmark section | nothing but this directory |
| `1_simulated.ipynb` | runs stage 1 | the data bundle |
| `2_real_data.ipynb` | runs stage 2 and writes the results file | the data bundle |

`3_summary.ipynb` reads `results/benchmark_results.csv.gz`, which is included here, so it runs in a
few seconds without downloading anything. The two stage notebooks regenerate that file.

## The methods

Every table is split the same way. Left of the divider, SubCellSpace against the two other
published tools. Right of it, our own alternative implementations of the same idea — Point-VAE
takes a completely different approach, and the other three share the SubCellSpace pipeline exactly
and differ only in the encoder.

| method | what it is |
|---|---|
| **SubCellSpace** | convolutional VAE, encoder trained jointly against a reconstruction loss and two auxiliary pattern classifiers |
| Bento | 13 spatial features fed to the published RNAforest classifier |
| ELLA | radial non-homogeneous Poisson test; reports per gene, so it has no embedding |
| *Point-VAE* | ours — a VAE over transcript coordinates as a point cloud, 64-d |
| *CNN baseline* | ours — same pipeline, convolutional encoder, no variational term |
| *Image transformer (small)* | ours — same pipeline, ViT encoder trained from scratch, 3.8M parameters |
| *Image transformer (large)* | ours — same pipeline, pretrained ViT-B/16 encoder, 86M parameters |

## Running it

Everything reads from one directory. Point `BENCHMARK_DATA` at the download, or edit `DATA_ROOT`
in `config.py`:

```bash
export BENCHMARK_DATA=/path/to/benchmark_data
jupyter lab
```

Then `1_simulated.ipynb` → `2_real_data.ipynb` → `3_summary.ipynb`. Stage 1 takes a few hours;
stage 2 is dominated by APEX-seq at 8.4M gene-cell pairs.

The environment is `envs/benchmark.yml`:

```bash
mamba env create -f envs/benchmark.yml
```

Use mamba, or conda 23.10 and later where libmamba is the default solver — the
classic solver does not finish on this many exact pins.

## The data bundle

Everything the notebooks read lives in one directory, assembled from **two** Zenodo records:

- the benchmark record, [10.5281/zenodo.22897067](https://doi.org/10.5281/zenodo.22897067)
- the main SubCellSpace data record, [10.5281/zenodo.19708995](https://doi.org/10.5281/zenodo.19708995),
  whose four large files are not duplicated here

```
benchmark_data/
├── datasets/        the two real h5ads, the simulated split, the simulated training pair
├── embeddings/      one directory per method, one file per arm
├── gene_tables/     ELLA's per-gene output
├── models/          the SubCellSpace checkpoint and its two classifiers
├── controls/        the randomly repositioned transcripts each dataset is scored against
└── thresholds/      each method's distance cut-off, derived on simulated data
```

Unpack `benchmark_data.tar.gz` from the benchmark record, then copy these five into `datasets/`:

| file | from | copy to |
|---|---|---|
| `simulated_adata_all_proportions_only_test.h5ad` | [10.5281/zenodo.22897067](https://doi.org/10.5281/zenodo.22897067) | `simulated_patterned.h5ad` |
| `hek293T_Xenium_adata_subcell_including_randomized_embeddings.h5ad` | zenodo.19708995 | `apex_hek.h5ad` |
| `merfish_SI_adata_subcell_including_randomized_embeddings.h5ad` | zenodo.19708995 | `ileum.h5ad` |
| `simulated_adata_all_proportions_only_test_randomized.h5ad` | zenodo.19708995 | `simulated_shuffled.h5ad` |
| `simulated_adata_unbalanced_traintestsplit_with_umap.h5ad` | zenodo.19708995 | `simulated_validation.h5ad` |

`cache/` is created inside the bundle while running and holds derived intermediates; it is not part
of either download.

### Assembling it from the original locations

```bash
# the full, runnable bundle
python scripts/build_data_bundle.py --out /path/to/benchmark_data --copy

# only what is not already in the data record -- this is what gets archived and
# uploaded, and is not runnable on its own
python scripts/build_data_bundle.py --out /path/to/benchmark_data_upload --copy --upload-only
tar czf benchmark_data.tar.gz -C /path/to benchmark_data_upload

# file list with sizes and checksums, to verify an upload against
python scripts/zenodo_manifest.py --bundle /path/to/benchmark_data --checksums --out MANIFEST.txt
```

Without `--copy` it symlinks instead, which is what you want for a local run.
`ZENODO_DESCRIPTION.md` is the record description to go with the upload.

## Regenerating the embeddings

`embeddings/` is shipped because producing it needs each tool's own environment: Bento, ELLA and
Point-VAE will not coexist in one interpreter, and the transformer variants need a GPU. The scripts
that produced it are under `scripts/`, with their own README — they are there so the step is
inspectable, not as a turnkey path, and `scripts/README.md` says what each one needs.

## Layout

```
config.py     where the data is, and the list of methods
loading.py    reading datasets and embeddings
scoring.py    running stage 1 and stage 2
tables.py     the tables and figures
revbench/     the benchmark library
vendor/       published functions this reuses, copied in so the directory stands alone
```
