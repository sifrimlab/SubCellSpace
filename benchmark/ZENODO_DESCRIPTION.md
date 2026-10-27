## Benchmark data for SubCellSpace: comparison against other subcellular localization methods

**Author:** David Wouters

**Associated publication:** Wouters et al. 2026 (under review)

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

**Cite as:** Wouters, D. Benchmark data for SubCellSpace. Zenodo. https://doi.org/10.5281/zenodo.22897067

### Overview

This record contains the data needed to reproduce the method benchmark accompanying Wouters et al.
2026, in which SubCellSpace is compared against Bento and ELLA, and against four alternative
implementations of our own. The code that reads it is at `benchmark/` in the
SubCellSpace repository.

It is a companion to the main data record, **SubCellSpace: Automated characterization of
subcellular mRNA localization patterns** (https://doi.org/10.5281/zenodo.19708995), and does not duplicate
the four large files already deposited there. You need both records to run the benchmark; the
assembly table below says which file comes from where.

### Files in this record

| File | Description |
|------|-------------|
| benchmark_data.tar.gz | The benchmark's data directory. Unpacks to the structure shown below. Deposited as a single archive because it holds 91 files nested one directory deep per method, whose names are only unique within those directories — Zenodo does not preserve folder structure, and flattening them would collide. |
| simulated_adata_all_proportions_only_test.h5ad | AnnData object containing simulated data of a different simulation iteration, focused on different proportions of pattern-presence. This is the patterned half of a matched pair; the randomized half is `simulated_adata_all_proportions_only_test_randomized.h5ad` in the main record. Every benchmarked method's classifier is fitted on the pair. |

### What is inside the archive

| Directory | Contents |
|------|-------------|
| `embeddings/` | One directory per method, one file per arm: the per-observation output each method produces for the simulated training pair, the simulated validation split, and both real datasets. These are shipped ready-made because producing them requires each tool's own software environment, and those environments cannot coexist in one interpreter. |
| `models/` | The SubCellSpace checkpoint and the two RandomForest classifiers published with it. |
| `controls/` | The randomly repositioned transcript sets each dataset is scored against, so every method is compared to the same null. |
| `thresholds/` | Each method's earth-mover distance cut-off, derived on simulated data only. |
| `gene_tables/` | ELLA's per-gene output, which is what it reports instead of an embedding. |
| `datasets/` | The canonical observation order for the validation split, and the dataset sizes the notebooks assert against. |

### Assembling the data directory

Unpack `benchmark_data.tar.gz`, then copy five files into `benchmark_data/datasets/` under the
names the code expects. One comes from this record, four from the main data record.

| File | From | Copy to |
|------|------|---------|
| simulated_adata_all_proportions_only_test.h5ad | this record | `datasets/simulated_patterned.h5ad` |
| hek293T_Xenium_adata_subcell_including_randomized_embeddings.h5ad | 10.5281/zenodo.19708995 | `datasets/apex_hek.h5ad` |
| merfish_SI_adata_subcell_including_randomized_embeddings.h5ad | 10.5281/zenodo.19708995 | `datasets/ileum.h5ad` |
| simulated_adata_all_proportions_only_test_randomized.h5ad | 10.5281/zenodo.19708995 | `datasets/simulated_shuffled.h5ad` |
| simulated_adata_unbalanced_traintestsplit_with_umap.h5ad | 10.5281/zenodo.19708995 | `datasets/simulated_validation.h5ad` |

Then point the code at the directory:

```bash
export BENCHMARK_DATA=/path/to/benchmark_data
```

and run the notebooks in order: `1_simulated.ipynb`, `2_real_data.ipynb`, `3_summary.ipynb`.

The summary notebook ships with its own results table in the repository, so it reproduces every
figure and table without downloading anything. The two stage notebooks are what need this data.

### Data Format

All `.h5ad` files are AnnData objects and can be loaded in Python using the `anndata` package:

```python
import anndata as ad
adata = ad.read_h5ad("filename.h5ad")
```

The `.npz` embedding files are read with `numpy.load`; each holds the matrix `X` and the matching
observation keys `obs_index`. The `.parquet` files are read with `pandas.read_parquet`.

### Directory structure of the archive

```
benchmark_data/
  datasets/
    validation_order.parquet          canonical row order of the validation split
    scale.json                        the counts the notebooks assert against
    (the five h5ads above are copied in here)
  embeddings/
    bento_features/                   each holds seven arms:
    pointnet_vae_64/                    patterned, shuffled, validation,
    cnn_baseline/                       ileum_real, ileum_control,
    image_transformer_small/            apex_hek_real, apex_hek_control
    image_transformer_large/
                                      SubCellSpace has no directory here: its
                                      embeddings are stored in the h5ads themselves
  models/
    SubCellSpace                      the published checkpoint
    rf_pattern_nonpattern.pkl         the published pattern/no-pattern classifier
    multiclass_rf.pkl                 the published subtype classifier
  controls/
    control_apex_hek.parquet
    control_ileum.parquet
    control_patterned.parquet
    control_shuffled.parquet
  gene_tables/
    ella/                             per-gene output for both real datasets
  thresholds/
    one JSON per method
```

### Contact

David Wouters — david.wouters@kuleuven.be
Alejandro Sifrim — alejandro.sifrim@kuleuven.be
