# Producing the embeddings

`embeddings/` in the data bundle is shipped ready-made. These scripts are how it was produced, kept
here so the step is inspectable.

**They are not turnkey.** Each one runs in the environment of the tool it drives, and between them
they need the raw simulation trees (several hundred GB), the two real h5ads, a GPU, and three
things we cannot redistribute: the Point-VAE repository and its checkpoint, Bento's shipped
classifier pickle, and the ELLA package. They also still resolve their inputs through the layout
they were written against rather than through `config.py`, so paths at the top of each script need
setting before anything runs.

| script | environment | produces |
|---|---|---|
| `train_cnn_baseline.py`, `encode_cnn_baseline.py`, `encode_cnn_real.py` | torch 1.13 | the CNN baseline |
| `train_vit_cvae.py`, `encode_vit_reduce.py` | torch 2.x + transformers | both image transformers |
| `bento_geometry_*.py`, `bento_score_*.py` | bento-tools, xgboost 2.0.3 | Bento's 13 features |

`build_data_bundle.py` is the exception and does work as-is: it collects everything the benchmark
reads into the bundle layout `config.py` expects.

The SubCellSpace embeddings are not produced here — they are stored in the datasets themselves.
ELLA's per-gene tables come from ELLA's own command-line tool.
