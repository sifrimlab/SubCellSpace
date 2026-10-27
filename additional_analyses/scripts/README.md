# How the cached inputs were made

These scripts produced the files in the data folder that the notebooks read. They are here so
that the step is inspectable, not as a turnkey path: they read the original simulation tree and
model checkpoints at the paths of the original run, several need a GPU, and some import helper
modules that are not part of this repository. Each script starts with a header saying what it
produces. `/path/to/work` stands for the working directory the intermediates were written to;
set it to a directory of your own before running a script.

| folder | script | produces |
|---|---|---|
| `cell_boundary/` | `simulate_cytoplasm.py` | the cell-boundary and cytoplasmic simulations on the 64 held-out cell templates |
| | `step3_separation.py` | `combined.npz` (latents of the held-out and additional simulations) and the eight-class probe results |
| `cell_mask/` | `train_long_2ch_3ch.py` | the 2- and 3-channel models at 100 and 300 epochs |
| | `encode_and_score.py` | `epoch_sweep_raw.csv` (pattern-probe κ per checkpoint) |
| | `encode_matched_reduce.py` | real-data embeddings of the two 300-epoch models |
| | `score_matched_detection.py` | `matched_detection_raw.csv` (genes called at the published thresholds) |
| `composite_patterns/` | `extract_and_encode_composites.py` | `latents.npy` and `metadata.csv` of the pure and composite simulations |
| `input_resolution/` | `measure_input_resolution.py` | every table in `input_resolution/`, from the APEX-seq and small-intestine h5ads of the main data record ([10.5281/zenodo.19708995](https://doi.org/10.5281/zenodo.19708995)) |
| | `extract_example_cell.py` | `example_cell_apex.npz`, the one cell drawn in the notebook |
| `reconstruction/` | `train_long_2ch_3ch.py` | the 500-epoch 2-channel run used for the sweep (same script as in `cell_mask/`) |
| | `measure_reconstruction.py` | the reconstruction error, PCA references, pair separability and sweep tables |

The rotation latents and their UMAP are rebuilt by `rotation_in_latent_space.ipynb` itself when
they are missing. The confusion matrices in `recall/` were computed with
`revbench.detection.classifier.train_subtype_classifier(qc=True)` (three folds grouped by cell)
on each representation's embedding of the simulated training pair, and Supplementary Table 3 is
exported from `make_paper_figures/fig4_classifying_and_quantifying_degrees_of_patterning.ipynb`.
