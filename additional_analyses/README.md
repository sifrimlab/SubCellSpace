# Additional analyses

Supplementary analyses of SubCellSpace: what the embedding encodes, what it can and cannot
resolve, and how sensitive it is to design choices. Each notebook runs the analysis from
cached inputs and ends with a cell that draws its supplementary figure exactly as published.

| notebook | what it measures | figure |
|---|---|---|
| `reconstruction_error.ipynb` | Reconstruction error of the CVAE against PCA codes and the ceiling of its 26×26 decoder; separability of pattern pairs in the input, the reconstruction and the latent; a 500-epoch run showing that reconstruction error does not track representation quality | Supp. Fig. 1 |
| `pattern_confidence_and_recall.ipynb` | Per-class recall on held-out simulations for SubCellSpace and five other representations; why recall on the Xenium/APEX-seq validation is 0.31 (the significance gate caps it at 0.50, and passing it depends on sequencing depth) | Supp. Fig. 2 |
| `rotation_in_latent_space.ipynb` | How image rotation is distributed through the embedding, and how far rotation moves an embedding compared with re-simulation, a different cell or a different pattern | Supp. Fig. 5 |
| `cell_boundary_vs_cytoplasmic.ipynb` | Two additional simulated patterns, cell-boundary and cytoplasmic RNA, embedded with the published model: an eight-class probe and where they land | Supp. Fig. 7 |
| `composite_patterns.ipynb` | Where RNAs combining two patterns land: between their two parents, in proportion to the mixture, except nuclear pairs, which fall into `random` | Supp. Fig. 11 |
| `cell_mask_channel.ipynb` | The same CVAE trained with and without the segmented cell mask as a third input channel, on simulated pattern recovery and on genes called in real data | Supp. Fig. 17 |
| `input_resolution.ipynb` | What the 100×100 input grid costs on the APEX-seq and small-intestine data: cell sizes, transcript displacement, pixel collisions, distance to the nuclear boundary | text only |

The benchmark against Bento, ELLA and alternative encoders (Supp. Fig. 10) is in
[`../benchmark/`](../benchmark/).

## Running

Use the analysis environment from the repository root (`yamls/analysis_figures_crossplatform.yml`,
env name `subcellspace_env`), point `ADDITIONAL_ANALYSIS_DATA` at the downloaded data, and run
the notebooks from this directory:

```bash
export ADDITIONAL_ANALYSIS_DATA=/path/to/additional_analysis
cd additional_analyses
jupyter lab
```

Every notebook runs on a CPU in a few minutes. Figures are written to `figures/<topic>/`, and
the supplementary pages to `figures/supplementary_pages/` (override with
`ADDITIONAL_ANALYSIS_FIGURES`).

Three notebooks (rotation, cell-boundary, reconstruction) contain an encoding step that runs on
a GPU, but only when its cached result is missing from the data folder.

## Data

The cached inputs (about 94 MB) are archived on Zenodo: [DOI to be added]. The download unpacks
to one folder per notebook:

```
additional_analysis/
├── cell_boundary/        latents, UMAP and probe results of the two additional patterns; example cells
├── cell_mask/            probe κ per checkpoint; genes called on real data by the 2- and 3-channel models
├── composite_patterns/   latents and metadata of pure and composite simulations; 22 example images
├── input_resolution/     per-cell sizes, displacement, pixel collisions, nuclear-boundary distances
├── recall/               confusion matrices of the 7-class probe; Supplementary Table 3
├── reconstruction/       per-image reconstruction error, PCA references, pair separability, 500-epoch sweep
└── rotation/             latents of the 44,706 held-out simulated images and their UMAP
```

`MANIFEST.txt` in that folder lists every file with its size and checksum.

## Layout

```
paths.py        where the data is and where figures go
palette.py      the paper's pattern colours, extended to the two additional patterns
supp_style.py   shared style for the supplementary pages
scripts/        the code that produced the cached inputs (see scripts/README.md)
```
