# SubCellSpace: A latent space of the subcellular mRNA localization patterns

This repository holds all code necessary to reproduce the results en figures of Wouters et al. 2026.

Each folder has its own purpose, and also its own environment, which can be created using the [yamls](./yamls/).

- To recreate the making of training data and training SubCellSpace, follow the steps in [this folder](./training_subcellspace/).

- To reproduce the analysis and figure creation of the manuscript, follow the steps in [this folder](./make_paper_figures/).

- To process your own dataset and create an AnnData object with a SubCellSpace layer, follow the steps in [this folder](./raw_spatial_data_processing/).

## Workflow

In general, the only thing necessary to run any section is a conda environment, create for example as:

```bash
conda env create -n subcellspace -f ./yamls/analysis_figures.yml
```
A docker container is also available if that is prefered, which can be run using:





