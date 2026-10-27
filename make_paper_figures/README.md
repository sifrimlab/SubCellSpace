# Reproducing the figures.

To reproduce the figures, create the conda environment from
[`analysis_figures_crossplatform.yml`](../yamls/analysis_figures_crossplatform.yml):

```bash
conda env create -f ../yamls/analysis_figures_crossplatform.yml
conda activate subcellspace_env
```

> The original [`analysis_figures.yml`](../yamls/analysis_figures.yml) is a `conda env export`
> from a Linux + CUDA machine: all 118 of its conda packages carry linux-64 build strings, six
> are Linux-only, and it pins the CUDA 11.6 toolkit from the `pytorch`/`nvidia` channels, so it
> **will not solve on macOS or Windows**. It is kept as an exact record of the environment the
> figures were produced in. The cross-platform file keeps every package version from it except
> three — Python 3.8 → 3.9 (3.8 is end-of-life), `torch`/`torchvision` from PyPI at the same
> 1.13.1/0.14.1, and `faiss-cpu` 1.7.0 → 1.7.4 (1.7.0 was never published to PyPI; it is used
> only for deterministic exact-L2 matching, so results are unchanged). It was verified to
> resolve on linux-x86_64, macOS-arm64, macOS-x86_64 and win-amd64.

All notebooks work standalone, the only thing necessary besides a notebook and the *conda_env* is processed data, which can be found on [https://zenodo.org/records/19708995](19708995).   

> **One exception: Fig. 1B.** The cell under "Figure 1B" in `fig2_technical_aspects_of_subcellspace.ipynb`
> simulates its example patterns live with `simfish`, which needs the simulation environment
> ([`make_simulations.yml`](../yamls/make_simulations.yml), Linux only, Python 3.6). Every other cell runs in
> `subcellspace_env`.

If wanting to make use of the interactive features of SubCellSpace (e.g.: clicking on observations in the latent space and plotting them, drawing your own patterns...), jupyter nbclassic is prefered. Technically modern jupyterlab interactive modes should work, but I've had bad experiences with them.


