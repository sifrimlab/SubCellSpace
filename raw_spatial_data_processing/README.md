# Processing your own spatial transcriptomics data.

This folder contains a pipeline to go from raw spatial transcriptomics data (either Xenium or MERFISH) to an AnnData object that represents your count matrix, that now also includes subcellular resolution information in the form of a **SubCellSpace** embedding, but also single-molecule coordinates are also stored in .obsm for later use and plotting.


For this, use the [pyxel_env](../yamls/pyxel_env.yaml) yaml, which contains most of the image processing packages. The scripts assume your data follows the standard folder structure of raw Spatial Transcriptomic data, but can be adapted. 

> **On macOS or Windows, use [`pyxel_env_crossplatform.yml`](../yamls/pyxel_env_crossplatform.yml) instead.**
> `pyxel_env.yaml` is a `conda env export` from a Linux machine: every conda package carries a
> linux-64 build string, six are Linux-only (`libgcc-ng`, `libgomp`, `libstdcxx-ng`, …), and the
> pip section pins CUDA wheels plus `triton`, which is published for Linux only and has no
> source distribution. The cross-platform file pins the same tested versions of the packages
> this pipeline actually imports, and has been verified to resolve on linux-x86_64,
> macOS-arm64, macOS-x86_64 and win-amd64:
>
> ```bash
> conda env create -f ../yamls/pyxel_env_crossplatform.yml
> conda activate pyxel_env
> ```
>
> Alternatively, skip environment setup entirely and use the prebuilt Docker image, which
> already contains this environment plus the pipeline code and the trained model:
>
> ```bash
> docker run -it --rm -p 8888:8888 -v /path/to/your/data:/data woutdavid/subcellspace:latest
> ```
>
> See the [top-level README](../README.md#option-2--docker) for the available tags and for the
> note on Apple Silicon.
>
> If you have no CUDA GPU, pass `gpu=False` to `segment_dual_channel` / `segment_dapi`.

Necessary flags to look at before running the pipeline:
- ```KEEP_UNMATCHED_NUCLEI``` decides whether or not to keep cells that have a succesfull nuclear segmentation, but no cell segmentation to combine it with. In our analyses we discard these as these can be indicative of segmentation artefacts.
- ```EXPAND_SEGMENTATION``` decides with how many pixels the segmentation should be expanded. Sometimes default segmentation udnerestimates the size of cell boundaries.
- ```USE_XENIUM_SEGMENTATION``` decides wheter to use the segmentation already performed by the Xenium instrument, or whether to redo it, which will be (by default) performed on a combination of the DAPI and the intra-RNA staining, which in our experience usually delivers the best results.

Results of the pipeline are a lot of intermediate files, but technically the only import thing is the ```adata_subcell.h5ad``` file, which will be your count matrix including SubCellSpace embeddings. This can then be used to follow the [example notebook](../example_usage_of_SubCellSpace_on_your_data.ipynb) to analyse your own data.
