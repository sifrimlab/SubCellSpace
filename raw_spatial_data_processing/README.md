# Processing your own spatial transcriptomics data.

This folder contains a pipeline to go from raw spatial transcriptomics data (either Xenium or MERFISH) to an AnnData object that represents your count matrix, that now also includes subcellular resolution information in the form of a **SubCellSpace** embedding, but also single-molecule coordinates are also stored in .obsm for later use and plotting.


For this, use the [pyxel_env](../yamls/pyxel_env.yaml) yaml, which contains most of the image processing packages. The scripts assume your data follows the standard folder structure of raw Spatial Transcriptomic data, but can be adapted. 

Necessary flags to look at before running the pipeline:
- ```KEEP_UNMATCHED_NUCLEI``` decides whether or not to keep cells that have a succesfull nuclear segmentation, but no cell segmentation to combine it with. In our analyses we discard these as these can be indicative of segmentation artefacts.
- ```EXPAND_SEGMENTATION``` decised with how many pixels the segmentation should be expanded. Sometimes default segmentation udnerestimates the size of cell boundaries.
- ```USE_XENIUM_SEGMENTATION``` decised wheter to use the segmentation already performed by the Xenium instrument, or whether to redo it, which will be (by default) performed on a combination of the DAPI and the intra-RNA staining, which in our experience usually delivers the best results.

Results of the pipeline are a lot of intermediate files, but technically the only import thing is the ```adata_subcell.h5ad```` file, which will be your count matrix including SubCellSpace embeddings. This can then be used to follow the [example notebook](../example_usage_of_SubCellSpace_on_your_data.ipynb) to analyse your own data.
