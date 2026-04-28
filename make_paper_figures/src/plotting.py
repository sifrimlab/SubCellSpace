from pathlib import Path
import matplotlib.pyplot as plt
from .adata_utils import convertRecordToDf

FIG_OUT_DIR = Path("/mnt/data/david/plots/manuscript/")
PATTERN_NAMES = ["spatial","random", "intranuclear", "nuclear-edge", "extranuclear", "foci", "protrusion", "cell-edge", "perinuclear", "pericellular","pattern"]

plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = [20,16]

def saveAFigure(fig, base_title, out_dir = FIG_OUT_DIR, png_dpi = 1200, svg_dpi = 300, rasterize=False):
    out_dir = Path(out_dir)
    fig.tight_layout()
    fig.savefig(out_dir / f"{base_title}.png", dpi=png_dpi, bbox_inches='tight')
    
    if rasterize:
        for ax in fig.axes:
            for artist in ax.collections:
                artist.set_rasterized(True)
    
    fig.savefig(out_dir / f"{base_title}.svg", dpi=svg_dpi, bbox_inches='tight')
    
def get_pattern_palette():
    def get_cmap(n, name='tab10'):
        ##Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        ##RGB color; the keyword argument name must be a standard mpl colormap name.
        return plt.cm.get_cmap(name, n)

    global_cmap = get_cmap(len(tmp_patterns))
    # cmap = get_cmap(len(adata.obs["pattern"].unique()), name="tab10")
    palette = {}
    for i, el in enumerate(tmp_patterns):
        palette[el] =  global_cmap(i)
    return palette

def plotFullDataPoint(adata, tile_nr = None, cell_id = None, gene = None, ax = None, color="red", size=None, row_coords = "subcell_row", col_coords= "subcell_col", return_fig=False):
#     adata_for_this_cell = adata[adata.obs["cell_id"] == cell_id]
    if cell_id is not None:
        adata_for_this_cell_gene = adata[(adata.obs["tile_nr"] == tile_nr) & (adata.obs["cell_id"] == cell_id), gene]
    else:
        adata_for_this_cell_gene = adata
        gene = adata_for_this_cell_gene.var_names[0]
        cell_id = adata_for_this_cell_gene.obs_names[0]
    # this function assumes you either pass an adata slice of len == 1, or that you pass identifiers that could slice it that way
    assert len(adata_for_this_cell_gene) == 1
    
    cell_mask_for_this_cell = (adata_for_this_cell_gene.obsm["cell_mask"][0].reshape(100,100) > 0) * 2
    dapi_mask_for_this_cell = adata_for_this_cell_gene.obsm["dapi_mask"][0].reshape(100,100) > 0
    
    tile_nr = str(adata_for_this_cell_gene.obs["tile_nr"][0])
    measured_df_for_this_tile = adata_for_this_cell_gene.uns["measured_df"][tile_nr]
    detected_transcripts_for_this_cell = convertRecordToDf(adata_for_this_cell_gene.obsm["detected_transcripts"][0])
    
    detected_transcripts_for_this_gene = detected_transcripts_for_this_cell[detected_transcripts_for_this_cell["gene"] == gene]
    
    assert adata_for_this_cell_gene.X == len(detected_transcripts_for_this_gene)
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
 
    ax.imshow(cell_mask_for_this_cell + dapi_mask_for_this_cell)
    if size is None:
        ax.scatter(detected_transcripts_for_this_gene[col_coords], detected_transcripts_for_this_gene[row_coords], color=color)
    else:
        ax.scatter(detected_transcripts_for_this_gene[col_coords], detected_transcripts_for_this_gene[row_coords], color=color, s=size)
        
    if return_fig:
        return fig

