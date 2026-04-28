import pandas as pd
from skimage import io
import numpy as np
from pathlib import Path
from tqdm import tqdm
from xml.etree import ElementTree
import shapely
from tifffile import TiffFile
import PIL
import json

def transform_detected_transcripts_to_pixels(df, matrix, offset):
    new_global_x = []
    new_global_y = []
    for row in tqdm(df.itertuples(), total=len(df), unit="pts"):
        point = np.array([row.global_y, row.global_x]).T
        new_point = np.dot(matrix, point).T + np.flip(offset)
        new_global_x.append(new_point[1])
        new_global_y.append(new_point[0])
    
    df["col"] = new_global_x
    df["row"] = new_global_y
    return df

def faster_transform_detected_transcripts_to_pixels(df, matrix, offset):
    points = np.array([df["global_y"], df["global_x"]])

    new_points = np.dot(matrix, points).T + np.flip(offset)

    df["row"] =list(new_points[:, 0])
    df["col"] =list(new_points[:, 1])

    try:
        df = df.drop(["x","y","fov"], axis=1)
    except KeyError:
        pass
    return df


def parse_merfish_transform_matrix(transform_path):
    transform = pd.read_csv(transform_path, sep=" ", header=None).values
    matrix = transform[:2, :2]
    offset = transform[:2, 2]

    return matrix, offset



def parse_detected_transcripts(df_path, transform_path):
    detected_transcripts = pd.read_csv(df_path, index_col=0)
    matrix, offset =parseMerfishTransformMatrix(transform_path)
    detected_transcripts  = fasterTransformDetectedTranscriptsToPixels(detected_transcripts, matrix, offset) 
    return detected_transcripts

def extract_physical_size_from_ome_tiff(ome_tif_path):
    with TiffFile(ome_tif_path) as tif:
        for page in tif.series[0].pages:
            pixels_element = ElementTree.fromstring(page.description).find('.//{*}Pixels')
            physical_size_x = float(pixels_element.get('PhysicalSizeX'))
            physical_size_y = float(pixels_element.get('PhysicalSizeY'))
    return physical_size_x, physical_size_y 

def extract_physical_size_experiment_file(experiment_file):
    with open(experiment_file, 'r') as f:
        exp = json.load(f)
        pixel_size =float( exp["pixel_size"])
    return pixel_size, pixel_size 



def parse_xenium_detected_transcripts(df_path, experiment_path):
    if not isinstance(df_path, Path):
        df_path = Path(df_path)

    if str(experiment_path).endswith(".ome.tif"):
        physical_size_x, physical_size_y = extractPhysicalSizeFromOmeTiff(experiment_path)
    else:
        physical_size_x, physical_size_y = extractPhysicalSizeExperimentFile(experiment_path)

    if df_path.stem.endswith("csv"):
        df = pd.read_csv(df_path,index_col=0)
    else:
        df = pd.read_parquet(df_path)
    df["col"] = df["x_location"].apply(lambda x: x / physical_size_x )
    df["row"] = df["y_location"].apply(lambda x: x / physical_size_y )
    df = df.rename(columns={"feature_name": "gene"})
    # Remove xenium specific cols that I'll maky my own for anyway
    df = df.drop(["cell_id", "overlaps_nucleus", "qv", "fov_name", "nucleus_distance", "codeword_index"], axis=1)
    return df

def parse_cell_boundaries(parquet_path, transform_path):
    cell_boundaries = pd.read_parquet(parquet_path)

    matrix, offset =parseMerfishTransformMatrix(transform_path)

    coords_lists = []
    for tmp_row in cell_boundaries.itertuples():
        multipolygon = shapely.from_wkb(tmp_row.Geometry)
        # We assume that every polygon only has one set of coords
        try:
            polygon = multipolygon.geoms[0] 
            coords_list = list(polygon.exterior.coords)
            points = np.array(coords_list).T
            new_points = np.dot(matrix, points).T + np.flip(offset)
            coords_lists.append(list(new_points))
        except IndexError:
            coords_lists.append(np.nan)

    cell_boundaries["parsed_geometry"] = coords_lists
    cell_boundaries = cell_boundaries[cell_boundaries['parsed_geometry'].notna()]

    return cell_boundaries



if __name__ == '__main__':
    pass
    # # Detected transcripts
    # df = pd.read_csv("/mnt/data/MERFISH/moor_data/merfish/ileum/detected_transcripts.csv", index_col=0)
    # matrix, offset = parseMerfishTransformMatrix("/mnt/data/MERFISH/moor_data/merfish/ileum/images/micron_to_mosaic_pixel_transform.csv")
    # parsed_df = fasterTransformDetectedTranscriptsToPixels(df, matrix, offset)
    # parsed_df.to_csv("/mnt/data/MERFISH/moor_data/merfish/detected_transcripts2_parsed.csv")

    # Parse segmentation coords
    # parsed_cell_boundaries = parseCellBoundaries("/mnt/data/syno_voet_backup/MERSCOPE/helmsley_project/202311101528_Helmsley-slide1-FF-colon_VMSC07501/region_0/cell_boundaries.parquet","/mnt/data/syno_voet_backup/MERSCOPE/helmsley_project/202311101528_Helmsley-slide1-FF-colon_VMSC07501/region_0/images/micron_to_mosaic_pixel_transform.csv")
    # parsed_cell_boundaries.to_parquet("/mnt/data/syno_voet_backup/MERSCOPE/helmsley_project/202311101528_Helmsley-slide1-FF-colon_VMSC07501/region_0/cell_boundaries_parsed.parquet")
