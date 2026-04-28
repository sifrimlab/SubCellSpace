import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# import seaborn as sns
import scanpy as sc
import os
import datetime
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import trange, tqdm
from PIL import Image
import time
from glob import glob
from skimage import io
import re
Image.MAX_IMAGE_PIXELS = None


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, glob_pattern):

        convert_tensor = transforms.ToTensor()
#         normalize_tensor = transforms.Normalize(0,1)
        
        # Add possibility to create an empty object when argument is None, intented to use as start for the sum() function
        if glob_pattern is None:
            self.image_list = []
            self.scaled_data = []
        else:
            self.image_list = sorted(glob(glob_pattern))
            images = []
            scaled_images = []
    #         for image in trange(self.image_list, desc="Loading images", unit="image"):
            for image in tqdm(self.image_list, desc="Loading images", unit="im", total=len(self.image_list)):
                if ".tif" not in image:
                    continue

                # im = Image.open(image)
                im = io.imread(image)
                image_tensor = convert_tensor(im).float()
    #             image_tensor = normalize_tensor(image_tensor.float())
                try:
                    image_tensor = transforms.Normalize(torch.mean(image_tensor),torch.std(image_tensor))(image_tensor)
                except ValueError:
                    print(image)

                image_tensor = torch.unsqueeze(image_tensor, dim=0)
    #             images.append(image_tensor)
            
                scaled_image = self.scale(image_tensor, 0, 1)
                scaled_images.append(scaled_image)

    #         self.data = torch.cat(images, dim=0)
            self.scaled_data = torch.cat(scaled_images, dim=0)
    #         self.scaled_data = self.scale(self.data, 0, 1)

    def __getitem__(self, index):
#         x = self.data[index]
        x = self.scaled_data[index]
        return x

    def __len__(self):
        return len(self.scaled_data)
    
    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __add__(self, other):
        return_dataset = self.__copy__()
        return_dataset.scaled_data = torch.cat((self.scaled_data, other.scaled_data))
        return_dataset.image_list = self.image_list + other.image_list
        return return_dataset

    def __radd__(self, other): 
        return_dataset = self.__copy__()
        return_dataset.scaled_data = torch.cat((self.scaled_data, other.scaled_data))
        return_dataset.image_list = self.image_list + other.image_list
        return return_dataset
    
    def subsample(self, n):
        import random
        return_dataset = self.__copy__()
        image_list, scaled_data = zip(*random.sample(list(zip(self.image_list, self.scaled_data)), n))
        return_dataset.scaled_data = scaled_data
        return_dataset.image_list = image_list
        return return_dataset

    def scale(self, tensor, min_value, max_value):
        v_min, v_max = tensor.min(), tensor.max()
        new_min, new_max = min_value, max_value
        v_p = (tensor - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        return v_p


def conv_output_shape(h, w, kernel_size, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes height an width and returns a tuple of the (h,w)
    after convolution.
    """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((w + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w


def scaleImage(path, out_dir):

    def scale(tensor, min_value, max_value):
        v_min, v_max = tensor.min(), tensor.max()
        new_min, new_max = min_value, max_value
        v_p = (tensor - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        return v_p

    convert_tensor = transforms.ToTensor()
    img = io.imread(path)
    image_tensor = convert_tensor(img).float()
    image_tensor = transforms.Normalize(torch.mean(image_tensor),torch.std(image_tensor))(image_tensor)
    scaled_image = scale(image_tensor, 0, 1)
    # scaled_image = scaled_image.permute(1,2,0)
    np_arr = scaled_image.numpy()
    io.imsave(os.path.join(out_dir, os.path.basename(path)), np_arr)

        
class ImageDatasetWithDapis(torch.utils.data.Dataset):
    def __init__(self, glob_pattern, dapi_glob_pattern, dapi_prefixes = [], stem=None):
        # stem is a temp argument for some brute forcing, please remove if you don't remember what this is for
        # Dapi prexises is a list of strings that combined with their numeric value in teh path (e.g.: prefix105) link the data to their respective dapi/cell image

        convert_tensor = transforms.ToTensor()
#         normalize_tensor = transforms.Normalize(0,1)
        
        if isinstance(glob_pattern, list):
            self.image_list = glob_pattern
        else:
            self.image_list = sorted(glob(glob_pattern))

        if isinstance(dapi_glob_pattern, list):
            self.dapi_list = dapi_glob_pattern
        else:
            self.dapi_list = sorted(glob(dapi_glob_pattern))

        self.dapi_base_dir = Path(self.dapi_list[0]).parent

        def getDapiBase(dapi_file):
            for prefix in dapi_prefixes:
                prefix_value = re.findall('\d+',re.findall(rf"{prefix}\d+", str(Path(dapi_file).name))[0])[0]
            # cell = re.findall('\d+',re.findall("cell\d+", str(Path(dapi_file).name))[0])[0]
            # dapi_file = dapi_file.replace(f"tile{tile}", "tileNumber")
                dapi_file = dapi_file.replace(f"{prefix}{prefix_value}", f"{prefix}Number")
            return dapi_file

        self.corresponding_dapis = []

        if stem is not None:
            for image_path in self.image_list:
                cell_id = re.findall("cellID(\d+)", image_path)
                dapi_of_that_cell =  glob(str(self.dapi_base_dir / f"*cellID{cell_id}*.tif"))[0]
                self.corresponding_dapis.append(dapi_of_that_cell)

        if dapi_prefixes:
            self.dapi_base = getDapiBase(self.dapi_list[0])
            for image_path in self.image_list:
                tmp_dapi_base = self.dapi_base
                for prefix in dapi_prefixes:
                    prefix_value = re.findall('\d+',re.findall(rf"{prefix}\d+", str(Path(image_path).name))[0])[0]
                    tmp_dapi_base = tmp_dapi_base.replace(f"{prefix}Number", f"{prefix}{prefix_value}")

                corresponding_dapi_file = self.dapi_base_dir  / tmp_dapi_base
                self.corresponding_dapis.append(corresponding_dapi_file)
        else:
            for image_path in self.image_list:
                corresponding_dapi_file = self.dapi_base_dir / f"{Path(image_path).stem}_DAPI.tif"
                self.corresponding_dapis.append(corresponding_dapi_file)

        scaled_images = []
        scaled_dapis = []
        for image_path, dapi_path in tqdm(zip(self.image_list[:], self.corresponding_dapis[:]), desc="Loading images", unit="image", total=len(self.image_list)):
            # [:] syntax here makes sure we iterate over a copy of the image lists, and not the actual lists, because if there's an error in them (division by zero error or anything like that, we'll catch the arror and remove this entry from the list.
            # Sometimes a cell will be dropped in parsing, so we double check whether the dapi actually exists
            if not Path(dapi_path).is_file():
                print(image_path, dapi_path)
                self.image_list.remove(image_path)
                self.corresponding_dapis.remove(dapi_path)
                continue
            # # Manipulating a list while iterating over it would be a terrible idea)
            # if ".tif" not in image_path:
            #     continue

            # im = Image.open(image)
            im = io.imread(image_path)
            dapi = io.imread(dapi_path)
            dapi = dapi.astype(np.float32)
            image_tensor = convert_tensor(im).float()
            dapi_tensor =  convert_tensor(dapi).float()


            # image_tensor = normalize_tensor(image_tensor.float())
            # dapi_tensor = normalize_tensor(dapi_tensor.float())
            try:
                image_tensor = transforms.Normalize(torch.mean(image_tensor),torch.std(image_tensor))(image_tensor)
                dapi_tensor = transforms.Normalize(torch.mean(dapi_tensor),torch.std(dapi_tensor))(dapi_tensor)
            except ValueError:
                # print(image_path, dapi_path)
                self.image_list.remove(image_path)
                self.corresponding_dapis.remove(dapi_path)
                continue

            image_tensor = torch.unsqueeze(image_tensor, dim=0)
            dapi_tensor = torch.unsqueeze(dapi_tensor, dim=0)
        

            scaled_image = self.scale(image_tensor, 0, 1)
            scaled_dapi = self.scale(dapi_tensor, 0, 1)
            if (torch.isnan(scaled_image).any()) or (torch.isnan(scaled_dapi).any()):
                self.image_list.remove(image_path)
                self.corresponding_dapis.remove(dapi_path)
                continue
            scaled_images.append(scaled_image)
            scaled_dapis.append(scaled_dapi)

        self.scaled_data = torch.cat(scaled_images, dim=0)
        self.scaled_dapi = torch.cat(scaled_dapis, dim=0)

    def __getitem__(self, index):
        x = self.scaled_data[index]
        y = self.scaled_dapi[index]
        z = torch.cat((x,y), dim=0)
        return z

    def __len__(self):
        return len(self.scaled_data)
    
    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __add__(self, other):
        return_dataset = self.__copy__()
        return_dataset.scaled_data = torch.cat((self.scaled_data, other.scaled_data))
        return_dataset.scaled_dapi = torch.cat((self.scaled_dapi, other.scaled_dapi))
        return_dataset.image_list = self.image_list + other.image_list
        return_dataset.corresponding_dapis = self.corresponding_dapis + other.corresponding_dapis
        return return_dataset
        # return_dataset = self.__copy__()
        # return_dataset.scaled_data = torch.cat((self.scaled_data, other.scaled_data))
        # return_dataset.scaled_dapi = torch.cat((self.scaled_dapi, other.scaled_dapi))
        # return_dataset.image_list = self.image_list + other.image_list
        # return return_dataset

    def __radd__(self, other): 
        return_dataset = self.__copy__()
        return_dataset.scaled_data = torch.cat((self.scaled_data, other.scaled_data))
        return_dataset.scaled_dapi = torch.cat((self.scaled_dapi, other.scaled_dapi))
        return_dataset.image_list = self.image_list + other.image_list
        return_dataset.corresponding_dapis = self.corresponding_dapis + other.corresponding_dapis
        return return_dataset
    
    def subsample(self, n):
        import random
        return_dataset = self.__copy__()
        image_list, scaled_data = zip(*random.sample(list(zip(self.image_list, self.scaled_data)), n))
        return_dataset.scaled_data = scaled_data
        return_dataset.image_list = image_list
        return return_dataset

    def scale(self, tensor, min_value, max_value):
        v_min, v_max = tensor.min(), tensor.max()
        new_min, new_max = min_value, max_value
        v_p = (tensor - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        return v_p


class ImageDatasetWithDapisAndLabels(torch.utils.data.Dataset):
    def __init__(self, glob_pattern, dapi_glob_pattern, dapi_prefixes = [], n_classes = 2, unique_patterns = [], also_binary=False):
        # Dapi prexises is a list of strings that combined with their numeric value in teh path (e.g.: prefix105) link the data to their respective dapi/cell image
        def oneHotEncode(index: int, array_len: int):
            zeros = np.zeros(array_len)
            zeros[index] = 1
            return zeros

        convert_tensor = transforms.ToTensor()
#         normalize_tensor = transforms.Normalize(0,1)
        
        self.also_binary = also_binary
        if isinstance(glob_pattern, list):
            self.image_list = glob_pattern
        else:
            self.image_list = sorted(glob(glob_pattern))

        # TMP DEF REMOVE
        # self.image_list =self.image_list[:1000]
        self.dapi_list = sorted(glob(dapi_glob_pattern))
        self.dapi_base_dir = Path(self.dapi_list[0]).parent
        if self.also_binary:
            binary_labels = []
        labels = []

        if not unique_patterns:
            self.unique_patterns = ["random" , "foci", "intranuclear", "extranuclear", "nuclear-edge", "perinuclear", "cell-edge", "pericellular", "protrusion"]
        else:
            self.unique_patterns = unique_patterns

        def getDapiBase(dapi_file):
            for prefix in dapi_prefixes:
                prefix_value = re.findall('\d+',re.findall(rf"{prefix}\d+", str(Path(dapi_file).name))[0])[0]
            # cell = re.findall('\d+',re.findall("cell\d+", str(Path(dapi_file).name))[0])[0]
            # dapi_file = dapi_file.replace(f"tile{tile}", "tileNumber")
                dapi_file = dapi_file.replace(f"{prefix}{prefix_value}", f"{prefix}Number")
            return dapi_file

        self.dapi_base = getDapiBase(self.dapi_list[0])


        self.corresponding_dapis = []
        # for image_path in self.image_list:
        #     tmp_dapi_base = self.dapi_base
        #     for prefix in dapi_prefixes:
        #         prefix_value = re.findall('\d+',re.findall(rf"{prefix}\d+", str(Path(image_path).name))[0])[0]
        #         tmp_dapi_base = tmp_dapi_base.replace(f"{prefix}Number", f"{prefix}{prefix_value}")

        #     corresponding_dapi_file = self.dapi_base_dir  / tmp_dapi_base
        #     self.corresponding_dapis.append(corresponding_dapi_file)
        for image_path in self.image_list:
            corresponding_dapi_file = self.dapi_base_dir / f"{Path(image_path).stem}_DAPI.tif"
            self.corresponding_dapis.append(corresponding_dapi_file)

        scaled_images = []
        scaled_dapis = []
        for image, dapi in tqdm(zip(self.image_list, self.corresponding_dapis), desc="Loading images", unit="image"):
        # for image in tqdm(self.image_list, desc="Loading images", unit="im", total=len(self.image_list)):
            if ".tif" not in image:
                continue

            # if binary
            if n_classes == 2:
                pattern = int(os.path.basename(image).split("_")[0] != "random")
                one_hot_encoded_pattern = oneHotEncode(pattern, 2)
            # if more than 2 classes
            else:
                pattern = os.path.basename(image).split("_")[0]
                try:
                    index = self.unique_patterns.index(pattern)
                    one_hot_encoded_pattern = oneHotEncode(index, len(self.unique_patterns))
                except: # if pattern not in unique patterns, we skip this image
                    continue
                if self.also_binary:
                    binary_pattern = int(os.path.basename(image).split("_")[0] != "random")
                    binary_one_hot_encoded_pattern = oneHotEncode(binary_pattern, 2)
                # pattern = os.path.basename(image).split("_")[0]
                # try:
                #     index = self.unique_patterns.index(pattern)
                #     one_hot_encoded_pattern = oneHotEncode(index, len(self.unique_patterns))
                # except: # if pattern not in unique patterns, we skip this image
                #     continue

            im = io.imread(image)
            dapi = io.imread(dapi)
            dapi = dapi.astype(np.float32)
            image_tensor = convert_tensor(im).float()
            dapi_tensor =  convert_tensor(dapi).float()
            # image_tensor = normalize_tensor(image_tensor.float())
            # dapi_tensor = normalize_tensor(dapi_tensor.float())
            try:
                image_tensor = transforms.Normalize(torch.mean(image_tensor),torch.std(image_tensor))(image_tensor)
                dapi_tensor = transforms.Normalize(torch.mean(dapi_tensor),torch.std(dapi_tensor))(dapi_tensor)
            except ValueError:
                print(image, dapi)

            image_tensor = torch.unsqueeze(image_tensor, dim=0)
            dapi_tensor = torch.unsqueeze(dapi_tensor, dim=0)
        

            scaled_image = self.scale(image_tensor, 0, 1)
            scaled_images.append(scaled_image)
            scaled_dapi = self.scale(dapi_tensor, 0, 1)
            scaled_dapis.append(scaled_dapi)
            labels.append(one_hot_encoded_pattern)
            if self.also_binary:
                binary_labels.append(binary_one_hot_encoded_pattern)

        self.scaled_data = torch.cat(scaled_images, dim=0)
        self.scaled_dapi = torch.cat(scaled_dapis, dim=0)
        self.labels = labels
        if self.also_binary:
            self.binary_labels = binary_labels

    def __getitem__(self, index):
#         x = self.data[index]
        x = self.scaled_data[index]
        y = self.scaled_dapi[index]
        z = torch.cat((x,y), dim=0)
        label = self.labels[index].astype(np.float64)
        if self.also_binary:
            bin_label =  self.binary_labels[index].astype(np.float64)
            return z, label, bin_label
        else:
            return z,label

    def __len__(self):
        return len(self.scaled_data)
    
    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __add__(self, other):
        return_dataset = self.__copy__()
        return_dataset.scaled_data = torch.cat((self.scaled_data, other.scaled_data))
        return_dataset.scaled_dapi = torch.cat((self.scaled_dapi, other.scaled_dapi))
        return_dataset.image_list = self.image_list + other.image_list
        return_dataset.corresponding_dapis = self.corresponding_dapis + other.corresponding_dapis
        return_dataset.labels = self.labels + other.labels
        return_dataset.binary_labels = self.binary_labels + other.binary_labels
        return return_dataset

    def __radd__(self, other): 
        return_dataset = self.__copy__()
        return_dataset.scaled_data = torch.cat((self.scaled_data, other.scaled_data))
        return_dataset.scaled_dapi = torch.cat((self.scaled_dapi, other.scaled_dapi))
        return_dataset.image_list = self.image_list + other.image_list
        return_dataset.corresponding_dapis = self.corresponding_dapis + other.corresponding_dapis
        return return_dataset
    
    def subsample(self, n):
        import random
        return_dataset = self.__copy__()
        image_list, scaled_data = zip(*random.sample(list(zip(self.image_list, self.scaled_data)), n))
        return_dataset.scaled_data = scaled_data
        return_dataset.image_list = image_list
        return return_dataset

    def scale(self, tensor, min_value, max_value):
        v_min, v_max = tensor.min(), tensor.max()
        new_min, new_max = min_value, max_value
        v_p = (tensor - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        return v_p

def scale(self, tensor, min_value, max_value):
    v_min, v_max = tensor.min(), tensor.max()
    new_min, new_max = min_value, max_value
    v_p = (tensor - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    return v_p


def turn_im_dataset_into_adata(im_dataset, latent):
    latent_df = pd.DataFrame(latent, columns=["LatentDim{}".format(i+1) for i in range(latent.shape[1])])#,

    adata = ad.AnnData(X=latent_df)
    adata.obsm["latent"] = latent
    adata.uns["latent_df_keys"] = latent_df.columns # for later to plot activation per latent dimension

    patterns = [Path(img_path).stem.split("_")[0] for img_path in im_dataset.image_list]
    #if the first thing is a tile then it's real spatial data, so then there's no pattern to find in the ground truth
    rand_or_pat = ["random" if el == "random" else "pattern" for el in patterns ]
    adata.obs["pattern"] = patterns
    adata.obs["random_or_pattern"] = rand_or_pat

    # Find number of spots
    n_spots = []
    for img_path in im_dataset.image_list:
        img_path = Path(img_path)
        spots = int(re.findall(r"(?:spots(\d+)|(\d+)spots)", img_path.stem)[0][1]) # First index is for first match, second index is for matching group of regex
        n_spots.append(spots)
    adata.obs["n_spots"] = n_spots

    # Bin the number of spots so we can plot it later in discrete categories
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('Inf')]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "70-80","80-90","90-100","100+"]
    n_spots_interval = pd.cut(adata.obs['n_spots'], bins=bins, labels=labels)
    adata.obs["n_spots_interval"] = n_spots_interval

    try:
        props = []
        for img_path in im_dataset.image_list:
            img_path = Path(img_path)
            rex_find = re.findall(r"(?:prop(\d\.\d+)|(\d\.\d+)prop)", img_path.stem)[0][0]
            prop = float(rex_find) # First index is for first match, second index is for matching group of regex
            props.append(prop)
        adata.obs["prop"] = props
    except IndexError:
        print("no prop")
    
    # Find cell identity
    cell_ids = []
    for img_path in im_dataset.image_list:
        img_path = Path(img_path)
        try:
            cell_id = re.findall(r'cellID(\d+)',img_path.stem)[0]
        except IndexError:
            cell_id = "blank"
        cell_ids.append(cell_id)
    adata.obs["cell_id"] = cell_ids

    # Find whether it has a gene name
    genes = []
    for img_path in im_dataset.image_list:
        img_path = Path(img_path)
        try:

            gene = re.findall(r'genName([a-zA-Z0-9]+)',img_path.stem)[0]
        except IndexError:
            gene = "blank"
        genes.append(gene)
    adata.obs["genes"] = genes

    try:
        rotation = [int(re.findall('rotated(\d+)', Path(img_path).stem)[0]) for img_path in im_dataset.image_list]
        adata.obs["rotation"] = rotation
        adata.obs["rotation_interval"] =  pd.cut(adata.obs['rotation'], bins=[0, 60, 120, 180, 240, 300, float('Inf')], labels=["0-60", "60-120", "120-180", "180-240", "240-300", "300+"])
    except (IndexError,KeyError):
        print("no rotation")

    # Find corresponding_dapis dapis
    corresponding_dapis = im_dataset.corresponding_dapis
    adata.obs["corresponding_dapis"] = corresponding_dapis
    adata.obs["corresponding_dapis"] = adata.obs["corresponding_dapis"].astype(str)

    corresponding_dapi_images =np.array([io.imread(file) for file in adata.obs["corresponding_dapis"]])
    adata.obsm["corresponding_dapi_images"] = corresponding_dapi_images

    if (img_path.parent.parent / "cell_mask_zoomed").is_dir():
        corresponding_cells = []
        for file in corresponding_dapis:
            file = str(file)
            corresponding_cell = file.replace("dapi_zoomed", "cell_mask_zoomed").replace("DAPI.tif", "cell-mask.tif")
            corresponding_cells.append(corresponding_cell)
        adata.obs["corresponding_cells"] = corresponding_cells

        corresponding_cell_images = np.array([io.imread(file) for file in adata.obs["corresponding_cells"]])
        adata.obsm["corresponding_cell_images"] = corresponding_cells_images

     # Get paths to input data as well
    adata.obs["original_image_paths"] = im_dataset.image_list
        
    train_list = ["train"] * len(im_dataset.image_list)    
    for index, img_path in enumerate(im_dataset.image_list):
        if "test" in img_path:
            train_list[index] = "val"
            
    adata.obs["train_or_val"] = train_list
    return adata
