# imports
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
from tqdm import tqdm
import torch.nn as nn
from skimage import io
from pathlib import Path
Image.MAX_IMAGE_PIXELS = None 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import ImageDatasetWithDapis, turn_im_dataset_into_adata

seed = 101
torch.manual_seed(seed)
import random
random.seed(seed)
np.random.seed(seed)

# need to be able to load CVAE.py, which lives in the repo root
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
model_path = "../SubCellSpace" # Which model to create the embeddings with

from CVAE import CVAE_hard_coded
subcellspace = torch.load(model_path)


im_dataset = "/path/to/parsed/images/*.tif"
dapi_dataset =  "/path/to/parsed/dapi_images/dapi_zoomed/*.tif"

im_dataset = ImageDatasetWithDapis(im_dataset, dapi_dataset)

train_loader = torch.utils.data.DataLoader(im_dataset, batch_size=256, shuffle=False)

latents = []
with torch.no_grad():
#         print(row.device)
    device = subcellspace["cvae_state_dict"]['encoder_conv.0.weight'].device
    subcellspace["cvae"].eval()
    subcellspace["cvae"].to(device)

    latent_list = []
    for batch in tqdm(train_loader):
        latent, _, _ = subcellspace["cvae"].embed(batch.to(device))
    #         latent = latent.cpu().numpy() #.detach().numpy()
        latent_list.append(latent.to("cpu"))
    latent = torch.concat(latent_list, dim=0)
    latent = latent.numpy()
latents.append(latent)

adata = turn_im_dataset_into_adata(im_dataset, latent)

adata.write_h5ad("output.h5ad")
