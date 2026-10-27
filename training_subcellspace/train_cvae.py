#!/usr/bin/env python
import os
import sys
import time

# needs to happen before importing CVAE, which lives in the repo root
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
import torch.nn as nn
from skimage import io
from pathlib import Path
from datetime import datetime
from tqdm import trange, tqdm
Image.MAX_IMAGE_PIXELS = None 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from CVAE import  CVAE_hard_coded #, CVAE
from torchvision import datasets, transforms
from utils import ImageDataset, ImageDatasetWithDapisAndLabels, conv_output_shape#, train_model

model_path = "../SubCellSpace" # Which model to create the embeddings with

# Hardcode seed
seed = 101
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

import torch
from torch.utils.tensorboard import SummaryWriter

ap = argparse.ArgumentParser(description="Train the GE model with dapi")

ap.add_argument('-epochs',type=int, nargs='?', default=100, help="Nr of epochs.")
ap.add_argument('-lr',type=float, nargs='?', default=1e-5, help="Learning rate.")

ap.add_argument('-class_lr',type=float, nargs='?', default=1e-3, help="Learning rate for the classifier.")
ap.add_argument('-n_classes',type=int, nargs='?', default=6, help="Number of different classes the classifier needs to distinguish.")

ap.add_argument('-beta',type=float, nargs='?', default=0.0001, help="Beta")
ap.add_argument('-batch_size',type=int, nargs='?', default= 512, help="Batch size")
ap.add_argument('-n_latent_dims',type=int, nargs='?', default=15, help="Number of latent dimensions")
ap.add_argument('-device',type=int, nargs='?', default=0, help="Device to train on")
ap.add_argument('-reduction',type=str, nargs='?', default="sum", help="Reduction")

args = ap.parse_args()

train_dataset = '/path/to/output/of/rotationally_augmented_simulations/coordinate_parsed/cellids_left_out/train/*.tif'
test_dataset = '/path/to/output/of/rotationally_augmented_simulations/coordinate_parsed/cellids_left_out/test/*.tif'
dapi_dataset = '/path/to/output/of/rotationally_augmented_simulations/coordinate_parsed/dapi_zoomed/*.tif'

# unique_patterns = ["random", "intranuclear", "extranuclear", "nuclear_edge", "perinuclear", "cell_edge", "pericellular"]

patterns  =[str(Path(file).stem).split( "_" )[0] for file in glob(train_dataset)]
unique_patterns =sorted(list(set(patterns)))
print(unique_patterns)
# unique_patterns = ["random", "pattern"]
assert args.n_classes == len(unique_patterns)

dapi_prefixes = ["cellID","rotated"]

train_data = ImageDatasetWithDapisAndLabels(train_dataset, dapi_dataset, dapi_prefixes=dapi_prefixes, n_classes = args.n_classes, unique_patterns = unique_patterns, also_binary=True)
val_data = ImageDatasetWithDapisAndLabels(test_dataset, dapi_dataset, dapi_prefixes=dapi_prefixes, n_classes = args.n_classes, unique_patterns = unique_patterns, also_binary=True)

# train_percentage = 0.8
# train_data, val_data = torch.utils.data.random_split(im_dataset, [int(train_percentage*len(im_dataset.scaled_data)), 
                                           # len(im_dataset.scaled_data) - int(train_percentage*len(im_dataset.scaled_data))])


cvae = CVAE_hard_coded(n_channels=2, img_shape=tuple(train_data.scaled_data.shape)[2:], latent_dim=args.n_latent_dims, kernel_size=3)


device = torch.device("cuda", args.device) if torch.cuda.is_available() else 'cpu'

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

model_name = f"{datetime.now().strftime('%Y-%m-%d')}_cvae__train_lr{args.lr:.0e}_classlr{args.class_lr:.0e}_nclasses{args.n_classes}_beta{args.beta:.0e}_{args.n_latent_dims}ld_{args.epochs}epochs"

writer = SummaryWriter(f"./runs/{model_name}/")
cvae.to(args.device)
args_dict = vars(args)

losses = cvae.train_model_with_validation_with_2_classifiers(train_loader, val_loader, args_dict, writer=writer, with_dapi=True)

import os
out_dir = "/path/to/out/dir/"
os.makedirs(out_dir, exist_ok=True)

torch.save({"cvae": cvae, 
            "cvae_state_dict": cvae.state_dict(),
            "epochs": args.epochs,
            "losses": losses,
            "lr": args.lr,
            "ld": args.n_latent_dims,
            "dataset": train_dataset,
            "dapi_dataset": dapi_dataset,
            "beta":args.beta,
            "device": args.device,
            "n_classes": args.n_classes,
            "class_lr": args.class_lr,
            "unique_patterns": unique_patterns,
            "dapi_prefixes": dapi_prefixes,
            "reduction": args.reduction}
           , os.path.join(out_dir, model_name)
           )


