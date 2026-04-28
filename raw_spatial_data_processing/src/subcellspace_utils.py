from pathlib import Path
import torch
import numpy as np
from torchvision import  transforms
from skimage import io

def scale(tensor, min_value, max_value):
    v_min, v_max = tensor.min(), tensor.max()
    new_min, new_max = min_value, max_value
    v_p = (tensor - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    return v_p

def prepare_new_input(arr):
    tf = transforms.ToTensor()
    image_tensor = tf(arr).float()
    image_tensor = transforms.Normalize(torch.mean(image_tensor),torch.std(image_tensor))(image_tensor) 
    image_tensor = torch.unsqueeze(image_tensor, dim=0)
    scaled_image = scale(image_tensor, 0, 1)
    return scaled_image

def create_subcell_representation(input_img, dapi, model, device):
    prepared = prepare_new_input(input_img)
    prepared_dapi = prepare_new_input(dapi)
        
    input_model = torch.cat((prepared, prepared_dapi), dim=1)
    # input_model.to(device)
    embedding, _, _ = model["cvae"].embed(input_model.to(device))
    return embedding

