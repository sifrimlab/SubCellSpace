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

def createSubcellRepresentation(input_img, dapi, model, device):
    prepared = prepare_new_input(input_img)
    prepared_dapi = prepare_new_input(dapi)

    input_model = torch.cat((prepared, prepared_dapi), dim=1)
    # input_model.to(device)
    embedding, _, _ = model["cvae"].embed(input_model.to(device))
    return embedding


def prepare_batch(images):
    """Batched, exact equivalent of ``prepare_new_input`` for a (B, H, W) tensor.

    Per image: z-normalise by its own mean/std (unbiased, matching ``torch.std``),
    then min-max rescale to [0, 1]. ``ToTensor`` on a float image is a no-op
    scaling-wise, so this reproduces ``prepare_new_input`` for a whole batch at
    once -- avoiding the per-image torchvision object construction that dominated
    the runtime.
    """
    b = images.shape[0]
    flat = images.reshape(b, -1)
    mean = flat.mean(dim=1).view(b, 1, 1)
    std = flat.std(dim=1, unbiased=True).view(b, 1, 1)
    z = (images - mean) / std
    zflat = z.reshape(b, -1)
    zmin = zflat.amin(dim=1).view(b, 1, 1)
    zmax = zflat.amax(dim=1).view(b, 1, 1)
    return (z - zmin) / (zmax - zmin)


def embedSubcellBatch(gene_imgs, dapi_imgs, model, device):
    """Embed many (gene-image, dapi) pairs in one GPU forward pass.

    ``gene_imgs`` / ``dapi_imgs`` are lists of equally-shaped 2-D arrays. Returns
    an ``(N, latent_dim)`` numpy array. Equivalent to calling
    :func:`createSubcellRepresentation` per pair (the encoder has no
    batch-dependent layers), but far faster because preparation is vectorised and
    the forward pass is batched.
    """
    gene_t = torch.from_numpy(np.stack(gene_imgs)).float().to(device)
    dapi_t = torch.from_numpy(np.stack(dapi_imgs)).float().to(device)
    with torch.no_grad():
        batch = torch.stack([prepare_batch(gene_t), prepare_batch(dapi_t)], dim=1)  # (N, 2, H, W)
        embedding, _, _ = model["cvae"].embed(batch)
    return embedding.cpu().detach().numpy()

