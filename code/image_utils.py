import cv2
import numpy as np
import h5py
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

def plot_image_with_spot(image, loc, spot_radius_pixel, mask_matrix):
    # Create a copy of the image to draw spots with transparency
    image_with_spots = image.copy()
    # Draw spots on the image
    for (x, y) in loc:
        #! Notice that this the first column of loc is the col index for the image matrix, which is the x axis direction on the plot!
        #! For cv2, (x,y) is (col_index, row_index)
        image_with_spots = cv2.circle(image_with_spots, (int(x), int(y)), int(spot_radius_pixel), (0, 255, 0), -1)
    # Create a mask for the spots
    # spots_mask = np.zeros_like(image_with_spots, dtype=np.uint8)
    # for (x, y) in loc:
    #     spots_mask = cv2.circle(spots_mask, (int(x), int(y)), int(spot_radius_pixel), (0, 255, 0), -1)
    # Blend the spots with the original image
    image_with_spots = cv2.addWeighted(image, 0.7, image_with_spots, 0.3, 0)
    image_with_spots[mask_matrix == 0] = [255, 255, 255]  # Color the masked area as white
    image_with_spots = cv2.resize(image_with_spots, (0,0), fx=0.1, fy=0.1)
    return image_with_spots


def cut_and_rescale_image(image, mask, loc, raw_pixel_size, target_pixel_size = 0.5):
    scale_factor = raw_pixel_size/target_pixel_size
    # Cut the white border of the image
    # Find the bounding box of the region of interest in the mask
    coords = np.column_stack(np.where(mask > 0))
    row_min, col_min = coords.min(axis=0)
    row_max, col_max = coords.max(axis=0)
    image = image[row_min:row_max, col_min:col_max]
    mask = mask[row_min:row_max, col_min:col_max]
    #! Notice that this the first column of loc is the col index for the image matrix, which is the x axis direction on the plot!
    if loc is not None:
        loc = loc - [col_min, row_min]
    # Rescale the image
    image = cv2.resize(image, (0,0), fx=scale_factor, fy=scale_factor)
    mask = cv2.resize(mask.astype('float32'), (0,0), fx=scale_factor, fy=scale_factor)
    if loc is not None:
        loc = loc * scale_factor
        loc = loc.round().astype(int)
    return image, mask, loc, (row_min, row_max, col_min, col_max, scale_factor)

def get_UNI_model(token, version = "UNI2-h"):
    login(token=token)
    # pretrained=True needed to load UNI weights (and download weights for the first time)
    # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
    print(f"Loading model: {version}")
    if version == "UNI":
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif version == "UNI2-h":
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
            }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    print(f"Finished loading model: {version}")
    return model, transform

def get_HEST1k_patch(h5_file_path):
    # Open the .h5 file
    print(f"Fetching patches in {h5_file_path.replace('.h5', '')}")
    img_meta = {}
    meta_key = ['downsample', 'patch_size_src', 'patch_size_target', 'pixel_size']
    with h5py.File(h5_file_path, 'r') as f:
        barcode = f['barcode'][:]
        img = f['img'][:]
        coords = f['coords'][:]
        for key in meta_key:
            img_meta[key] = f['img'].attrs[key]
    print(f"Finished fetching patches in {h5_file_path.replace('.h5', '')}")
    return img, img_meta, barcode, coords
