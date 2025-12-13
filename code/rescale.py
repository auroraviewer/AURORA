import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import tifffile
import geopandas as gpd
import scanpy as sc
from rasterio.features import rasterize
import utils
from image_utils import cut_and_rescale_image, plot_image_with_spot
import cv2
import argparse

parser = argparse.ArgumentParser(description='Rescale H&E images to target pixel size and adjust ST spot locations accordingly.')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--pixel_size', type=float, default=0.5, help='Target pixel size in microns')
parser.add_argument('--output_path', '-o', type=str, default='AURORA_interim', help='Output directory')
parser.add_argument('--metadata_path', '-m', type=str, default='Sample_metadata.csv', help='Sample metadata CSV file name')
parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
parser.add_argument('--plot', action='store_true', help='Plot intermediate results for checking')
parser.add_argument('--inference', action='store_true', help='Inference mode flag')
parser.add_argument('--image_path', type=str, default='./wsis/', help='Data directory path')
parser.add_argument('--seg_path', type=str, default='./wsis_segmentation/', help='Save directory path')
args = parser.parse_args()

target_pixel_size = args.pixel_size  # in microns
main_dir = args.project_path
save_dir = f"{main_dir}/{args.output_path}/UNI_multiscale_patches"
inference_flag = args.inference

if inference_flag:
    import openslide

os.makedirs(f"{save_dir}", exist_ok=True)
os.makedirs(f"{save_dir}/wsis_rescaled", exist_ok=True)
if args.plot:
    os.makedirs(f"{save_dir}/check_figure", exist_ok=True)
if not inference_flag:
    meta = pd.read_csv(f"{main_dir}/{args.metadata_path}",index_col=0, header=0)
    image_files = list(meta.index)
else:
    meta = pd.read_csv(f"{main_dir}/{args.metadata_path}", header=0)
    image_files = list(meta['Image_name']) # image names with suffix
    meta.set_index('Image_name', inplace=True)

if not inference_flag:
    rescale_parameters = pd.DataFrame(index=meta.index, columns=['row_min', 'row_max', 'col_min', 'col_max', 'scale_factor'])
else:
    rescale_parameters = pd.DataFrame(index=meta['Sample_name'], columns=['row_min', 'row_max', 'col_min', 'col_max', 'scale_factor'])

for image_file in tqdm(image_files):
    if not inference_flag:
        image_name = image_file.split('.')[0]
    else:
        image_name = meta.loc[image_file,'Sample_name']
    if os.path.exists(f"{save_dir}/wsis_rescaled/{image_name}_mask_rescaled.tif"):
        print(f"{image_name} already processed")
        continue
    print(f"Processing {image_name}")
    if not inference_flag:
    # Load image metadata from json
        with open(os.path.join(main_dir, args.processed_data_path, 'metadata', f"{image_name}.json"), 'r') as f:
        # Load the JSON data into a Python dictionary
            metadata = json.load(f)
        pixel_size_raw = metadata['pixel_size_um_estimated']
        print(f"    Raw pixel size: {pixel_size_raw}")
        # Load image
        image_path = os.path.join(main_dir, args.processed_data_path, 'wsis', f"{image_name}.tif")
        mask_path = os.path.join(main_dir, args.processed_data_path,  'tissue_seg', f'{image_name}_contours.geojson')
        if os.path.exists(mask_path):
            with tifffile.TiffFile(image_path) as tif:
                # Access the image data at different levels
                image = tif.pages[0].asarray()
                # print(image.shape)
            mask_gdf = gpd.read_file(mask_path)
        else:
            print(f"    No mask found for {image_name}")
            continue
    else:
        if 'Image_folder' in meta.columns:
            image_path = os.path.join(main_dir, f'{args.image_path}/{meta.loc[image_file,"Image_folder"]}/', image_file)
        else:
            image_path = os.path.join(main_dir, f'{args.image_path}/', image_file)
        mask_file = '.'.join(image_file.split('.')[:-1]) + '.geojson'
        mask_path = os.path.join(main_dir, f'{args.seg_path}/masks', mask_file)
        image_os = openslide.open_slide(image_path)
        level_dim = image_os.level_dimensions
        if '.svs' in image_file:
            pixel_size_raw = float(image_os.properties['openslide.mpp-x'])
        else:
            pixel_size_raw = float(meta.loc[image_file,'Pixel_size_um'])
        print(f"    Raw pixel size: {pixel_size_raw}")
        image = np.array(image_os.read_region((0,0),0,level_dim[0]))
        image_os.close()
        image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
        mask_gdf = gpd.read_file(mask_path)
        if mask_gdf.loc[0,"geometry"].is_empty:
            print(f"    No contour is detected for {image_name}. Possibly the tissue is too scattered. Please double check.")
            continue
    ## Get mask matrix
    # Get the dimensions of the image
    height, width, _ = image.shape
    # Create an empty mask matrix
    mask_matrix = np.zeros((height, width), dtype=np.uint8)
    # Rasterize the geometries in mask_gdf
    shapes = ((geom, 1) for geom in mask_gdf.geometry)
    mask_matrix = rasterize(shapes, out_shape=(height, width))
    if not inference_flag:
        # Load ST data
        st_data_path = os.path.join(main_dir, args.processed_data_path, 'st', f"{image_name}.h5ad")
        st_data = sc.read_h5ad(st_data_path)
        #! Notice that this the first column of loc is the col index for the image matrix, which is the x axis direction on the plot!
        loc = st_data.obsm['spatial']
        bar = st_data.obs_names
        # Check spots on HE image
        spot_radius_pixel = metadata['spot_diameter']/ 2 / pixel_size_raw
        if args.plot:
            image_check = plot_image_with_spot(image, loc, spot_radius_pixel, mask_matrix)
            # Save image
            cv2.imwrite(f"{save_dir}/check_figure/{image_name}_raw_check.jpg", cv2.cvtColor(image_check, cv2.COLOR_RGB2BGR))
    else:
        loc = None
    # Cut and rescale image
    image_rescaled, mask_rescaled, loc_rescaled, para = cut_and_rescale_image(image, mask_matrix, loc, pixel_size_raw, target_pixel_size)
    print(f"    Rescaled image shape: {image_rescaled.shape}")
    if not inference_flag:
        # Check spots on HE image
        spot_radius_pixel *= pixel_size_raw/target_pixel_size
        if args.plot:
            image_check = plot_image_with_spot(image_rescaled, loc_rescaled, spot_radius_pixel, mask_rescaled)
            cv2.imwrite(f"{save_dir}/check_figure/{image_name}_rescaled_check.jpg", cv2.cvtColor(image_check, cv2.COLOR_RGB2BGR))
    bigtiff = False
    if height > 40000 and width > 60000:
        bigtiff = True
        print(f"    Image is too large, saving as BigTIFF")
    # Save image
    tifffile.imwrite(f"{save_dir}/wsis_rescaled/{image_name}_rescaled.tif", image_rescaled, compression='zlib', bigtiff=bigtiff)
    # Save mask
    tifffile.imwrite(f"{save_dir}/wsis_rescaled/{image_name}_mask_rescaled.tif", mask_rescaled, compression='zlib', bigtiff=bigtiff)
    if not inference_flag:
        # Save loc
        np.save(f"{save_dir}/wsis_rescaled/{image_name}_loc_rescaled.npy", loc_rescaled)
        # Save barcode
        utils.save_pickle(bar,f"{save_dir}/wsis_rescaled/{image_name}_barcode.pickle")
        np.save(f"{save_dir}/wsis_rescaled/{image_name}_pixel_size.npy", np.array([spot_radius_pixel]))
    rescale_parameters.loc[image_name,:] = list(para)
    print(f"    Finished processing {image_name}")

from datetime import datetime
# Get the current date and time
now = datetime.now()
# Format the datetime object into a string suitable for a filename
# Example format: YYYYMMDD-HHMMSS (e.g., 20251210-171530)
timestamp_string = now.strftime("%Y%m%d-%H%M%S")
rescale_parameters.to_csv(f"{save_dir}/rescale_parameters_{timestamp_string}.csv")