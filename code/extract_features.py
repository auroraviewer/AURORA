import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
# loading all packages here to start
from tqdm import tqdm
import tifffile
import utils
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import argparse
import gc
from image_utils import get_UNI_model, get_HEST1k_patch
from datasets import PatchDataset

parser = argparse.ArgumentParser(description='Rescale H&E images to target pixel size and adjust ST spot locations accordingly.')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size for processing')
parser.add_argument('--output_path', '-o', type=str, default='AURORA_interim', help='Output directory')
parser.add_argument('--patch_sizes', nargs='+', type=int, default=[3584, 896, 224], help='List of patch sizes')
parser.add_argument('--metadata_path', '-m', type=str, default='Sample_metadata.csv', help='Sample metadata CSV file name')
parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
parser.add_argument('--plot', action='store_true', help='Plot intermediate results for checking')
parser.add_argument('--token', '-t', type=str, default='', help='Hugging Face token')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
parser.add_argument('--inference', action='store_true', help='Inference mode flag')
parser.add_argument('--image_path', type=str, default='./wsis/', help='Data directory path')
parser.add_argument('--seg_path', type=str, default='./wsis_segmentation/', help='Save directory path')
args = parser.parse_args()

batch_size = args.batch_size # Choose 2048 or 4096 on A100, 256 or 512 on 2080Ti
num_workers = args.num_workers
main_dir = args.project_path
save_dir = f"{main_dir}/{args.output_path}/UNI_multiscale_patches"
patch_sizes = args.patch_sizes
inference_flag = args.inference
# Load UNI model
## Load the model and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model, transform = get_UNI_model(args.token, "UNI2-h")
model.to(device).eval()

os.makedirs(f"{save_dir}", exist_ok=True)
os.makedirs(f"{save_dir}/check_figure", exist_ok=True)
os.makedirs(f"{save_dir}/wsis_rescaled", exist_ok=True)
if not inference_flag:
    os.makedirs(f"{save_dir}/locs", exist_ok=True)
    meta = pd.read_csv(f"{main_dir}/{args.metadata_path}",index_col=0, header=0)
    image_files = list(meta.index)
else:
    meta = pd.read_csv(f"{main_dir}/{args.metadata_path}", header=0)
    image_files = list(meta['Image_name']) # image names with suffix
    meta.set_index('Image_name', inplace=True)

print(f"Start extracting features")
for image_file in tqdm(image_files, desc="Images"):
    if not inference_flag:
        image_name = image_file.split('.')[0]
    else:
        image_name = meta.loc[image_file,'Sample_name']
    if os.path.exists(f"{save_dir}/emb_{min(patch_sizes)}/{image_name}.npz"):
        print(f"{image_name} already processed")
        continue
    image_rescaled = tifffile.imread(f"{save_dir}/wsis_rescaled/{image_name}_rescaled.tif")
    mask_rescaled = tifffile.imread(f"{save_dir}/wsis_rescaled/{image_name}_mask_rescaled.tif")
    if not inference_flag:
        loc_rescaled = np.load(f"{save_dir}/wsis_rescaled/{image_name}_loc_rescaled.npy")
        spot_radius_pixel = np.load(f"{save_dir}/wsis_rescaled/{image_name}_pixel_size.npy")[0]
    else:
        loc_rescaled = None
        spot_radius_pixel = None
    print(f"Processing {image_name}")
    for patch_size in patch_sizes:
        print(f"    Patch size: {patch_size}")
        # Create dataset
        # mkdir(f"{main_dir}/HEST1K_data/UNI_multiscale_patches/{patch_size}patches")
        if patch_size == min(patch_sizes):
            dataset = PatchDataset(image_rescaled, mask_rescaled, transform, loc_rescaled, spot_radius_pixel, patch_size=patch_size, stride=patch_size)
        # elif patch_size == max(patch_sizes):
        #     dataset = PatchDataset(image_rescaled, mask_rescaled, transform, loc_rescaled, spot_radius_pixel, patch_size=patch_size, stride=patch_size//4)
        else:
            dataset = PatchDataset(image_rescaled, mask_rescaled, transform, loc_rescaled, spot_radius_pixel, patch_size=patch_size, stride=patch_size//2)
        if args.plot:
            image_check = dataset.plot_mask()
            cv2.imwrite(f"{save_dir}/check_figure/{image_name}_mask_check.jpg", image_check)
            image_check = dataset.plot_patch_loc()
            cv2.imwrite(f"{save_dir}/check_figure/{image_name}_{patch_size}patch_check.jpg", image_check)
        # print(f"    Total patches: {dataset.num_patches}")
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        os.makedirs(f"{save_dir}/emb_{patch_size}", exist_ok=True)

        patch_embeddings = []
        patch_positions_x = []
        patch_positions_y = []
        patch_flags = []
        with torch.no_grad():
            for batch_idx, (patches, positions, keep_flag) in enumerate(tqdm(dataloader, total=len(dataloader), desc="    Batches")):
                patches = patches.to(device, non_blocking=True)
                # if batch_idx == 0:
                #     print(f"    Shape of patches: {patches.shape}")
                #     print(f"    Shape of positions[0]: {positions[0].shape}")
                #     print(f"    Content of positions[0][:10]: {positions[0][:10]}")
                #     print(f"    Content of positions[1][:10]: {positions[1][:10]}")
                patch_emb = model(patches)
                # if batch_idx == 0:
                #     print(f"    Shape of patch_emb: {patch_emb.shape}")
                patch_embeddings.append(patch_emb.cpu().numpy())
                patch_positions_x.append(positions[0].numpy())
                patch_positions_y.append(positions[1].numpy())
                patch_flags.append(keep_flag)

        patch_embeddings = np.concatenate(patch_embeddings, axis=0)
        patch_positions = np.vstack((np.concatenate(patch_positions_x,axis=0),np.concatenate(patch_positions_y,axis=0))).T
        patch_flags = np.concatenate(patch_flags, axis=0)

        if args.plot:
            if patch_size == 224:
                kmeans = KMeans(n_clusters=5)
                kmeans.fit(patch_embeddings[patch_flags,:])
                plt.scatter(x=patch_positions[patch_flags,1],
                            y=-patch_positions[patch_flags,0],
                            c=kmeans.labels_,
                            marker='s',
                            cmap='tab20',
                            s=20)
                plt.savefig(f"{save_dir}/check_figure/{image_name}_emb_{patch_size}_kmeans.png")
                plt.close()

        np.savez(f"{save_dir}/emb_{patch_size}/{image_name}.npz",
                emb=patch_embeddings,
                pos=patch_positions,
                flag=patch_flags,
                pad=(dataset.row_pad, dataset.col_pad))
        gc.collect()
        torch.cuda.empty_cache()
    print(f"    Finished extracting features for {image_name}")
    if not inference_flag:
        img, img_meta, barcode, coords = get_HEST1k_patch(os.path.join(main_dir, args.processed_data_path, 'patches', f"{image_name}.h5"))
        utils.save_pickle(barcode,f"{save_dir}/locs/{image_name}-bars.pickle")
        utils.save_pickle(coords,f"{save_dir}/locs/{image_name}-locs.pickle")
        utils.save_pickle(img_meta,f"{save_dir}/locs/{image_name}-meta.pickle")