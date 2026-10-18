import torch
import os
import subprocess
import re
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
# loading all packages here to start
from tqdm import tqdm
import tifffile
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import argparse
import gc
# from image_utils import get_UNI_model, get_HEST1k_patch
from datasets import PatchDataset
import yaml
from image_utils import get_UNI_model, extract_highres_features, smooth_embedding_halves, _merge_embeddings_into_state, _finalize_patch_state
import cv2
import shutil

parser = argparse.ArgumentParser(description='Rescale H&E images to target pixel size and adjust ST spot locations accordingly.')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--code_path', '-c', type=str, default='./AURORA/code/', help='Code directory path')
parser.add_argument('--catch_path', type=str, default='./CATCH/', help='CATCH code directory path')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size for processing')
parser.add_argument('--output_path', '-o', type=str, default='AURORA_interim', help='Output directory')
parser.add_argument('--patch_sizes', nargs='+', type=int, default=[56,14], help='List of patch sizes')
parser.add_argument('--metadata_path', '-m', type=str, default='Sample_metadata.csv', help='Sample metadata CSV file name')
parser.add_argument('--sample_name_col', type=int, default=0, help='Column index for sample names in metadata CSV')
# parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
parser.add_argument('--plot', action='store_true', help='Plot intermediate results for checking')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
parser.add_argument('--inference', action='store_true', help='Inference mode flag')
parser.add_argument('--stride', type=int, default=112, help='Stride for patch extraction')
# parser.add_argument('--image_path', type=str, default='./wsis/', help='Data directory path')
# parser.add_argument('--seg_path', type=str, default='./wsis_segmentation/', help='Save directory path')
parser.add_argument('--UNI_local_path', type=str, default=None, help='Path to the local UNI model folder')
parser.add_argument('--CATCH_config_templet_path', type=str, default='CATCH_config_templet.yaml', help='Path to the CATCH config file')
parser.add_argument('--use-CATCH', action='store_true', help='Use CATCH for feature extraction')
args = parser.parse_args()


# batch_size = args.batch_size # Choose 2048 or 4096 on A100, 256 or 512 on 2080Ti
main_dir = args.project_path
save_dir = f"{main_dir}/{args.output_path}/UNI_multiscale_patches"
patch_sizes = args.patch_sizes
inference_flag = args.inference
# Load UNI model
## Load the model and set the device
if args.UNI_local_path is None:
    raise ValueError("--UNI_local_path must be provided.")

os.makedirs(f"{save_dir}", exist_ok=True)
os.makedirs(f"{save_dir}/check_figure", exist_ok=True)
# os.makedirs(f"{save_dir}/wsis_rescaled", exist_ok=True)
if not inference_flag:
    # os.makedirs(f"{save_dir}/locs", exist_ok=True)
    meta = pd.read_csv(f"{main_dir}/{args.metadata_path}",index_col=0, header=0)
    image_files = list(meta.index)
else:
    meta = pd.read_csv(f"{main_dir}/{args.metadata_path}", header=0)
    image_files = meta.iloc[:, args.sample_name_col].values # image names with suffix
    meta.set_index(meta.columns[args.sample_name_col], inplace=True)

if args.use_CATCH:
    import pyvips
    print(f"Start extracting features using CATCH")
    for image_file in tqdm(image_files, desc="Images"):
        image_name = image_file.split('.')[0]
        if os.path.exists(f"{save_dir}/emb_{min(patch_sizes)}/{image_name}.npz"):
            print(f"{image_name} already processed")
            continue
        image_rescaled = tifffile.imread(f"{save_dir}/wsis_rescaled/{image_name}_rescaled.tif")
        mask_rescaled = tifffile.imread(f"{save_dir}/wsis_rescaled/{image_name}_mask_rescaled.tif")
        # if not inference_flag:
        #     loc_rescaled = np.load(f"{save_dir}/wsis_rescaled/{image_name}_loc_rescaled.npy")
        #     spot_radius_pixel = np.load(f"{save_dir}/wsis_rescaled/{image_name}_pixel_size.npy")[0]
        # else:
        #     loc_rescaled = None
        #     spot_radius_pixel = None
        print(f"Processing {image_name}")
        for patch_size in patch_sizes:
            print(f"    Patch size: {patch_size}")
            # # Create dataset
            # # mkdir(f"{main_dir}/HEST1K_data/UNI_multiscale_patches/{patch_size}patches")
            # if patch_size == min(patch_sizes):
            #     dataset = PatchDataset(image_rescaled, mask_rescaled, transform, loc_rescaled, spot_radius_pixel, patch_size=patch_size, stride=patch_size)
            # # elif patch_size == max(patch_sizes):
            # #     dataset = PatchDataset(image_rescaled, mask_rescaled, transform, loc_rescaled, spot_radius_pixel, patch_size=patch_size, stride=patch_size//4)
            # else:
            #     dataset = PatchDataset(image_rescaled, mask_rescaled, transform, loc_rescaled, spot_radius_pixel, patch_size=patch_size, stride=patch_size//2)
            # if args.plot:
            #     image_check = dataset.plot_mask()
            #     cv2.imwrite(f"{save_dir}/check_figure/{image_name}_mask_check.jpg", image_check)
            #     image_check = dataset.plot_patch_loc()
            #     cv2.imwrite(f"{save_dir}/check_figure/{image_name}_{patch_size}patch_check.jpg", image_check)
            # # print(f"    Total patches: {dataset.num_patches}")
            # dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

            os.makedirs(f"{save_dir}/emb_{patch_size}", exist_ok=True)
            os.makedirs(f"{save_dir}/emb_{patch_size}/CATCH_tmp", exist_ok=True)
            scale_factor = 14/patch_size
            image_use = pyvips.Image.new_from_array(image_rescaled).resize(scale_factor)
            mask_use = pyvips.Image.new_from_array(mask_rescaled.astype('float32')).resize(scale_factor)
            # image_use = cv2.resize(image_rescaled, (0,0), fx=scale_factor, fy=scale_factor)
            # mask_use = cv2.resize(mask_rescaled.astype('float32'), (0,0), fx=scale_factor, fy=scale_factor)
            
            image_use.write_to_file(f"{save_dir}/emb_{patch_size}/CATCH_tmp/{image_name}_rescaled.tif")
            mask_use.write_to_file(f"{save_dir}/emb_{patch_size}/CATCH_tmp/{image_name}_mask_rescaled.tif")
            # cv2.imwrite(f"{save_dir}/emb_{patch_size}/CATCH_tmp/{image_name}_rescaled.tif", image_use)
            # cv2.imwrite(f"{save_dir}/emb_{patch_size}/CATCH_tmp/{image_name}_mask_rescaled.tif", mask_use)
            with open(f"{args.CATCH_config_templet_path}", 'r') as f:
                config = yaml.safe_load(f)
            config['global']['dataset'] = f'CATCH_tmp'
            config['global']['sample_list'] = [f'{image_name}']
            config['global']['read_base_dir'] = f'{save_dir}/emb_{patch_size}'
            config['global']['result_base_dir'] = f'{save_dir}/emb_{patch_size}'
            config['global']['weight_dir'] = args.UNI_local_path
            with open(f"{save_dir}/emb_{patch_size}/CATCH_tmp/config.yaml", "w") as f:
                yaml.dump(config, f)
            
            result = subprocess.run(f"python -u {args.code_path}/run_CATCH.py --config {save_dir}/emb_{patch_size}/CATCH_tmp/config.yaml --catch_path {args.catch_path}", shell=True, executable="/bin/bash")
            if result.returncode != 0:
                print(f"Error in running CATCH for {image_name} at patch size {patch_size}")
                continue

            ## remove tmp directory
            subprocess.run(f"mv {save_dir}/emb_{patch_size}/CATCH_tmp/{image_name}_emb14.npz {save_dir}/emb_{patch_size}/{image_name}.npz", shell=True, executable="/bin/bash")
            subprocess.run(f"rm -r {save_dir}/emb_{patch_size}/CATCH_tmp", shell=True, executable="/bin/bash")
            if args.plot:
                catch_result = np.load(f"{save_dir}/emb_{patch_size}/{image_name}.npz")
                patch_embeddings = catch_result['emb']
                patch_positions = catch_result['pos']
                patch_flags = catch_result['flag']
                print(f"    Plotting KMeans for checking")
                kmeans = KMeans(n_clusters=5)
                kmeans.fit(patch_embeddings[patch_flags,:])
                plt.scatter(x=patch_positions[patch_flags,1],
                            y=-patch_positions[patch_flags,0],
                            c=kmeans.labels_,
                            marker=',',
                            cmap='tab20',
                            s=0.1)
                plt.savefig(f"{save_dir}/check_figure/{image_name}_emb_{patch_size}_kmeans.png")
                plt.close()
            ## Correct locs
            if patch_size != 14:
                catch_result = np.load(f"{save_dir}/emb_{patch_size}/{image_name}.npz")
                pad = catch_result['pad']
                patch_positions = catch_result['pos']
                patch_positions_before_pad = patch_positions - [pad[0]//2, pad[1]//2]
                patch_positions_corrected = patch_positions_before_pad * (patch_size/14)
                patch_positions_corrected = patch_positions_corrected.astype(int)
                catch_result_corrected = {'emb': catch_result['emb'],
                                        'pos': patch_positions_corrected,
                                        'flag': catch_result['flag'],
                                        'pad': [0,0],
                                        'raw_pos': catch_result['pos'],
                                        'raw_pad': catch_result['pad']}
                np.savez(f"{save_dir}/emb_{patch_size}/{image_name}.npz", **catch_result_corrected)
            gc.collect()
        print(f"    Finished extracting features for {image_name}")
        # if not inference_flag:
        #     img, img_meta, barcode, coords = get_HEST1k_patch(os.path.join(main_dir, args.processed_data_path, 'patches', f"{image_name}.h5"))
        #     utils.save_pickle(barcode,f"{save_dir}/locs/{image_name}-bars.pickle")
        #     utils.save_pickle(coords,f"{save_dir}/locs/{image_name}-locs.pickle")
        #     utils.save_pickle(img_meta,f"{save_dir}/locs/{image_name}-meta.pickle")
else:
    print(f"Start extracting features using UNI")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # if args.token is None:
    if args.UNI_local_path is None:
        raise ValueError("Either --token or --UNI_local_path must be provided.")
    model, transform = get_UNI_model(args.UNI_local_path, "UNI2-h", from_local=True)
    # else:
    #     model, transform = get_UNI_model(args.token, "UNI2-h")
    model.to(device).eval()
    model.requires_grad_(False)
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    batch_size = args.batch_size
    for image_file in tqdm(image_files, desc="Images"):
        image_name = image_file.split('.')[0]
        if os.path.exists(f"{save_dir}/emb_{min(patch_sizes)}/{image_name}.npz"):
            print(f"{image_name} already processed")
            continue
        print(f"Processing {image_name}")
        image_rescaled = tifffile.imread(f"{save_dir}/wsis_rescaled/{image_name}_rescaled.tif")
        mask_rescaled = tifffile.imread(f"{save_dir}/wsis_rescaled/{image_name}_mask_rescaled.tif")
        if not inference_flag:
            loc_rescaled = np.load(f"{save_dir}/wsis_rescaled/{image_name}_loc_rescaled.npy")
            spot_radius_pixel = np.load(f"{save_dir}/wsis_rescaled/{image_name}_pixel_size.npy")[0]
        else:
            loc_rescaled = None
            spot_radius_pixel = None
        dataset = PatchDataset(image_rescaled, 
                                mask_rescaled, 
                                transform,
                                #! Notice that this the first column of loc is the col index for the image matrix
                                loc_rescaled, 
                                spot_radius_pixel, patch_size=224, stride=args.stride, high_res=True)
        if args.plot:
            image_check = dataset.plot_mask()
            cv2.imwrite(f"{save_dir}/check_figure/{image_name}_mask_check.jpg", image_check)
            image_check = dataset.plot_patch_loc()
            cv2.imwrite(f"{save_dir}/check_figure/{image_name}_highres_patch_check.jpg", image_check)
        # print(f"    Total patches: {dataset.num_patches}")
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True)

        patch_size_states = {
            patch_size: {"sum": {}, "count": {}}
            for patch_size in patch_sizes
        }
        feature_dim = None

        with torch.no_grad():
            for batch_idx, (patches, positions, keep_flag) in enumerate(tqdm(dataloader, total=len(dataloader), desc="    Batches")):
                patches = patches.to(device, non_blocking=True)
                feature_emb, patch_emb = extract_highres_features(model, patches, amp_dtype, device)
                # Process each patch
                for idx in range(len(positions[0])):
                    # Extract features
                    center_feature = feature_emb[idx, :]  # Use the [CLS] token as the 224-level feature
                    patch_feature = patch_emb[idx, :, :, :]
                    patch_x = positions[0][idx].item()
                    patch_y = positions[1][idx].item()
                    valid_coords = torch.nonzero(keep_flag[idx], as_tuple=False)
                    if valid_coords.numel() == 0:
                        continue

                    subpixel_features = patch_feature[:, valid_coords[:, 0], valid_coords[:, 1]].T
                    center_features = center_feature.unsqueeze(0).expand(valid_coords.shape[0], -1)
                    combined_features = torch.cat([center_features, subpixel_features], dim=1)
                    combined_features = combined_features.cpu().numpy().astype(np.float16)

                    valid_coords_np = valid_coords.cpu().numpy().astype(np.int32)
                    patch_positions_curr = np.empty((valid_coords_np.shape[0], 2), dtype=np.int32)
                    patch_positions_curr[:, 0] = patch_x + 14 * valid_coords_np[:, 0]
                    patch_positions_curr[:, 1] = patch_y + 14 * valid_coords_np[:, 1]

                    if feature_dim is None:
                        feature_dim = combined_features.shape[1]

                    for patch_size in patch_sizes:
                        _merge_embeddings_into_state(
                            patch_size_states[patch_size],
                            combined_features,
                            patch_positions_curr,
                            patch_size,
                        )

        pad = (dataset.row_pad, dataset.col_pad)

        for patch_size in patch_sizes:
            print(f"    Summarizing results for patch size: {patch_size}")
            os.makedirs(f"{save_dir}/emb_{patch_size}", exist_ok=True)
            patch_embeddings_curr, patch_positions_curr, patch_flags_curr = _finalize_patch_state(
                patch_size_states[patch_size],
                patch_size,
            )
            print(f"    Deduplicated patches: {patch_embeddings_curr.shape[0]} unique positions")
            gc.collect()
            torch.cuda.empty_cache()
            if args.plot:
                print(f"    Plotting KMeans for checking")
                kmeans = MiniBatchKMeans(
                    n_clusters=20,
                    batch_size=4096,
                    n_init="auto",
                    random_state=0,
                )
                kmeans.fit(patch_embeddings_curr[patch_flags_curr,:].astype(np.float32))
                labels = np.full(len(patch_embeddings_curr), -1, dtype=np.int32)
                labels[patch_flags_curr] = kmeans.labels_
                kmeans_img = np.zeros((image_rescaled.shape[0], image_rescaled.shape[1], 3), dtype=np.uint8)
                tab20 = (plt.get_cmap('tab20')(np.arange(20))[:, :3] * 255).astype(np.uint8)
                for idx, (x, y) in enumerate(patch_positions_curr[patch_flags_curr,:].astype(int)):
                    if labels[idx] >= 0:
                        x_start = max(0,x)
                        x_end = min(image_rescaled.shape[0], x+patch_size)
                        y_start = max(0,y)
                        y_end = min(image_rescaled.shape[1], y+patch_size)
                        kmeans_img[x_start:x_end, y_start:y_end, :] = tab20[labels[idx] % 10][::-1]
                kmeans_img = cv2.resize(kmeans_img, (0,0), fx = 0.1, fy = 0.1)
                cv2.imwrite(f"{save_dir}/check_figure/{image_name}_emb_{patch_size}_kmeans.png", kmeans_img)
            np.savez(f"{save_dir}/emb_{patch_size}/{image_name}_raw.npz",
                                emb=patch_embeddings_curr,
                                pos=patch_positions_curr,
                                flag=patch_flags_curr,
                                pad=pad)
            print(f"    Smoothing")
            if patch_size == 14:
                smoothed_embeddings = smooth_embedding_halves(
                            patch_embeddings_curr,
                            patch_positions_curr,
                            patch_flags_curr,
                            position_unit=patch_size,
                            first_size=16,
                            second_size=4)
            else:
                smoothed_embeddings = smooth_embedding_halves(
                            patch_embeddings_curr,
                            patch_positions_curr,
                            patch_flags_curr,
                            position_unit=patch_size,
                            first_size=4,
                            second_size=2)
            smoothed_embeddings = smoothed_embeddings.astype(np.float16)
            np.savez(f"{save_dir}/emb_{patch_size}/{image_name}.npz",
                    emb=smoothed_embeddings,
                    pos=patch_positions_curr,
                    flag=patch_flags_curr,
                    pad=pad)
            if args.plot:
                print(f"    Plotting KMeans for checking")
                kmeans = MiniBatchKMeans(
                    n_clusters=20,
                    batch_size=4096,
                    n_init="auto",
                    random_state=0,
                )
                kmeans.fit(smoothed_embeddings[patch_flags_curr,:].astype(np.float32))
                labels = np.full(len(smoothed_embeddings), -1, dtype=np.int32)
                labels[patch_flags_curr] = kmeans.labels_
                kmeans_img = np.zeros((image_rescaled.shape[0], image_rescaled.shape[1], 3), dtype=np.uint8)
                tab20 = (plt.get_cmap('tab20')(np.arange(20))[:, :3] * 255).astype(np.uint8)
                for idx, (x, y) in enumerate(patch_positions_curr[patch_flags_curr,:].astype(int)):
                    if labels[idx] >= 0:
                        x_start = max(0,x)
                        x_end = min(image_rescaled.shape[0], x+patch_size)
                        y_start = max(0,y)
                        y_end = min(image_rescaled.shape[1], y+patch_size)
                        kmeans_img[x_start:x_end, y_start:y_end, :] = tab20[labels[idx] % 10][::-1]
                kmeans_img = cv2.resize(kmeans_img, (0,0), fx = 0.1, fy = 0.1)
                cv2.imwrite(f"{save_dir}/check_figure/{image_name}_smoothed_emb_{patch_size}_kmeans.png", kmeans_img)

        print(f"    Finished extracting features for {image_name}")