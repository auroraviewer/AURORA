import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
# loading all packages here to start
from tqdm import tqdm
import tifffile
import utils
# NOTE (aurora 2.0 reorganisation): matplotlib and scikit-learn are only needed
# by the --plot branch.  Importing them at module level made them mandatory for
# inference-only users, so they are now imported inside that branch.
# Original statements preserved for reference:
#   import matplotlib.pyplot as plt
#   from sklearn.cluster import KMeans
import cv2
import argparse
import gc
from image_utils import get_UNI_model, get_HEST1k_patch
from datasets import PatchDataset


def plot_embedding_kmeans(patch_embeddings, patch_flags, patch_positions, save_path):
    """Debug KMeans scatter plot used by ``--plot``.

    NOTE (aurora 2.0 reorganisation): matplotlib and scikit-learn are optional
    here -- neither is needed to extract features.  Their import is deferred into
    this helper so an inference-only installation does not have to provide
    scikit-learn; when it is missing the check figure is simply skipped.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
    except ImportError as exc:
        print(f"    Skipping the KMeans check figure ({exc})")
        return
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(patch_embeddings[patch_flags,:])
    plt.scatter(x=patch_positions[patch_flags,1],
                y=-patch_positions[patch_flags,0],
                c=kmeans.labels_,
                marker='s',
                cmap='tab20',
                s=20)
    plt.savefig(save_path)
    plt.close()

parser = argparse.ArgumentParser(description='Rescale H&E images to target pixel size and adjust ST spot locations accordingly.')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size for processing')
parser.add_argument('--output_path', '-o', type=str, default='AURORA_interim', help='Output directory')
parser.add_argument('--patch_sizes', nargs='+', type=int, default=[3584, 896, 224], help='List of patch sizes')
parser.add_argument('--metadata_path', '-m', type=str, default='Sample_metadata.csv', help='Sample metadata CSV file name')
parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
parser.add_argument('--plot', action='store_true', help='Plot intermediate results for checking')
parser.add_argument('--token', '-t', type=str, default=None, help='Hugging Face token')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
parser.add_argument('--inference', action='store_true', help='Inference mode flag')
parser.add_argument('--image_path', type=str, default='./wsis/', help='Data directory path')
parser.add_argument('--seg_path', type=str, default='./wsis_segmentation/', help='Save directory path')
parser.add_argument('--UNI_local_path', type=str, default=None, help='Path to the local UNI model folder')
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
if args.token is None:
    if args.UNI_local_path is None:
        raise ValueError("Either --token or --UNI_local_path must be provided.")
    model, transform = get_UNI_model(args.UNI_local_path, "UNI2-h", from_local=True)
else:
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
    # An embeddings file alone does not mean the sample finished: the barcode /
    # coordinate pickles are written *after* the extraction loop, so a crash in
    # between leaves a state this check would skip forever. In training mode require
    # the pickles too, and redo the sample if they are missing.
    # if os.path.exists(f"{save_dir}/emb_{min(patch_sizes)}/{image_name}.npz"):
    #     print(f"{image_name} already processed")
    #     continue
    already_processed = os.path.exists(f"{save_dir}/emb_{min(patch_sizes)}/{image_name}.npz")
    if already_processed and not inference_flag:
        side_files = [
            f"{save_dir}/locs/{image_name}-{suffix}.pickle"
            for suffix in ("bars", "locs", "meta")
        ]
        missing_side = [p for p in side_files if not os.path.exists(p)]
        if missing_side:
            print(
                f"{image_name} has embeddings but is missing "
                f"{', '.join(os.path.basename(p) for p in missing_side)}; reprocessing"
            )
            already_processed = False
    if already_processed:
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
                plot_embedding_kmeans(
                    patch_embeddings,
                    patch_flags,
                    patch_positions,
                    f"{save_dir}/check_figure/{image_name}_emb_{patch_size}_kmeans.png",
                )

        np.savez(f"{save_dir}/emb_{patch_size}/{image_name}.npz",
                emb=patch_embeddings,
                pos=patch_positions,
                flag=patch_flags,
                pad=(dataset.row_pad, dataset.col_pad))
        gc.collect()
        torch.cuda.empty_cache()
    print(f"    Finished extracting features for {image_name}")
    if not inference_flag:
        # preprocess (HEST_preprocess.py) names the dump after the sheet's Sample
        # column, while `image_name` above is the Image_name stem. They need not be
        # the same, so try both names instead of assuming one.
        # img, img_meta, barcode, coords = get_HEST1k_patch(os.path.join(main_dir, args.processed_data_path, 'patches', f"{image_name}.h5"))
        patch_candidates = [f"{image_name}.h5"]
        try:
            sample_name = meta.loc[image_file, 'Sample_name']
            patch_candidates.append(f"{sample_name}.h5")
        except Exception:
            pass
        try:
            sample_col = meta.loc[image_file, 'Sample']
            patch_candidates.append(f"{sample_col}.h5")
        except Exception:
            pass
        patch_dir = os.path.join(main_dir, args.processed_data_path, 'patches')
        patch_path = next(
            (os.path.join(patch_dir, name) for name in patch_candidates
             if os.path.exists(os.path.join(patch_dir, name))),
            os.path.join(patch_dir, patch_candidates[0]),
        )
        img, img_meta, barcode, coords = get_HEST1k_patch(patch_path)
        utils.save_pickle(barcode,f"{save_dir}/locs/{image_name}-bars.pickle")
        utils.save_pickle(coords,f"{save_dir}/locs/{image_name}-locs.pickle")
        utils.save_pickle(img_meta,f"{save_dir}/locs/{image_name}-meta.pickle")