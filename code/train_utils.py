import json
# NOTE (aurora 2.0 reorganisation): scikit-learn is a TRAINING-ONLY dependency.
# It used to be imported at module import time, which silently forced every
# inference-only user to install it, because train_utils is imported by
# predict.py / predict_highres.py / datasets.py.  The import is now resolved on
# first attribute access (PEP 562), so `train_utils.MiniBatchKMeans` and
# `from train_utils import MiniBatchKMeans` keep working exactly as before
# whenever scikit-learn is installed, and an inference-only environment no
# longer needs it.  The original statement is preserved below.
#   from sklearn.cluster import MiniBatchKMeans
_LAZY_SKLEARN_NAMES = {
    "MiniBatchKMeans": ("sklearn.cluster", "MiniBatchKMeans"),
    "KMeans": ("sklearn.cluster", "KMeans"),
    "AgglomerativeClustering": ("sklearn.cluster", "AgglomerativeClustering"),
    "GaussianMixture": ("sklearn.mixture", "GaussianMixture"),
}


def __getattr__(name):
    """Resolve the scikit-learn names above lazily, on first use."""
    target = _LAZY_SKLEARN_NAMES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


from tqdm import tqdm
import tifffile
import os
import numpy as np
import pandas as pd
import cv2
import gc
import threading
import queue
from collections import OrderedDict
from scipy.spatial import KDTree
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import torch
import torch.nn.functional as F
import utils
import json
def load_image(dataset, main_dir, processed_data_dir, interim_dir, target_radius=0.5):
    print(f"Loading image {dataset}...")
    # Load image metadata from json
    with open(f"{main_dir}/{processed_data_dir}/metadata/{dataset}.json", 'r') as f:
    # Load the JSON data into a Python dictionary
        metadata = json.load(f)
    spot_radius_pixel = metadata['spot_diameter']/ 2 / target_radius
    print(f"    Raw pixel size: {target_radius}")
    print(f"    Spot radius in pixels: {spot_radius_pixel}")
    # Load image
    image_path = f"{main_dir}/{interim_dir}/UNI_multiscale_patches/wsis_rescaled/{dataset}_rescaled.tif"
    mask_path = f"{main_dir}/{interim_dir}/UNI_multiscale_patches/wsis_rescaled/{dataset}_mask_rescaled.tif"
    with tifffile.TiffFile(image_path) as tif:
        # Access the image data at different levels
        image = tif.pages[0].asarray()
    with tifffile.TiffFile(mask_path) as tif:
        # Access the image data at different levels
        mask = tif.pages[0].asarray()
    image[mask == 0] = 255
    return image, spot_radius_pixel

def plot_image_with_spot(image, loc, spot_radius_pixel):
    # Create a copy of the image to draw spots with transparency
    image_with_spots = image.copy()
    # Draw spots on the image
    for (x, y) in loc:
        image_with_spots = cv2.circle(image_with_spots, (int(x), int(y)), int(spot_radius_pixel), (0, 255, 0), -1)
    # Create a mask for the spots
    spots_mask = np.zeros_like(image_with_spots, dtype=np.uint8)
    for (x, y) in loc:
        spots_mask = cv2.circle(spots_mask, (int(x), int(y)), int(spot_radius_pixel), (0, 255, 0), -1)
    # Blend the spots with the original image
    image_with_spots = cv2.addWeighted(image, 0.7, image_with_spots, 0.3, 0)
    return image_with_spots

def plot_image_with_patch(image, loc, patch_size, stroke_width=5):
    # Create a copy of the image to draw spots with transparency
    image_with_spots = image.copy()
    # Draw spots on the image
    for (x, y) in loc:
        image_with_spots = cv2.rectangle(image_with_spots, (int(x), int(y)), (int(x)+patch_size, int(y)+patch_size), (0, 255, 0), stroke_width)
    # Create a mask for the spots
    spots_mask = np.zeros_like(image_with_spots, dtype=np.uint8)
    for (x, y) in loc:
        spots_mask = cv2.rectangle(spots_mask, (int(x), int(y)), (int(x)+patch_size, int(y)+patch_size), (0, 255, 0), stroke_width)
    # Blend the spots with the original image
    image_with_spots = cv2.addWeighted(image, 0.7, image_with_spots, 0.3, 0)
    return image_with_spots

def get_unique_indices(tensor, device):
    unique_value, inverse_indices = torch.unique(tensor, return_inverse=True)
    # Use argsort with stable=True to maintain original order for equal values
    indices_sorted_by_appearance = inverse_indices.argsort(stable=True)
    # Get the indices of the first occurrences
    first_occurrence_indices = indices_sorted_by_appearance[torch.cat([torch.tensor([0],device=device), torch.diff(inverse_indices[indices_sorted_by_appearance]).nonzero().flatten() + 1])]
    # Sort the first occurrence indices to match the order of unique values
    return unique_value, first_occurrence_indices.sort()[0]

def query_batch(tree,
                query_points: np.ndarray,
                k=4) -> np.ndarray:
    return tree.query(query_points,k=k)[1]


def query_kdtree_parallel(tree,
                          query_points: np.ndarray,
                          k=4,
                          workers: int = 1) -> np.ndarray:
    # Split query points into multiple batches for parallel querying
    split_indices = np.array_split(query_points, workers)
    # Use ProcessPoolExecutor for parallel querying
    with ProcessPoolExecutor(max_workers=workers) as executor:
        partial_query_batch = partial(query_batch, tree, k=k)
        futures = [executor.submit(partial_query_batch, points) for points in split_indices]
        results = [future.result() for future in futures]
    # Merge results
    nearest_indices = np.concatenate(results)
    return nearest_indices

def find_nearby_spots(spot_coord: np.ndarray,
                      query: np.ndarray,
                      k=4,
                      workers=4) -> np.ndarray:
        # KDTree for nearest spot querying
        tree = KDTree(spot_coord)
        return query_kdtree_parallel(tree, query,k=k, workers=workers)

def square_intersection_area(center1, side1, center2, side2):
    """
    Calculates the intersection area of two squares.
    Args:
        center1: Tuple (x, y) representing the center of the first square.
        side1: Side length of the first square.
        center2: Tuple (x, y) representing the center of the second square.
        side2: Side length of the second square.
    Returns:
        The intersection area of the two squares.
    """
    x1_min = center1[0] - side1 / 2
    y1_min = center1[1] - side1 / 2
    x1_max = center1[0] + side1 / 2
    y1_max = center1[1] + side1 / 2
    x2_min = center2[0] - side2 / 2
    y2_min = center2[1] - side2 / 2
    x2_max = center2[0] + side2 / 2
    y2_max = center2[1] + side2 / 2
    # Calculate intersection boundaries
    x_intersect_min = max(x1_min, x2_min)
    y_intersect_min = max(y1_min, y2_min)
    x_intersect_max = min(x1_max, x2_max)
    y_intersect_max = min(y1_max, y2_max)
    # If there is no overlap, return 0
    if x_intersect_min >= x_intersect_max or y_intersect_min >= y_intersect_max:
        return 0
    # Calculate intersection area
    intersection_area = (x_intersect_max - x_intersect_min) * (y_intersect_max - y_intersect_min)
    return intersection_area

def js_divergence(P_log, Q, reduction='batchmean'):
    """
    Computes Jensen-Shannon Divergence between two distributions.
    Args:
        P_log: log-probabilities from model (log P)
        Q: target probabilities (Q)
        reduction: 'batchmean', 'mean', or 'none'
    Returns:
        JSD between P and Q
    """
    # Convert log P to P
    P = P_log.exp()
    # Mixture distribution M = 0.5 * (P + Q)
    M = 0.5 * (P + Q)
    # Compute log M (to use in KL terms)
    log_M = torch.log(M + 1e-8)
    # KL(P || M)
    kl_pm = F.kl_div(log_M, P, reduction='none')  # note: first arg = log_target
    # KL(Q || M)
    kl_qm = F.kl_div(log_M, Q, reduction='none')
    jsd = 0.5 * (kl_pm + kl_qm)
    if reduction == 'batchmean':
        return jsd.sum() / P.shape[0]
    elif reduction == 'mean':
        return jsd.mean()
    elif reduction == 'none':
        return jsd
    else:
        raise ValueError("Unsupported reduction type.")

def reverse_kl(log_probs, target, reduction='batchmean'):
    probs = log_probs.exp()
    return F.kl_div(torch.log(target + 1e-8), probs, reduction=reduction, log_target = False)

def js_reverse_mix(P_log, Q, reduction='batchmean', rkl_weight=0.01):
    return js_divergence(P_log, Q, reduction=reduction) + rkl_weight * reverse_kl(P_log, Q)

def wasserstein_loss(P_log, Q, backend_fun):
    return backend_fun(torch.exp(P_log), Q)

def masked_focal_kl_loss(log_pred, target_probs, gamma=2.0, mask_thresh=0.2):
    """
    Masked focal version of KLDivLoss for distribution prediction.
    pred_probs: [B, C] (probabilities from softmax)
    target_probs: [B, C] (target distribution, sums to 1)
    gamma: focal exponent (default 2.0)
    mask_thresh: minimum threshold in target to consider a class important
    """
    pred_probs = log_pred.exp()  # shape: [B, C]
    # Standard KL: KL(target || pred) = sum target * log(target / pred)
    kl_elementwise = target_probs * (torch.log(target_probs + 1e-8) - log_pred)  # shape: [B, C]
    # Mask: only focus on target classes with meaningful mass
    mask = (target_probs > mask_thresh).float()  # shape: [B, C]
    # Focal weight: boost areas where prediction is confidently wrong
    focal_weight = ((1.0 - pred_probs) ** gamma) * mask  # shape: [B, C]
    # Apply focal mask to KL loss
    masked_kl = focal_weight * kl_elementwise  # shape: [B, C]
    # Sum over classes, average over batch
    return masked_kl.sum(dim=1).mean()

# Function to load embeddings and gene expression data
def load_data(datasets, gene_list, main_dir, processed_data_dir,
              interim_dir, train_dir,
              bulk_hvg, bulk_norm_factor, cell_type_list, load_cell_type = True,
              patch_sizes=[3584, 896, 224], plot_mask = True, 
              from_file = False, file_dir = None, num_workers=4,
              pixel_size=0.5,
              barcode_dir=None, load_graph=False, rand_count=False,
              smooth_cnts=False, patch_folder_name = "UNI_multiscale_patches",
              cnts_smooth_folder_name = "cnts_smooth",
              bulk_folder_name = "bulk", prop_folder_name = "deconvolution"):
    if not from_file:
        import os
        os.makedirs(f"{main_dir}/{train_dir}/image_check/",exist_ok=True)
        # We load the intersection between gene_list and gene detected in the dataset. So, each dataset's loss have different number of genes involved.
        data = []
        emb_folder = f"{main_dir}/{interim_dir}/{patch_folder_name}"
        if not smooth_cnts:
            cnt_folder = f"{main_dir}/{interim_dir}/cnts"
        else:
            cnt_folder = f"{main_dir}/{interim_dir}/{cnts_smooth_folder_name}"
        bulk_folder = f"{main_dir}/{interim_dir}/{bulk_folder_name}"
        prop_folder = f"{main_dir}/{interim_dir}/{prop_folder_name}"
        for dataset in datasets:
            dataset = dataset.replace(".h5", "")
            # if plot_mask:
            image, spot_radius_pixel = load_image(dataset, main_dir = main_dir, processed_data_dir = processed_data_dir, 
                                                    interim_dir = interim_dir, target_radius=pixel_size)
            # Load spot locations
            locs_file = f"{emb_folder}/wsis_rescaled/{dataset}_loc_rescaled.npy"
            barcode = utils.load_pickle(f"{emb_folder}/wsis_rescaled/{dataset}_barcode.pickle")
            #! Notice that this the first column of loc is the col index for the image matrix, which is the x axis direction on the plot!
            spot_locs = np.load(locs_file)
            # Load cell type proportions
            if load_cell_type:
                major_prop = pd.read_csv(f"{prop_folder}/{dataset}_prop.csv",index_col=0,header=0)
            # Only keep spots used in cnts
            if barcode_dir is None:
                barcode_use = utils.load_pickle(f"{main_dir}/{interim_dir}/UNI_multiscale_patches/locs/{dataset}-bars.pickle")
            else:
                barcode_use = utils.load_pickle(f"{barcode_dir}/{dataset}-bars.pickle")
            barcode_use = [code.decode('utf-8') for code in barcode_use.flatten()]
            barcode_flag = np.isin(barcode,barcode_use)
            barcode = barcode[barcode_flag]
            spot_locs = spot_locs[barcode_flag,:]
            # Load embeddings
            embs = {}
            window_masks = {}
            spot_weight = {}
            patches_loc_before_pad = {}
            father_patches = {}
            if load_graph:
                edge_index_list = {}
            for patch_size in patch_sizes:
                tmp = np.load(f"{emb_folder}/emb_{patch_size}/{dataset}.npz")
                keep_flag = tmp['flag']
                embs[patch_size] = tmp['emb'][keep_flag,:]
                patch_loc_before_pad = tmp['pos'][keep_flag,:] - [tmp['pad'][0]//2, tmp['pad'][1]//2]
                #! Switch the column of patch_loc_before_pad to match spot coordinate
                patch_loc_before_pad = patch_loc_before_pad[:,[1,0]]
                patches_loc_before_pad[patch_size] = patch_loc_before_pad
                if load_graph:
                    patch_center_loc_before_pad = patch_loc_before_pad + patch_size//2
                    nearby_index_adj = find_nearby_spots(patch_center_loc_before_pad, patch_center_loc_before_pad, k=9, workers=num_workers)
                    knn_edge = []
                    for i in range(nearby_index_adj.shape[0]):
                        for j in range(nearby_index_adj.shape[1]):
                            if nearby_index_adj[i,j] != i:
                                knn_edge.append([i, nearby_index_adj[i,j]])
                    if len(knn_edge) > 0:
                        knn_edge = np.array(knn_edge, dtype=np.int32)
                        knn_edge = np.unique(knn_edge, axis=0).T
                    else:
                        knn_edge = np.empty((2, 0), dtype=np.int32)
                    edge_index_list[patch_size] = knn_edge
                if patch_size == max(patch_sizes):
                    #TODO: Check
                    windows_mask = np.repeat(True,embs[patch_size].shape[0])
                else:
                    window_size = patch_size*4
                    stride = patch_size*3
                    windows_mask = []
                    x_start = spot_locs[:,0].min()
                    y_start = spot_locs[:,1].min()
                    x_curr = x_start
                    y_curr = y_start
                    if plot_mask:
                        image_plot = plot_image_with_spot(image, spot_locs, spot_radius_pixel)
                        # Create a mask for the patches
                        patch_mask = np.zeros_like(image_plot, dtype=np.uint8)
                        plot_color = (0, 0, 255)
                        patch_index = 0
                    #
                    while y_curr < spot_locs[:,1].max():
                        while x_curr < spot_locs[:,0].max():
                            mask_tmp = np.logical_and(np.logical_and(spot_locs[:,0] >= x_curr, spot_locs[:,0] < x_curr+window_size), np.logical_and(spot_locs[:,1] >= y_curr, spot_locs[:,1] < y_curr+window_size))
                            if mask_tmp.sum() > 1:
                                windows_mask.append(mask_tmp)
                                if plot_mask:
                                    shift = 0
                                    if plot_color == (255, 0, 0):
                                        shift = 10
                                    patch_mask = cv2.rectangle(patch_mask, (int(x_curr)+shift, int(y_curr)+shift), (int(x_curr+window_size)+shift, int(y_curr+window_size)+shift), plot_color, 20)
                                    patch_mask = cv2.putText(patch_mask, str(patch_index), (int(x_curr),int(y_curr+window_size)), cv2.FONT_HERSHEY_SIMPLEX, 7, (255,255,255), 5, cv2.LINE_AA)
                                    patch_index += 1
                            x_curr += stride
                        x_curr = x_start
                        y_curr += stride
                        if plot_mask:
                            if plot_color == (0, 0, 255):
                                plot_color = (255, 0, 0)
                            else:
                                plot_color = (0, 0, 255)
                    #
                    windows_mask = np.array(windows_mask)
                    window_masks[patch_size] = windows_mask
                    if plot_mask:
                        image_plot = cv2.addWeighted(image_plot, 0.7, patch_mask, 0.3, 0)
                        image_plot = cv2.resize(image_plot,(0,0),fx=0.1,fy=0.1)
                        cv2.imwrite(f"{main_dir}/{train_dir}/image_check/{dataset}_{patch_size}patch.png", image_plot)
                # Spot weight mask
                patch_center_loc_before_pad = patch_loc_before_pad + patch_size//2
                spot_weight_mask = np.zeros((spot_locs.shape[0],patch_center_loc_before_pad.shape[0]))
                # ## For simplicity, we first shift the origin to make sure that all values in patch_center_loc_before_pad can be divided by patch_size
                # row_shift = patch_size - patch_center_loc_before_pad[:,0].min()
                # col_shift = patch_size - patch_center_loc_before_pad[:,1].min()
                # patch_center_use = patch_center_loc_before_pad + [row_shift, col_shift]
                # spot_locs_use = spot_locs + [row_shift, col_shift]
                # for i in range(spot_locs.shape[0]):
                #     spot_row_index = spot_locs_use[i,0]//patch_size
                #     spot_col_index = spot_locs_use[i,1]//patch_size
                #     cross_center_loc = [spot_row_index * patch_size + patch_size//2,spot_col_index * patch_size + patch_size//2]
                #     spot_top_left = spot_locs_use[i,:] - [spot_radius_pixel,spot_radius_pixel]
                #     spot_weight_mask[i,:] = np.linalg.norm(patch_center_loc_before_pad - spot_locs[i],axis=1)
                nearby_index = find_nearby_spots(patch_center_loc_before_pad,spot_locs,k=4, workers=num_workers)
                for i in range(spot_locs.shape[0]):
                    for j in range(4):
                        spot_weight_mask[i,nearby_index[i,j]] = square_intersection_area(patch_center_loc_before_pad[nearby_index[i,j],:],patch_size,spot_locs[i,:],spot_radius_pixel*2) / ((spot_radius_pixel*2)**2)
                ## row normalize since some spots at the margin only intersect with 1 or 2 patches
                norm_tmp = spot_weight_mask.sum(axis=1)[:,None]
                norm_tmp[norm_tmp==0] = 1
                spot_weight_mask = spot_weight_mask / norm_tmp
                spot_weight[patch_size] = spot_weight_mask
                # if plot_mask:
                #     import matplotlib as mpl
                #     image_plot = plot_image_with_spot(image, spot_locs, spot_radius_pixel)
                #     # Create a mask for the patches
                #     patch_mask = np.zeros_like(image_plot, dtype=np.uint8)
                #     color_bar = mpl.colormaps['Reds']
                #     for i in range(patch_center_loc_before_pad.shape[0]):
                #         x, y = patch_center_loc_before_pad[i,:]
                #         x_start = max(x-patch_size//2,0)
                #         y_start = max(y-patch_size//2,0)
                #         x_end = min(x+patch_size//2,image.shape[1])
                #         y_end = min(y+patch_size//2,image.shape[0])
                #         color_use = color_bar(min(spot_weight_mask[:,i].max(),0.99))[0:3] # 1 in this color bar is (255,255,255)
                #         color_use = np.array(color_use) * 255
                #         color_use = color_use.astype(np.uint8)
                #         color_use = color_use[[2,1,0]]
                #         patch_mask[y_start:y_end,x_start:x_end,:] = color_use
                #     image_plot = cv2.addWeighted(image_plot, 0.5, patch_mask, 0.5, 0)
                #     image_plot = cv2.resize(image_plot,(0,0),fx=0.1,fy=0.1)
                #     cv2.imwrite(f"{main_dir}/{train_dir}/image_check/{dataset}_spotweight_{patch_size}patch.png", image_plot)
            #
            # Get father-children relationship
            patch_size_min = min(patch_sizes)
            for patch_size in patch_sizes:
                if patch_size != min(patch_sizes):
                    father_patches[patch_size] = find_nearby_spots(patches_loc_before_pad[patch_size]+ patch_size//2,
                                                                patches_loc_before_pad[patch_size_min]+ patch_size_min//2,
                                                                k=1,workers=num_workers)
                    # if plot_mask:
                    #     image_plot = plot_image_with_spot(image, spot_locs, spot_radius_pixel)
                    #     patch_mask = np.zeros_like(image_plot, dtype=np.uint8)
                    #     for i in range(patches_loc_before_pad[patch_size_min].shape[0]):
                    #         x, y = patches_loc_before_pad[patch_size_min][i,:]
                    #         x_start = max(x,0)
                    #         y_start = max(y,0)
                    #         x_end = min(x+patch_size_min,image.shape[1])
                    #         y_end = min(y+patch_size_min,image.shape[0])
                    #         if father_patches[patch_size][i] % 2 == 0:
                    #             color_use = (0,0,255)
                    #         else:
                    #             color_use = (255,0,0)
                    #         patch_mask[y_start:y_end,x_start:x_end,:] = color_use
                    #         patch_mask = cv2.putText(patch_mask, str(father_patches[patch_size][i]), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 7, (255,255,255), 5, cv2.LINE_AA)
                    #     image_plot = cv2.addWeighted(image_plot, 0.5, patch_mask, 0.5, 0)
                    #     image_plot = cv2.resize(image_plot,(0,0),fx=0.1,fy=0.1)
                    #     cv2.imwrite(f"{main_dir}/{train_dir}/image_check/{dataset}_father_{patch_size}patch.png", image_plot)
            # For cells with no prop estimation, we set the prop to 0
            if load_cell_type:
                major_prop = major_prop.loc[:,cell_type_list]
                major_prop_filtered = np.zeros((len(barcode),len(cell_type_list)))
                for i in range(len(barcode)):
                    if barcode[i] in major_prop.index:
                        major_prop_filtered[i,:] = major_prop.loc[barcode[i],:].values
            else:
                major_prop_filtered = np.zeros((len(barcode),len(cell_type_list)))
            # Load gene expression
            cnts_file = f"{cnt_folder}/{dataset}-cnts.tsv"
            cnts = utils.load_tsv(cnts_file)
            cnts = cnts.loc[barcode,:]
            # Initialize an array with zeros for all genes in gene_list
            cnts_filtered = np.zeros((cnts.shape[0], len(gene_list)))
            # Fill in the gene expressions for the common genes
            for i, gene in enumerate(gene_list):
                if gene in cnts.columns:
                    if not rand_count:
                        cnts_filtered[:, i] = cnts[gene].values
                    else:
                        cnts_filtered[:, i] = np.random.uniform(0, 1, size=cnts.shape[0])
            # Create an array indicating whether each gene in gene_list is present in cnts
            gene_mask = np.isin(gene_list, cnts.columns)
            # Load bulk gene expression
            bulk_file = f"{bulk_folder}/{dataset}-bulk.tsv"
            bulk = utils.load_tsv(bulk_file)
            # Initialize an array with zeros for all genes in bulk_hvg
            bulk_filtered = np.zeros(len(bulk_hvg))
            # Fill in the gene expressions for the common genes
            for i, gene in enumerate(bulk_hvg):
                if gene in bulk.index:
                    bulk_filtered[i] = bulk.loc[gene].values[0]

            bulk_rank = bulk_filtered/bulk_norm_factor
            bulk_rank = pd.Series(bulk_rank).rank(method = 'first', ascending=True, pct=True).values
            if rand_count:
                np.random.shuffle(bulk_rank)
            max_file = f"{bulk_folder}/{dataset}-max.tsv"
            max_exp = utils.load_tsv(max_file)
            # Initialize an array with zeros for all genes in bulk_hvg
            max_filtered = np.zeros(len(bulk_hvg))
            # Fill in the gene expressions for the common genes
            for i, gene in enumerate(bulk_hvg):
                if gene in bulk.index:
                    max_filtered[i] = max_exp.loc[gene].values[0]
            if rand_count:
                np.random.shuffle(max_filtered)
            #
            if not load_graph:
                data_tmp = (embs, window_masks,spot_weight, cnts_filtered, gene_mask, bulk_filtered, bulk_rank, max_filtered, father_patches, major_prop_filtered)
                data.append(data_tmp)
                # Save the data to a file
                utils.save_pickle(data_tmp,f"{file_dir}/{dataset.replace('.h5','')}.pkl")
            else:
                data_tmp = (embs, window_masks,spot_weight, cnts_filtered, gene_mask, bulk_filtered, bulk_rank, max_filtered, father_patches, major_prop_filtered, edge_index_list)
                data.append(data_tmp)
                # Save the data to a file
                utils.save_pickle(data_tmp,f"{file_dir}/{dataset.replace('.h5','')}.pkl")
    else:
        data = []
        for dataset in datasets:
            dataset = dataset.replace(".h5", "")
            data_tmp = utils.load_pickle(f"{file_dir}/{dataset}.pkl")
            data.append(data_tmp)
    return data

def load_data_pred(datasets, bulk_samples, bulk, main_dir, bulk_hvg, bulk_norm_factor, interim_dir,
              patch_sizes=[3584, 896, 224], from_file = False, file_dir = None, num_workers=4):
    if not from_file:
        data = []
        emb_folder = f"{main_dir}/{interim_dir}/UNI_multiscale_patches"
        # bulk_folder = f"{main_dir}/TCGA_LUAD/bulk_RNA"
        # bulk_metadata_file = f"{main_dir}/TCGA_LUAD/bulk_RNA/gdc_sample_sheet.2025-08-04.tsv"
        # bulk_metadata = pd.read_csv(bulk_metadata_file, sep='\t', index_col=6)
        # # remove duplicated index
        # bulk_metadata = bulk_metadata.loc[~bulk_metadata.index.duplicated(keep='first')]
        # # Load bulk gene expression
        # bulk_file = f"{bulk_folder}/TCGA_LUAD_expression.csv"
        # bulk = pd.read_csv(bulk_file, index_col=0,header=0)
        # image_metadata_file = f"{main_dir}/TCGA_LUAD/wsis_sample_sheet_filtered.csv"
        # image_metadata = pd.read_csv(image_metadata_file,header=0)
        for dataset, bulk_sample in zip(datasets,bulk_samples):
            # Load embeddings
            embs = {}
            patches_loc_before_pad = {}
            father_patches = {}
            print(f"Loading data for {dataset}...")
            for patch_size in patch_sizes:
                tmp = np.load(f"{emb_folder}/emb_{patch_size}/{dataset}.npz")
                keep_flag = tmp['flag']
                embs[patch_size] = tmp['emb'][keep_flag,:]
                patch_loc_before_pad = tmp['pos'][keep_flag,:] - [tmp['pad'][0]//2, tmp['pad'][1]//2]
                #! Switch the column of patch_loc_before_pad to match spot coordinate
                patch_loc_before_pad = patch_loc_before_pad[:,[1,0]]
                patches_loc_before_pad[patch_size] = patch_loc_before_pad
            # Get father-children relationship
            patch_size_min = min(patch_sizes)
            for patch_size in patch_sizes:
                if patch_size != min(patch_sizes):
                    father_patches[patch_size] = find_nearby_spots(patches_loc_before_pad[patch_size]+ patch_size//2,
                                                                patches_loc_before_pad[patch_size_min]+ patch_size_min//2,
                                                                k=1,workers=num_workers)
            # Initialize an array with zeros for all genes in bulk_hvg
            bulk_filtered = np.zeros(len(bulk_hvg))
            # sample_id = image_metadata.loc[image_metadata['File Name'].str.contains(dataset),"Sample ID"].values[0]
            # bulk_sample_name = bulk_metadata.loc[sample_id,'File ID']
            # Fill in the gene expressions for the common genes
            for i, gene in enumerate(bulk_hvg):
                if gene in bulk.index:
                    bulk_filtered[i] = bulk.loc[gene,bulk_sample]
            #
            bulk_rank = bulk_filtered/bulk_norm_factor
            bulk_rank = pd.Series(bulk_rank).rank(method = 'first', ascending=True, pct=True).values
            data_tmp = (embs, 
                        bulk_filtered, bulk_rank, 
                        father_patches, patches_loc_before_pad
                        )
            data.append(data_tmp)
            # Save the data to a file
            utils.save_pickle(data_tmp,f'{file_dir}/{dataset.replace(".h5","")}.pkl')
    else:
        data = []
        for dataset in datasets:
            dataset = dataset.replace(".h5", "")
            data_tmp = utils.load_pickle(f'{file_dir}/{dataset}.pkl')
            data.append(data_tmp)
    return data

class SuperpatchCollection:
    """Lazily loads per-superpatch .pkl files from disk on access.

    ``load_data2`` returns one of these instead of a fully-loaded list, so RAM
    scales with the number of superpatches held at once (an LRU cache of
    ``cache_size`` entries) rather than with the whole training set.
    """

    def __init__(self, superpatch_meta, file_dir, cache_size=2):
        self.meta = superpatch_meta.reset_index(drop=True)
        self.file_dir = file_dir
        self.names = list(self.meta['name'])
        self._cache_size = max(int(cache_size), 0)
        self._cache = OrderedDict()   # recency-ordered; popitem(last=False) evicts oldest

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        if name in self._cache:
            value = self._cache.pop(name)   # refresh recency
            self._cache[name] = value
            return value
        value = utils.load_pickle(f"{self.file_dir}/{name}.pkl", verbose=False)
        if self._cache_size > 0:
            self._cache[name] = value
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return value

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class PrefetchLoader:
    """Background-thread prefetch wrapper around an iterable (e.g. a DataLoader).

    Pulls up to ``prefetch`` items ahead in a daemon thread so that slow
    per-superpatch pkl loads / tensor conversion (done inside
    ``dataset.__getitem__``) overlap with the GPU compute in the main thread,
    without the IPC overhead of DataLoader worker processes.

    Keeps the wrapped loader's ``__len__`` so ``tqdm`` still shows a percentage
    bar. Use ``prefetch=0`` to disable prefetching entirely (plain iteration).
    """

    def __init__(self, loader, prefetch=4):
        self.loader = loader
        self.prefetch = max(int(prefetch), 0)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.prefetch <= 0:
            yield from self.loader
            return

        buffer = queue.Queue(maxsize=self.prefetch)
        sentinel = object()
        source = iter(self.loader)

        def _worker():
            try:
                for item in source:
                    buffer.put(item)
            except BaseException as exc:   # surfaced in the main thread
                try:
                    buffer.put(exc)
                except Exception:
                    pass
                return
            buffer.put(sentinel)

        worker = threading.Thread(target=_worker, name="prefetch-loader", daemon=True)
        worker.start()
        done = False
        try:
            while True:
                item = buffer.get()
                if item is sentinel:
                    done = True
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # On normal completion the worker has already finished; on an early
            # exit (break/exception) leave the daemon thread to be cleaned up.
            if done:
                worker.join()


def project_spot_weight(spot_weight_sparse, output, spot_chunk=128):
    """
    Sparse projection from compact spot-weight form to spot level.

    result[i, g] = sum_j spot_weight_sparse['values'][i, j] *
                   output[spot_weight_sparse['indices'][i, j], g]

    spot_weight_sparse: dict with 'indices' [S, k] (int) and 'values' [S, k] (float)
    output: [P, G] torch tensor (or numpy) -- 'indices' index along dim 0.
    Returns torch tensor [S, G] on output.device (spot rows x gene dims).
    """
    indices = spot_weight_sparse['indices']
    values = spot_weight_sparse['values']
    if not torch.is_tensor(indices):
        indices = torch.tensor(indices, dtype=torch.long)
    if not torch.is_tensor(values):
        values = torch.tensor(values, dtype=torch.float32)
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
    out_device = output.device
    indices = indices.to(out_device)
    values = values.to(out_device)
    S, k = indices.shape
    G = output.shape[1]
    if k == 0:
        return torch.zeros((S, G), dtype=values.dtype, device=out_device)
    results = []
    for start in range(0, S, spot_chunk):
        end = min(start + spot_chunk, S)
        idx = indices[start:end]                # [c, k]
        val = values[start:end]                 # [c, k]
        gathered = output[idx]                  # [c, k, G]
        results.append((val.unsqueeze(-1) * gathered).sum(dim=1))
    if len(results) == 1:
        return results[0]
    return torch.cat(results, dim=0)


def project_spot_weight_columns(spot_weight_sparse, output_tmp, father_patches, father_unique, spot_chunk=128):
    """
    Sparse projection of a coarse scale onto spots through unique father columns.

    result[i, g] = sum over j where indices[i, j] is one of father_unique of
                   values[i, j] * output_tmp[pos, g],
    where pos is the position of indices[i, j] inside father_unique.

    spot_weight_sparse: {'indices' [S, k], 'values' [S, k]} (numpy or torch)
    output_tmp: [F, G] tensor (numpy or torch) aligned to father_unique rows.
    father_patches: 1-D tensor/array mapping each fine patch to a father column index.
    father_unique: 1-D tensor/array of the unique father column indices (sorted).
    Returns torch tensor [S, G] on output_tmp.device.
    """
    indices = spot_weight_sparse['indices']
    values = spot_weight_sparse['values']
    if isinstance(output_tmp, np.ndarray):
        output_tmp = torch.from_numpy(output_tmp)
    if not torch.is_tensor(indices):
        indices = torch.tensor(indices, dtype=torch.long)
    if not torch.is_tensor(values):
        values = torch.tensor(values, dtype=torch.float32)
    if not torch.is_tensor(father_patches):
        father_patches = torch.tensor(father_patches, dtype=torch.long)
    if not torch.is_tensor(father_unique):
        father_unique = torch.tensor(father_unique, dtype=torch.long)
    # Map each father column index -> its position in father_unique (or -1).
    max_col = -1
    if father_patches.numel():
        max_col = max(max_col, int(father_patches.max().item()))
    if indices.numel():
        max_col = max(max_col, int(indices.max().item()))
    col_to_father = torch.full((max_col + 1,), -1, dtype=torch.long)
    col_to_father[father_unique.to(torch.long)] = torch.arange(father_unique.numel(), dtype=torch.long)
    pos = col_to_father[indices]                 # [S, k]
    valid = pos >= 0
    pos_clipped = pos.clamp(min=0)
    out_device = output_tmp.device
    pos_clipped = pos_clipped.to(out_device)
    valid = valid.to(out_device)
    values = values.to(out_device)
    S, k = indices.shape
    G = output_tmp.shape[1]
    if k == 0:
        return torch.zeros((S, G), dtype=values.dtype, device=out_device)
    results = []
    for start in range(0, S, spot_chunk):
        end = min(start + spot_chunk, S)
        pc = pos_clipped[start:end]              # [c, k]
        va = values[start:end] * valid[start:end].to(values.dtype)  # zero invalid cols
        gathered = output_tmp[pc]                # [c, k, G]
        results.append((va.unsqueeze(-1) * gathered).sum(dim=1))
    if len(results) == 1:
        return results[0]
    return torch.cat(results, dim=0)


def load_data2(datasets, main_dir, processed_data_dir,
              interim_dir, train_dir,bulk_hvg, bulk_norm_factor,
              load_cell_type = True,
              patch_sizes=[56, 14], plot_mask = True, 
              from_file = False, file_dir = None, num_workers=4,
              pixel_size=0.5, superpatch_size = 5000, superpatch_stride = 4000, superpatch_minsize = 1000, superpatch_minlowres_patch = 10,
              global_size=16, local_size=4, cache_embeddings=True, cache_size=4):
    if not from_file:
        os.makedirs(f"{main_dir}/{train_dir}/image_check/",exist_ok=True)
        # superpatch_meta = pd.DataFrame(columns=['name','dataset','superpatch_id','x_start','y_start','x_end','y_end'])
        superpatch_meta = {'name':[], 'dataset':[], 'superpatch_id':[], 'x_start':[], 'y_start':[], 'x_end':[], 'y_end':[]}
        # We load the intersection between gene_list and gene detected in the dataset. So, each dataset's loss have different number of genes involved.
        # Resume support: remember any previously written metadata so partially finished
        # runs can be continued instead of recomputed.
        superpatch_meta_csv = f"{file_dir}/superpatch_meta.csv"
        superpatch_meta_prev = None
        if os.path.exists(superpatch_meta_csv):
            superpatch_meta_prev = pd.read_csv(superpatch_meta_csv)
        # Note: superpatches are written to per-superpatch .pkl files and loaded
        # lazily afterwards, so the whole training set is never kept in RAM at once.
        # Persist the metadata index after every completed dataset so it always reflects
        # finished work and survives crashes. (The per-superpatch .pkl files are the real
        # checkpoint for resuming; this just keeps the CSV index in sync.)
        def write_superpatch_meta():
            pd.DataFrame(superpatch_meta).to_csv(superpatch_meta_csv, index=False)
        emb_folder = f"{main_dir}/{interim_dir}/UNI_multiscale_patches"
        bulk_folder = f"{main_dir}/{interim_dir}/bulk"
        superpatch_info = {'size':superpatch_size, 'stride':superpatch_stride, 'minsize':superpatch_minsize}
        with open(f"{file_dir}/superpatch_info.json", 'w') as f:
            json.dump(superpatch_info, f, indent=4)
        # cnt_folder = f"{main_dir}/{interim_dir}/cnts"
        # bulk_folder = f"{main_dir}/{interim_dir}/bulk"
        # prop_folder = f"{main_dir}/{interim_dir}/deconvolution"
        for dataset in datasets:
            dataset = dataset.replace(".h5", "")
            # if plot_mask:
            image, spot_radius_pixel = load_image(dataset, main_dir = main_dir, processed_data_dir = processed_data_dir, 
                                                    interim_dir = interim_dir, target_radius=pixel_size)
            superpatch_num = [int(image.shape[1]//superpatch_stride) + 1, int(image.shape[0]//superpatch_stride) + 1]
            superpatch_total = superpatch_num[0] * superpatch_num[1]
            #
            # Resume support: figure out which superpatches still need to be processed.
            # A superpatch is considered finished when its output .pkl already exists on
            # disk (too-small superpatches are simply not processed at all).
            superpatch_bounds = {}
            pending_superpatch_ids = []
            for superpatch_id in range(superpatch_total):
                superpatch_x_start = (superpatch_id % superpatch_num[0]) * superpatch_stride
                superpatch_y_start = (superpatch_id // superpatch_num[0]) * superpatch_stride
                superpatch_x_end = min(superpatch_x_start + superpatch_size, image.shape[1])
                superpatch_y_end = min(superpatch_y_start + superpatch_size, image.shape[0])
                if superpatch_x_end - superpatch_x_start < superpatch_minsize or superpatch_y_end - superpatch_y_start < superpatch_minsize:
                    print(f"    Superpatch {superpatch_id}: x [{superpatch_x_start}, {superpatch_x_end}), y [{superpatch_y_start}, {superpatch_y_end}) is too small, skipping...")
                    continue
                superpatch_bounds[superpatch_id] = (superpatch_x_start, superpatch_y_start, superpatch_x_end, superpatch_y_end)
                if not os.path.exists(f"{file_dir}/{dataset}_superpatch_{superpatch_id}.pkl"):
                    pending_superpatch_ids.append(superpatch_id)
            # If every valid superpatch for this dataset is already finished, load them
            # all from disk and skip the heavy per-scale embedding / lowres loading.
            if not pending_superpatch_ids:
                print(f"    All {len(superpatch_bounds)} valid superpatches for {dataset} already processed, loading from disk...")
                for superpatch_id, (sx, sy, ex, ey) in superpatch_bounds.items():
                    name = f"{dataset}_superpatch_{superpatch_id}"
                    superpatch_meta['name'].append(name)
                    superpatch_meta['dataset'].append(dataset)
                    superpatch_meta['superpatch_id'].append(superpatch_id)
                    superpatch_meta['x_start'].append(sx)
                    superpatch_meta['y_start'].append(sy)
                    superpatch_meta['x_end'].append(ex)
                    superpatch_meta['y_end'].append(ey)
                write_superpatch_meta()
                continue
            #
            emb_tmp = np.load(f"{emb_folder}/emb_224/{dataset}.npz")
            lowres_result = utils.load_pickle(f"{main_dir}/{train_dir}/train_pred_lowres/{dataset}-pred.pickle")
            # lowres_result = utils.load_pickle(f"{main_dir}/{train_dir}/train_pred_lowres/{dataset}.h5-pred.pickle")  # !!! change!
            # Load the per-scale embedding files once per dataset and reuse them across
            # all superpatches. Previously every superpatch re-loaded the full emb_*
            # .npz from disk (emb_14 can be many GB for a whole WSI), causing repeated
            # huge I/O and transient RAM spikes. Set cache_embeddings=False to fall back
            # to the per-superpatch loading behaviour.
            if cache_embeddings:
                emb_arrays = {}
                emb_pos = {}
                emb_flag = {}
                emb_pad = {}
                for patch_size in patch_sizes:
                    tmp = np.load(f"{emb_folder}/emb_{patch_size}/{dataset}.npz")
                    emb_arrays[patch_size] = tmp['emb']
                    emb_pos[patch_size] = tmp['pos']
                    emb_flag[patch_size] = tmp['flag']
                    emb_pad[patch_size] = tmp['pad']
            else:
                emb_arrays = emb_pos = emb_flag = emb_pad = None
            #
            for superpatch_id in tqdm(range(superpatch_total), desc=f"Processing superpatches for {dataset}"):
                if superpatch_id not in superpatch_bounds:
                    continue
                superpatch_x_start, superpatch_y_start, superpatch_x_end, superpatch_y_end = superpatch_bounds[superpatch_id]
                name = f"{dataset}_superpatch_{superpatch_id}"
                # Resume support: skip superpatches that have already been processed.
                if os.path.exists(f"{file_dir}/{name}.pkl"):
                    print(f"    Superpatch {superpatch_id} already processed, loading from disk...")
                    superpatch_meta['name'].append(name)
                    superpatch_meta['dataset'].append(dataset)
                    superpatch_meta['superpatch_id'].append(superpatch_id)
                    superpatch_meta['x_start'].append(superpatch_x_start)
                    superpatch_meta['y_start'].append(superpatch_y_start)
                    superpatch_meta['x_end'].append(superpatch_x_end)
                    superpatch_meta['y_end'].append(superpatch_y_end)
                    continue
                print(f"    Superpatch {superpatch_id}: x [{superpatch_x_start}, {superpatch_x_end}), y [{superpatch_y_start}, {superpatch_y_end})")
                #
                superpatch_meta['name'].append(name)
                superpatch_meta['dataset'].append(dataset)
                superpatch_meta['superpatch_id'].append(superpatch_id)
                superpatch_meta['x_start'].append(superpatch_x_start)
                superpatch_meta['y_start'].append(superpatch_y_start)
                superpatch_meta['x_end'].append(superpatch_x_end)
                superpatch_meta['y_end'].append(superpatch_y_end)
                #
                keep_flag = emb_tmp['flag']
                lowres_loc = emb_tmp['pos'][keep_flag,:] - [emb_tmp['pad'][0]//2, emb_tmp['pad'][1]//2]
                lowres_loc = lowres_loc[:,[1,0]]
                lowres_loc_superpatch_flag = np.logical_and(np.logical_and(lowres_loc[:,0] >= superpatch_x_start, lowres_loc[:,0] < superpatch_x_end), np.logical_and(lowres_loc[:,1] >= superpatch_y_start, lowres_loc[:,1] < superpatch_y_end))
                lowres_loc = lowres_loc[lowres_loc_superpatch_flag,:]
                ### change !!!
                if lowres_loc.shape[0] < superpatch_minlowres_patch:
                    print(f"    Superpatch {superpatch_id}: less than {superpatch_minlowres_patch} lowres patches, skipping...")
                    continue
                # The low-resolution model returns (output, None) when pred_celltype
                # is false, so there are no proportions to read here; the consumer
                # below already substitutes -1 when load_cell_type is off.
                # lowres_cellprop = lowres_result['major_prop_pred'][:,2,:]
                # lowres_cellprop = lowres_cellprop[lowres_loc_superpatch_flag,:]
                if load_cell_type:
                    lowres_cellprop = lowres_result['major_prop_pred'][:,2,:]
                    lowres_cellprop = lowres_cellprop[lowres_loc_superpatch_flag,:]
                else:
                    lowres_cellprop = None
                lowres_geneexp = lowres_result['output'][:,2,:]
                lowres_geneexp = lowres_geneexp[lowres_loc_superpatch_flag,:]
                # latent_emb = lowres_result['latent_emb'][2]
                # latent_emb = latent_emb[lowres_loc_superpatch_flag,:]
                # Load embeddings
                embs = {}
                window_masks = {}
                spot_weight = {}
                patches_loc_before_pad = {}
                father_patches = {}
                for patch_size in patch_sizes:
                    if cache_embeddings:
                        keep_flag = emb_flag[patch_size]
                        pos_full = emb_pos[patch_size]
                        pad_full = emb_pad[patch_size]
                        emb_full = emb_arrays[patch_size]
                    else:
                        tmp = np.load(f"{emb_folder}/emb_{patch_size}/{dataset}.npz")
                        keep_flag = tmp['flag']
                        pos_full = tmp['pos']
                        pad_full = tmp['pad']
                        emb_full = tmp['emb']
                    keep_idx = np.flatnonzero(keep_flag)
                    patch_loc_before_pad = pos_full[keep_idx,:] - [pad_full[0]//2, pad_full[1]//2]
                    #! Switch the column of patch_loc_before_pad to match spot coordinate
                    patch_loc_before_pad = patch_loc_before_pad[:,[1,0]]
                    patch_loc_before_pad_superpatch_flag = np.logical_and(np.logical_and(patch_loc_before_pad[:,0] >= superpatch_x_start-patch_size, patch_loc_before_pad[:,0] < superpatch_x_end), np.logical_and(patch_loc_before_pad[:,1] >= superpatch_y_start-patch_size, patch_loc_before_pad[:,1] < superpatch_y_end))
                    patch_loc_before_pad = patch_loc_before_pad[patch_loc_before_pad_superpatch_flag,:]
                    patches_loc_before_pad[patch_size] = patch_loc_before_pad
                    # Single gather instead of two chained boolean-index copies.
                    embs[patch_size] = emb_full[keep_idx[patch_loc_before_pad_superpatch_flag],:]
                    #
                    # if patch_size == max(patch_sizes):
                    #     #TODO: Check
                    #     windows_mask = np.repeat(True,embs[patch_size].shape[0])
                    # else:
                    window_size = 224*4
                    stride = 224*3
                    windows_mask = []
                    x_start = lowres_loc[:,0].min()
                    y_start = lowres_loc[:,1].min()
                    x_curr = x_start
                    y_curr = y_start
                    if plot_mask:
                        image_plot = plot_image_with_patch(image, lowres_loc, 224)
                        # Create a mask for the patches
                        patch_mask = np.zeros_like(image_plot, dtype=np.uint8)
                        plot_color = (0, 0, 255)
                        patch_index = 0
                    #
                    while y_curr < lowres_loc[:,1].max():
                        while x_curr < lowres_loc[:,0].max():
                            mask_tmp = np.logical_and(np.logical_and(lowres_loc[:,0] >= x_curr, lowres_loc[:,0] < x_curr+window_size), np.logical_and(lowres_loc[:,1] >= y_curr, lowres_loc[:,1] < y_curr+window_size))
                            if mask_tmp.sum() > 1:
                                windows_mask.append(mask_tmp)
                                if plot_mask:
                                    shift = 0
                                    if plot_color == (255, 0, 0):
                                        shift = 10
                                    patch_mask = cv2.rectangle(patch_mask, (int(x_curr)+shift, int(y_curr)+shift), (int(x_curr+window_size)+shift, int(y_curr+window_size)+shift), plot_color, 20)
                                    patch_mask = cv2.putText(patch_mask, str(patch_index), (int(x_curr),int(y_curr+window_size)), cv2.FONT_HERSHEY_SIMPLEX, 7, (255,255,255), 5, cv2.LINE_AA)
                                    patch_index += 1
                            x_curr += stride
                        x_curr = x_start
                        y_curr += stride
                        if plot_mask:
                            if plot_color == (0, 0, 255):
                                plot_color = (255, 0, 0)
                            else:
                                plot_color = (0, 0, 255)
                    #
                    windows_mask = np.array(windows_mask)
                    window_masks[patch_size] = windows_mask
                    if plot_mask:
                        image_plot = cv2.addWeighted(image_plot, 0.7, patch_mask, 0.3, 0)
                        image_plot = cv2.resize(image_plot,(0,0),fx=0.1,fy=0.1)
                        cv2.imwrite(f"{main_dir}/{train_dir}/image_check/{dataset}_superpatch_{superpatch_id}_{patch_size}patch.png", image_plot)
                    # Spot weight mask. Stored sparsely: only the k nearest
                    # neighbours can have non-zero overlap, so we keep compact
                    # [S, k] indices+values instead of a dense [S, P] float32
                    # matrix (the dense matrix was a major source of OOM).
                    patch_center_loc_before_pad = patch_loc_before_pad + patch_size//2
                    lowres_center_loc = lowres_loc + 224//2
                    k_tmp = (224//patch_size + 1)**2
                    nearby_index = find_nearby_spots(patch_center_loc_before_pad,lowres_center_loc,k=k_tmp, workers=num_workers)
                    # Vectorized overlap-area computation (replaces the nested Python loop).
                    patch_center_use = patch_center_loc_before_pad[nearby_index]  # [S, k_tmp, 2]
                    half_patch = patch_size / 2.0
                    half_spot = 224 / 2.0
                    x_lo = np.maximum(patch_center_use[...,0] - half_patch, lowres_center_loc[:,None,0] - half_spot)
                    x_hi = np.minimum(patch_center_use[...,0] + half_patch, lowres_center_loc[:,None,0] + half_spot)
                    y_lo = np.maximum(patch_center_use[...,1] - half_patch, lowres_center_loc[:,None,1] - half_spot)
                    y_hi = np.minimum(patch_center_use[...,1] + half_patch, lowres_center_loc[:,None,1] + half_spot)
                    areas = np.maximum(x_hi - x_lo, 0.0) * np.maximum(y_hi - y_lo, 0.0) / (224.0**2)
                    values = areas.astype(np.float32)
                    ## row normalize since some spots at the margin only intersect with 1 or 2 patches
                    row_sum = values.sum(axis=1, keepdims=True)
                    row_sum[row_sum==0] = 1
                    values = values / row_sum
                    spot_weight[patch_size] = {
                        'indices': np.ascontiguousarray(nearby_index, dtype=np.int32),
                        'values': np.ascontiguousarray(values, dtype=np.float32),
                    }
                    if plot_mask:
                        import matplotlib as mpl
                        image_plot = plot_image_with_patch(image, lowres_loc, 224)
                        # Create a mask for the patches
                        patch_mask = np.zeros_like(image_plot, dtype=np.uint8)
                        color_bar = mpl.colormaps['Reds']
                        patch_max = np.zeros(patch_center_loc_before_pad.shape[0], dtype=np.float32)
                        np.maximum.at(patch_max, nearby_index.ravel(), values.ravel())
                        for i in range(patch_center_loc_before_pad.shape[0]):
                            x, y = patch_center_loc_before_pad[i,:]
                            x_start = max(x-patch_size//2,0)
                            y_start = max(y-patch_size//2,0)
                            x_end = min(x+patch_size//2,image.shape[1])
                            y_end = min(y+patch_size//2,image.shape[0])
                            color_use = color_bar(min(patch_max[i],0.99))[0:3] # 1 in this color bar is (255,255,255)
                            color_use = np.array(color_use) * 255
                            color_use = color_use.astype(np.uint8)
                            color_use = color_use[[2,1,0]]
                            patch_mask[y_start:y_end,x_start:x_end,:] = color_use
                        image_plot = cv2.addWeighted(image_plot, 0.3, patch_mask, 0.7, 0)
                        image_plot = cv2.resize(image_plot,(0,0),fx=0.1,fy=0.1)
                        cv2.imwrite(f"{main_dir}/{train_dir}/image_check/{dataset}_superpatch_{superpatch_id}_spotweight_{patch_size}patch.png", image_plot)
                    #
                # Get father-children relationship
                patch_size_min = min(patch_sizes)
                for patch_size in patch_sizes:
                    if patch_size != min(patch_sizes):
                        father_patches[patch_size] = find_nearby_spots(patches_loc_before_pad[patch_size]+ patch_size//2,
                                                                    patches_loc_before_pad[patch_size_min]+ patch_size_min//2,
                                                                    k=1,workers=num_workers)
                        if plot_mask:
                            image_plot = plot_image_with_patch(image, patches_loc_before_pad[patch_size], patch_size, stroke_width=2)
                            patch_mask = np.zeros_like(image_plot, dtype=np.uint8)
                            for i in range(patches_loc_before_pad[patch_size_min].shape[0]):
                                x, y = patches_loc_before_pad[patch_size_min][i,:]
                                x_start = max(x,0)
                                y_start = max(y,0)
                                x_end = min(x+patch_size_min,image.shape[1])
                                y_end = min(y+patch_size_min,image.shape[0])
                                if father_patches[patch_size][i] % 2 == 0:
                                    color_use = (0,0,255)
                                else:
                                    color_use = (255,0,0)
                                patch_mask[y_start:y_end,x_start:x_end,:] = color_use
                                # patch_mask = cv2.putText(patch_mask, str(father_patches[patch_size][i]), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 7, (255,255,255), 5, cv2.LINE_AA)
                            image_plot = cv2.addWeighted(image_plot, 0.7, patch_mask, 0.3, 0)
                            # image_plot = cv2.resize(image_plot,(0,0),fx=0.1,fy=0.1)
                            cv2.imwrite(f"{main_dir}/{train_dir}/image_check/{dataset}_superpatch_{superpatch_id}_father_{patch_size}patch.png", image_plot)
                #a matrix for latent embedding
                father_lowres = find_nearby_spots(lowres_loc + 224//2,
                                    patches_loc_before_pad[patch_size_min]+ patch_size_min//2,
                                    k=1,workers=num_workers)
                # For cells with no prop estimation, we set the prop to 0
                if load_cell_type:
                    major_prop_filtered = np.exp(lowres_cellprop)
                else:
                    major_prop_filtered = -1
                # Load gene expression
                cnts_filtered = lowres_geneexp
                # Create an array indicating whether each gene in gene_list is present in cnts
                gene_mask = np.full(cnts_filtered.shape[1], True) # Since cnts_filtered already only contains genes in gene_list, we can set all to True
                # Load bulk gene expression (for high resolution prediction, this is latent embedding)
                # Load bulk gene expression
                bulk_file = f"{bulk_folder}/{dataset}-bulk.tsv"
                bulk = utils.load_tsv(bulk_file)
                # Initialize an array with zeros for all genes in bulk_hvg
                bulk_filtered = np.zeros(len(bulk_hvg))
                # Fill in the gene expressions for the common genes
                for i, gene in enumerate(bulk_hvg):
                    if gene in bulk.index:
                        bulk_filtered[i] = bulk.loc[gene].values[0]
    
                bulk_rank = bulk_filtered/bulk_norm_factor
                bulk_rank = pd.Series(bulk_rank).rank(method = 'first', ascending=True, pct=True).values
                # max_filtered = None
                #
                data_tmp = (embs, window_masks,spot_weight, cnts_filtered, gene_mask, bulk_filtered, bulk_rank, father_lowres, father_patches, major_prop_filtered)
                # Save the data to a file (the .pkl is the source of truth; do not
                # keep data_tmp in RAM so memory stays bounded per superpatch).
                utils.save_pickle(data_tmp,f"{file_dir}/{name}.pkl")
                del data_tmp
            # Persist progress after each dataset so a crash mid-run can be resumed.
            write_superpatch_meta()
            # Free the full-WSI embedding arrays (and the full image / lowres
            # predictions) before moving on to the next dataset.
            del emb_arrays, emb_pos, emb_flag, emb_pad, emb_tmp, lowres_result, image
            gc.collect()
        # Merge with any previously written metadata, keep one row per superpatch, and
        # return the data in the same order as the metadata so the from_file=True path
        # stays consistent.
        superpatch_meta = pd.DataFrame(superpatch_meta)
        if superpatch_meta_prev is not None:
            superpatch_meta = pd.concat([superpatch_meta_prev, superpatch_meta], ignore_index=True)
        superpatch_meta = superpatch_meta.loc[superpatch_meta['dataset'].isin([d.replace(".h5","") for d in datasets])]
        superpatch_meta = superpatch_meta.drop_duplicates(subset=['name'], keep='first')
        superpatch_meta = superpatch_meta.sort_values(['dataset', 'superpatch_id']).reset_index(drop=True)
        superpatch_meta.to_csv(superpatch_meta_csv, index=False)
        # Only return entries whose .pkl actually exists on disk (superpatches
        # with too few lowres patches are skipped and have no pkl).
        superpatch_meta = superpatch_meta.loc[
            [os.path.exists(f"{file_dir}/{name}.pkl") for name in superpatch_meta['name']]
        ].reset_index(drop=True)
    else:
        superpatch_meta = pd.read_csv(f"{file_dir}/superpatch_meta.csv")
        superpatch_meta = superpatch_meta.loc[superpatch_meta['dataset'].isin([dataset.replace(".h5","") for dataset in datasets]),:].reset_index(drop=True)
        superpatch_meta = superpatch_meta.loc[
            [os.path.exists(f"{file_dir}/{name}.pkl") for name in superpatch_meta['name']]
        ].reset_index(drop=True)
    # Return a lazy collection: each superpatch is loaded from disk on access
    # (with a small LRU cache of ``cache_size`` entries), so RAM scales with the
    # active superpatch/batch instead of the whole training set.
    return SuperpatchCollection(superpatch_meta, file_dir, cache_size=cache_size)

def load_data2_pred(datasets, main_dir, interim_dir,pred_dir,
              patch_sizes=[56, 14], 
              from_file = False, file_dir = None, num_workers=4,
              superpatch_size = 5000, superpatch_stride = 4000, superpatch_minsize = 1000, superpatch_minlowres_patch = 10,
              global_size=20, local_size=4, cache_embeddings=True, cache_size=2):
    if not from_file:
        # superpatch_meta = pd.DataFrame(columns=['name','dataset','superpatch_id','x_start','y_start','x_end','y_end'])
        superpatch_meta = {'name':[], 'dataset':[], 'superpatch_id':[], 'x_start':[], 'y_start':[], 'x_end':[], 'y_end':[]}
        # We load the intersection between gene_list and gene detected in the dataset. So, each dataset's loss have different number of genes involved.
        # Resume support: remember any previously written metadata so partially finished
        # runs can be continued instead of recomputed.
        superpatch_meta_csv = f"{file_dir}/superpatch_meta.csv"
        superpatch_meta_prev = None
        if os.path.exists(superpatch_meta_csv):
            superpatch_meta_prev = pd.read_csv(superpatch_meta_csv)
        # Note: superpatches are written to per-superpatch .pkl files and loaded
        # lazily afterwards, so the whole set is never kept in RAM at once.
        # Persist the metadata index after every completed dataset so it always reflects
        # finished work and survives crashes. (The per-superpatch .pkl files are the real
        # checkpoint for resuming; this just keeps the CSV index in sync.)
        def write_superpatch_meta():
            pd.DataFrame(superpatch_meta).to_csv(superpatch_meta_csv, index=False)
        emb_folder = f"{main_dir}/{interim_dir}/UNI_multiscale_patches"
        superpatch_info = {'size':superpatch_size, 'stride':superpatch_stride, 'minsize':superpatch_minsize}
        with open(f"{file_dir}/superpatch_info.json", 'w') as f:
            json.dump(superpatch_info, f, indent=4)
        # cnt_folder = f"{main_dir}/{interim_dir}/cnts"
        # bulk_folder = f"{main_dir}/{interim_dir}/bulk"
        # prop_folder = f"{main_dir}/{interim_dir}/deconvolution"
        for dataset in datasets:
            dataset = dataset.replace(".h5", "")
            # if plot_mask:
            image = tifffile.imread(f"{main_dir}/{interim_dir}/UNI_multiscale_patches/wsis_rescaled/{dataset}_rescaled.tif")
            superpatch_num = [int(image.shape[1]//superpatch_stride) + 1, int(image.shape[0]//superpatch_stride) + 1]
            superpatch_total = superpatch_num[0] * superpatch_num[1]
            #
            # Resume support: figure out which superpatches still need to be processed.
            # A superpatch is considered finished when its output .pkl already exists on
            # disk (too-small superpatches are simply not processed at all).
            superpatch_bounds = {}
            pending_superpatch_ids = []
            for superpatch_id in range(superpatch_total):
                superpatch_x_start = (superpatch_id % superpatch_num[0]) * superpatch_stride
                superpatch_y_start = (superpatch_id // superpatch_num[0]) * superpatch_stride
                superpatch_x_end = min(superpatch_x_start + superpatch_size, image.shape[1])
                superpatch_y_end = min(superpatch_y_start + superpatch_size, image.shape[0])
                if superpatch_x_end - superpatch_x_start < superpatch_minsize or superpatch_y_end - superpatch_y_start < superpatch_minsize:
                    print(f"    Superpatch {superpatch_id}: x [{superpatch_x_start}, {superpatch_x_end}), y [{superpatch_y_start}, {superpatch_y_end}) is too small, skipping...")
                    continue
                superpatch_bounds[superpatch_id] = (superpatch_x_start, superpatch_y_start, superpatch_x_end, superpatch_y_end)
                if not os.path.exists(f"{file_dir}/{dataset}_superpatch_{superpatch_id}.pkl"):
                    pending_superpatch_ids.append(superpatch_id)
            # If every valid superpatch for this dataset is already finished, load them
            # all from disk and skip the heavy per-scale embedding loading.
            if not pending_superpatch_ids:
                print(f"    All {len(superpatch_bounds)} valid superpatches for {dataset} already processed, loading from disk...")
                for superpatch_id, (sx, sy, ex, ey) in superpatch_bounds.items():
                    name = f"{dataset}_superpatch_{superpatch_id}"
                    superpatch_meta['name'].append(name)
                    superpatch_meta['dataset'].append(dataset)
                    superpatch_meta['superpatch_id'].append(superpatch_id)
                    superpatch_meta['x_start'].append(sx)
                    superpatch_meta['y_start'].append(sy)
                    superpatch_meta['x_end'].append(ex)
                    superpatch_meta['y_end'].append(ey)
                write_superpatch_meta()
                continue
            #
            emb_tmp = np.load(f"{emb_folder}/emb_224/{dataset}.npz")
            # lowres_result = utils.load_pickle(f"{main_dir}/{train_dir}/train_pred_lowres/{dataset}-pred.pickle")
            # lowres_result = utils.load_pickle(f"{main_dir}/{pred_dir}/pred/{dataset}-pred.pickle") 
            # Load the per-scale embedding files once per dataset and reuse them across
            # all superpatches. Previously every superpatch re-loaded the full emb_*
            # .npz from disk (emb_14 can be many GB for a whole WSI), causing repeated
            # huge I/O and transient RAM spikes. Set cache_embeddings=False to fall back
            # to the per-superpatch loading behaviour.
            if cache_embeddings:
                emb_arrays = {}
                emb_pos = {}
                emb_flag = {}
                emb_pad = {}
                for patch_size in patch_sizes:
                    tmp = np.load(f"{emb_folder}/emb_{patch_size}/{dataset}.npz")
                    emb_arrays[patch_size] = tmp['emb']
                    emb_pos[patch_size] = tmp['pos']
                    emb_flag[patch_size] = tmp['flag']
                    emb_pad[patch_size] = tmp['pad']
            else:
                emb_arrays = emb_pos = emb_flag = emb_pad = None
            #
            for superpatch_id in tqdm(range(superpatch_total), desc=f"Processing superpatches for {dataset}"):
                if superpatch_id not in superpatch_bounds:
                    continue
                superpatch_x_start, superpatch_y_start, superpatch_x_end, superpatch_y_end = superpatch_bounds[superpatch_id]
                name = f"{dataset}_superpatch_{superpatch_id}"
                # Resume support: skip superpatches that have already been processed.
                if os.path.exists(f"{file_dir}/{name}.pkl"):
                    print(f"    Superpatch {superpatch_id} already processed, loading from disk...")
                    superpatch_meta['name'].append(name)
                    superpatch_meta['dataset'].append(dataset)
                    superpatch_meta['superpatch_id'].append(superpatch_id)
                    superpatch_meta['x_start'].append(superpatch_x_start)
                    superpatch_meta['y_start'].append(superpatch_y_start)
                    superpatch_meta['x_end'].append(superpatch_x_end)
                    superpatch_meta['y_end'].append(superpatch_y_end)
                    continue
                print(f"    Superpatch {superpatch_id}: x [{superpatch_x_start}, {superpatch_x_end}), y [{superpatch_y_start}, {superpatch_y_end})")
                #
                superpatch_meta['name'].append(name)
                superpatch_meta['dataset'].append(dataset)
                superpatch_meta['superpatch_id'].append(superpatch_id)
                superpatch_meta['x_start'].append(superpatch_x_start)
                superpatch_meta['y_start'].append(superpatch_y_start)
                superpatch_meta['x_end'].append(superpatch_x_end)
                superpatch_meta['y_end'].append(superpatch_y_end)
                #
                keep_flag = emb_tmp['flag']
                lowres_loc = emb_tmp['pos'][keep_flag,:] - [emb_tmp['pad'][0]//2, emb_tmp['pad'][1]//2]
                lowres_loc = lowres_loc[:,[1,0]]
                # Make sure to include lowres patches that are within 300 pixels of the superpatch boundary
                lowres_loc_superpatch_flag = np.logical_and(np.logical_and(lowres_loc[:,0] >= superpatch_x_start - 300, lowres_loc[:,0] < superpatch_x_end + 300), np.logical_and(lowres_loc[:,1] >= superpatch_y_start - 300, lowres_loc[:,1] < superpatch_y_end + 300))
                lowres_loc = lowres_loc[lowres_loc_superpatch_flag,:]
                ### change !!!
                if lowres_loc.shape[0] < superpatch_minlowres_patch:
                    print(f"    Superpatch {superpatch_id}: less than {superpatch_minlowres_patch} lowres patches, skipping...")
                    continue
                # latent_emb = lowres_result['latent_emb'][2]
                # latent_emb = latent_emb[lowres_loc_superpatch_flag,:]
                # Load embeddings
                embs = {}
                patches_loc_before_pad = {}
                father_patches = {}
                for patch_size in patch_sizes:
                    if cache_embeddings:
                        keep_flag = emb_flag[patch_size]
                        pos_full = emb_pos[patch_size]
                        pad_full = emb_pad[patch_size]
                        emb_full = emb_arrays[patch_size]
                    else:
                        tmp = np.load(f"{emb_folder}/emb_{patch_size}/{dataset}.npz")
                        keep_flag = tmp['flag']
                        pos_full = tmp['pos']
                        pad_full = tmp['pad']
                        emb_full = tmp['emb']
                    keep_idx = np.flatnonzero(keep_flag)
                    patch_loc_before_pad = pos_full[keep_idx,:] - [pad_full[0]//2, pad_full[1]//2]
                    #! Switch the column of patch_loc_before_pad to match spot coordinate
                    patch_loc_before_pad = patch_loc_before_pad[:,[1,0]]
                    patch_loc_before_pad_superpatch_flag = np.logical_and(np.logical_and(patch_loc_before_pad[:,0] >= superpatch_x_start-patch_size, patch_loc_before_pad[:,0] < superpatch_x_end), np.logical_and(patch_loc_before_pad[:,1] >= superpatch_y_start-patch_size, patch_loc_before_pad[:,1] < superpatch_y_end))
                    patch_loc_before_pad = patch_loc_before_pad[patch_loc_before_pad_superpatch_flag,:]
                    patches_loc_before_pad[patch_size] = patch_loc_before_pad
                    # Single gather instead of two chained boolean-index copies.
                    embs[patch_size] = emb_full[keep_idx[patch_loc_before_pad_superpatch_flag],:]
                # Get father-children relationship
                patch_size_min = min(patch_sizes)
                for patch_size in patch_sizes:
                    if patch_size != min(patch_sizes):
                        father_patches[patch_size] = find_nearby_spots(patches_loc_before_pad[patch_size]+ patch_size//2,
                                                                    patches_loc_before_pad[patch_size_min]+ patch_size_min//2,
                                                                    k=1,workers=num_workers)
                #a matrix for latent embedding
                father_lowres = find_nearby_spots(lowres_loc + 224//2,
                                    patches_loc_before_pad[patch_size_min]+ patch_size_min//2,
                                    k=1,workers=num_workers)
                #
                data_tmp = (embs, father_lowres, father_patches, patches_loc_before_pad)
                # Save the data to a file (the .pkl is the source of truth; do not
                # keep data_tmp in RAM so memory stays bounded per superpatch).
                utils.save_pickle(data_tmp,f"{file_dir}/{name}.pkl")
                del data_tmp
            # Persist progress after each dataset so a crash mid-run can be resumed.
            write_superpatch_meta()
            # Free the full-WSI embedding arrays (and the full image) before moving
            # on to the next dataset.
            del emb_arrays, emb_pos, emb_flag, emb_pad, emb_tmp, image
            gc.collect()
        # Merge with any previously written metadata, keep one row per superpatch, and
        # return the data in the same order as the metadata so the from_file=True path
        # stays consistent.
        superpatch_meta = pd.DataFrame(superpatch_meta)
        if superpatch_meta_prev is not None:
            superpatch_meta = pd.concat([superpatch_meta_prev, superpatch_meta], ignore_index=True)
        superpatch_meta = superpatch_meta.loc[superpatch_meta['dataset'].isin([d.replace(".h5","") for d in datasets])]
        superpatch_meta = superpatch_meta.drop_duplicates(subset=['name'], keep='first')
        superpatch_meta = superpatch_meta.sort_values(['dataset', 'superpatch_id']).reset_index(drop=True)
        superpatch_meta.to_csv(superpatch_meta_csv, index=False)
        # Only return entries whose .pkl actually exists on disk (superpatches
        # with too few lowres patches are skipped and have no pkl).
        superpatch_meta = superpatch_meta.loc[
            [os.path.exists(f"{file_dir}/{name}.pkl") for name in superpatch_meta['name']]
        ].reset_index(drop=True)
    else:
        superpatch_meta = pd.read_csv(f"{file_dir}/superpatch_meta.csv")
        superpatch_meta = superpatch_meta.loc[superpatch_meta['dataset'].isin([dataset.replace(".h5","") for dataset in datasets]),:].reset_index(drop=True)
        superpatch_meta = superpatch_meta.loc[
            [os.path.exists(f"{file_dir}/{name}.pkl") for name in superpatch_meta['name']]
        ].reset_index(drop=True)
    # Return a lazy collection: each superpatch is loaded from disk on access
    # (with a small LRU cache of ``cache_size`` entries), so RAM scales with the
    # active superpatch/batch instead of the whole training set.
    return SuperpatchCollection(superpatch_meta, file_dir, cache_size=cache_size)

from scipy.sparse import coo_matrix
def load_data_withadj(datasets, gene_list, main_dir, processed_data_dir,
              interim_dir, train_dir,
              bulk_hvg, bulk_norm_factor, cell_type_list, load_cell_type = True,
              patch_size=224, plot_mask = True, 
              from_file = False, file_dir = None, num_workers=4,
              pixel_size=0.5,
              barcode_dir=None,smooth_cnts=True):
    if not from_file:
        import os
        os.makedirs(f"{main_dir}/{train_dir}/image_check/",exist_ok=True)
        # We load the intersection between gene_list and gene detected in the dataset. So, each dataset's loss have different number of genes involved.
        data = []
        emb_folder = f"{main_dir}/{interim_dir}/UNI_multiscale_patches"
        if smooth_cnts:
            cnt_folder = f"{main_dir}/{interim_dir}/cnts_smooth"
        else:
            cnt_folder = f"{main_dir}/{interim_dir}/cnts"
        bulk_folder = f"{main_dir}/{interim_dir}/bulk"
        prop_folder = f"{main_dir}/{interim_dir}/deconvolution"
        for dataset in datasets:
            dataset = dataset.replace(".h5", "")
            # if plot_mask:
            image, spot_radius_pixel = load_image(dataset, main_dir = main_dir, processed_data_dir = processed_data_dir, 
                                                    interim_dir = interim_dir, target_radius=pixel_size)
            # Load spot locations
            locs_file = f"{emb_folder}/wsis_rescaled/{dataset}_loc_rescaled.npy"
            barcode = utils.load_pickle(f"{emb_folder}/wsis_rescaled/{dataset}_barcode.pickle")
            #! Notice that this the first column of loc is the col index for the image matrix, which is the x axis direction on the plot!
            spot_locs = np.load(locs_file)
            # Load cell type proportions
            if load_cell_type:
                major_prop = pd.read_csv(f"{prop_folder}/{dataset}_prop.csv",index_col=0,header=0)
            # Only keep spots used in cnts
            if barcode_dir is None:
                barcode_use = utils.load_pickle(f"{main_dir}/{interim_dir}/UNI_multiscale_patches/locs/{dataset}-bars.pickle")
            else:
                barcode_use = utils.load_pickle(f"{barcode_dir}/{dataset}-bars.pickle")
            barcode_use = [code.decode('utf-8') for code in barcode_use.flatten()]
            barcode_flag = np.isin(barcode,barcode_use)
            barcode = barcode[barcode_flag]
            spot_locs = spot_locs[barcode_flag,:]
            # Load embeddings
            embs = {}
            window_masks = {}
            spot_weight = {}
            patches_loc_before_pad = {}
            father_patches = {}
            tmp = np.load(f"{emb_folder}/emb_{patch_size}/{dataset}.npz")
            keep_flag = tmp['flag']
            embs[patch_size] = tmp['emb'][keep_flag,:]
            patch_loc_before_pad = tmp['pos'][keep_flag,:] - [tmp['pad'][0]//2, tmp['pad'][1]//2]
            #! Switch the column of patch_loc_before_pad to match spot coordinate
            patch_loc_before_pad = patch_loc_before_pad[:,[1,0]]
            patches_loc_before_pad[patch_size] = patch_loc_before_pad
            #TODO Build adjacency matrix for patches here
            patch_center_loc_before_pad = patches_loc_before_pad[patch_size] + patch_size//2
            nearby_index_adj = find_nearby_spots(patch_center_loc_before_pad, patch_center_loc_before_pad, k=9, workers=num_workers)
            knn_edge = []
            for i in range(nearby_index_adj.shape[0]):
                for j in range(nearby_index_adj.shape[1]):
                    if nearby_index_adj[i,j] != i:
                        knn_edge.append([i, nearby_index_adj[i,j]])
            if len(knn_edge) > 0:
                knn_edge = np.array(knn_edge, dtype=np.int32)
                knn_edge = np.unique(knn_edge, axis=0).T
            else:
                knn_edge = np.empty((2, 0), dtype=np.int32)
            window_size = patch_size*4
            stride = patch_size*3
            windows_mask = []
            x_start = spot_locs[:,0].min()
            y_start = spot_locs[:,1].min()
            x_curr = x_start
            y_curr = y_start
            while y_curr < spot_locs[:,1].max():
                while x_curr < spot_locs[:,0].max():
                    mask_tmp = np.logical_and(np.logical_and(spot_locs[:,0] >= x_curr, spot_locs[:,0] < x_curr+window_size), np.logical_and(spot_locs[:,1] >= y_curr, spot_locs[:,1] < y_curr+window_size))
                    if mask_tmp.sum() > 1:
                        windows_mask.append(mask_tmp)
                        if plot_mask:
                            shift = 0
                            patch_mask = cv2.rectangle(patch_mask, (int(x_curr)+shift, int(y_curr)+shift), (int(x_curr+window_size)+shift, int(y_curr+window_size)+shift), plot_color, 20)
                            patch_mask = cv2.putText(patch_mask, str(patch_index), (int(x_curr),int(y_curr+window_size)), cv2.FONT_HERSHEY_SIMPLEX, 7, (255,255,255), 5, cv2.LINE_AA)
                            patch_index += 1
                    x_curr += stride
                x_curr = x_start
                y_curr += stride
                #
            windows_mask = np.array(windows_mask)
            window_masks[patch_size] = windows_mask
            # Spot weight mask
            patch_center_loc_before_pad = patch_loc_before_pad + patch_size//2
            spot_weight_mask = np.zeros((spot_locs.shape[0],patch_center_loc_before_pad.shape[0]))
            nearby_index = find_nearby_spots(patch_center_loc_before_pad,spot_locs,k=4, workers=num_workers)
            for i in range(spot_locs.shape[0]):
                for j in range(4):
                    spot_weight_mask[i,nearby_index[i,j]] = square_intersection_area(patch_center_loc_before_pad[nearby_index[i,j],:],patch_size,spot_locs[i,:],spot_radius_pixel*2) / ((spot_radius_pixel*2)**2)
            ## row normalize since some spots at the margin only intersect with 1 or 2 patches
            norm_tmp = spot_weight_mask.sum(axis=1)[:,None]
            norm_tmp[norm_tmp==0] = 1
            spot_weight_mask = spot_weight_mask / norm_tmp
            spot_weight[patch_size] = spot_weight_mask
            # For cells with no prop estimation, we set the prop to 0
            if load_cell_type:
                major_prop = major_prop.loc[:,cell_type_list]
                major_prop_filtered = np.zeros((len(barcode),len(cell_type_list)))
                for i in range(len(barcode)):
                    if barcode[i] in major_prop.index:
                        major_prop_filtered[i,:] = major_prop.loc[barcode[i],:].values
            else:
                major_prop_filtered = np.zeros((len(barcode),len(cell_type_list)))
            # Load gene expression
            cnts_file = f"{cnt_folder}/{dataset}-cnts.tsv"
            cnts = utils.load_tsv(cnts_file)
            cnts = cnts.loc[barcode,:]
            # Initialize an array with zeros for all genes in gene_list
            cnts_filtered = np.zeros((cnts.shape[0], len(gene_list)))
            # Fill in the gene expressions for the common genes
            for i, gene in enumerate(gene_list):
                if gene in cnts.columns:
                    cnts_filtered[:, i] = cnts[gene].values
            # Create an array indicating whether each gene in gene_list is present in cnts
            gene_mask = np.isin(gene_list, cnts.columns)
            # Load bulk gene expression
            bulk_file = f"{bulk_folder}/{dataset}-bulk.tsv"
            bulk = utils.load_tsv(bulk_file)
            # Initialize an array with zeros for all genes in bulk_hvg
            bulk_filtered = np.zeros(len(bulk_hvg))
            # Fill in the gene expressions for the common genes
            for i, gene in enumerate(bulk_hvg):
                if gene in bulk.index:
                    bulk_filtered[i] = bulk.loc[gene].values[0]

            bulk_rank = bulk_filtered/bulk_norm_factor
            bulk_rank = pd.Series(bulk_rank).rank(method = 'first', ascending=True, pct=True).values
            max_file = f"{bulk_folder}/{dataset}-max.tsv"
            max_exp = utils.load_tsv(max_file)
            # Initialize an array with zeros for all genes in bulk_hvg
            max_filtered = np.zeros(len(bulk_hvg))
            # Fill in the gene expressions for the common genes
            for i, gene in enumerate(bulk_hvg):
                if gene in bulk.index:
                    max_filtered[i] = max_exp.loc[gene].values[0]
            #
            data_tmp = (embs, window_masks,spot_weight, cnts_filtered, gene_mask, bulk_filtered, bulk_rank, max_filtered, father_patches, major_prop_filtered, knn_edge)
            data.append(data_tmp)
            # Save the data to a file
            utils.save_pickle(data_tmp,f"{file_dir}/{dataset.replace('.h5','')}.pkl")
    else:
        data = []
        for dataset in datasets:
            dataset = dataset.replace(".h5", "")
            data_tmp = utils.load_pickle(f"{file_dir}/{dataset}.pkl")
            data.append(data_tmp)
    return data

_INTEGER_DTYPES_UNSIGNED = (
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)

_INTEGER_DTYPES_SIGNED = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
)


def as_smallest_int_array(
    values,
    *,
    allow_unsigned: bool = True,
) -> np.ndarray:
    """
    Convert an integer-valued array to the smallest NumPy integer dtype
    that can represent every value without loss.

    Examples
    --------
    [0, 255]       -> uint8
    [0, 256]       -> uint16
    [-1, 100]      -> int8
    [-200, 100]    -> int16

    Parameters
    ----------
    values:
        Array-like containing integer or integer-valued floating-point data.

    allow_unsigned:
        When True, nonnegative arrays may use uint8/uint16/uint32/uint64.
        When False, signed integer dtypes are always used.

    Returns
    -------
    np.ndarray:
        C-contiguous array with the smallest safe integer dtype.
    """
    arr = np.asarray(values)

    if arr.size == 0:
        dtype = np.uint8 if allow_unsigned else np.int8
        return np.ascontiguousarray(arr, dtype=dtype)

    # Accept integer-valued floats, but never silently truncate them.
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise ValueError("Integer array contains NaN or infinity.")

        rounded = np.rint(arr)
        if not np.array_equal(arr, rounded):
            raise ValueError(
                "Array contains non-integer floating-point values; "
                "refusing to truncate them."
            )

    elif not (
        np.issubdtype(arr.dtype, np.integer)
        or np.issubdtype(arr.dtype, np.bool_)
    ):
        raise TypeError(
            f"Expected integer-valued data, got dtype {arr.dtype}."
        )

    min_value = int(np.min(arr))
    max_value = int(np.max(arr))

    if allow_unsigned and min_value >= 0:
        candidates = _INTEGER_DTYPES_UNSIGNED
    else:
        candidates = _INTEGER_DTYPES_SIGNED

    for dtype in candidates:
        limits = np.iinfo(dtype)
        if limits.min <= min_value and max_value <= limits.max:
            return np.ascontiguousarray(arr, dtype=dtype)

    raise OverflowError(
        f"Values [{min_value}, {max_value}] cannot be represented "
        "by a standard NumPy integer dtype."
    )