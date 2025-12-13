import json
import tifffile
import os
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import KDTree
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import torch
import torch.nn.functional as F
import utils
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
              bulk_hvg, bulk_norm_factor, cell_type_list,
              patch_sizes=[3584, 896, 224], plot_mask = True, 
              from_file = False, file_dir = None, num_workers=4,
              pixel_size=0.5):
    if not from_file:
        import os
        os.makedirs(f"{main_dir}/{train_dir}/image_check/",exist_ok=True)
        # We load the intersection between gene_list and gene detected in the dataset. So, each dataset's loss have different number of genes involved.
        data = []
        emb_folder = f"{main_dir}/{interim_dir}/UNI_multiscale_patches"
        cnt_folder = f"{main_dir}/{interim_dir}/cnts"
        bulk_folder = f"{main_dir}/{interim_dir}/bulk"
        prop_folder = f"{main_dir}/{interim_dir}/deconvolution"
        for dataset in datasets:
            dataset = dataset.replace(".h5", "")
            if plot_mask:
                image, spot_radius_pixel = load_image(dataset, main_dir = main_dir, processed_data_dir = processed_data_dir, 
                                                      interim_dir = interim_dir, target_radius=pixel_size)
            # Load spot locations
            locs_file = f"{emb_folder}/wsis_rescaled/{dataset}_loc_rescaled.npy"
            barcode = utils.load_pickle(f"{emb_folder}/wsis_rescaled/{dataset}_barcode.pickle")
            #! Notice that this the first column of loc is the col index for the image matrix, which is the x axis direction on the plot!
            spot_locs = np.load(locs_file)
            # Load cell type proportions
            major_prop = pd.read_csv(f"{prop_folder}/{dataset}_prop.csv",index_col=0,header=0)
            # Only keep spots used in cnts
            barcode_use = utils.load_pickle(f"{main_dir}/{interim_dir}/UNI_multiscale_patches/locs/{dataset}-bars.pickle")
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
            for patch_size in patch_sizes:
                tmp = np.load(f"{emb_folder}/emb_{patch_size}/{dataset}.npz")
                keep_flag = tmp['flag']
                embs[patch_size] = tmp['emb'][keep_flag,:]
                patch_loc_before_pad = tmp['pos'][keep_flag,:] - [tmp['pad'][0]//2, tmp['pad'][1]//2]
                #! Switch the column of patch_loc_before_pad to match spot coordinate
                patch_loc_before_pad = patch_loc_before_pad[:,[1,0]]
                patches_loc_before_pad[patch_size] = patch_loc_before_pad
                #
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
            major_prop = major_prop.loc[:,cell_type_list]
            major_prop_filtered = np.zeros((len(barcode),len(cell_type_list)))
            for i in range(len(barcode)):
                if barcode[i] in major_prop.index:
                    major_prop_filtered[i,:] = major_prop.loc[barcode[i],:].values
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
            data_tmp = (embs, window_masks,spot_weight, cnts_filtered, gene_mask, bulk_filtered, bulk_rank, max_filtered, father_patches, major_prop_filtered)
            data.append(data_tmp)
            # Save the data to a file
            utils.save_pickle(data_tmp,f"{file_dir}/{dataset.replace(".h5","")}.pkl")
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
            utils.save_pickle(data_tmp,f"{file_dir}/{dataset.replace(".h5","")}.pkl")
    else:
        data = []
        for dataset in datasets:
            dataset = dataset.replace(".h5", "")
            data_tmp = utils.load_pickle(f"{file_dir}/{dataset}.pkl")
            data.append(data_tmp)
    return data