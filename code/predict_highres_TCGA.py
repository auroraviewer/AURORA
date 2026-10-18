import numpy as np
import pandas as pd
import os
import h5py
import torch
from tqdm import tqdm 
from train_utils import get_unique_indices, as_smallest_int_array, find_nearby_spots
from model import AURORA
import argparse
import json
import utils


def load_histosweep_mask(mask_path):
    """Load a HistoSweep 0/1 tissue mask as a (H, W) uint8 array."""
    import tifffile

    mask = np.asarray(tifffile.imread(str(mask_path)))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def histosweep_keep_flags(positions, mask, patch_size=14):
    """Boolean keep-flag per point against a HistoSweep tissue mask.

    ``positions`` is an (N, 2) array where column 0 indexes the mask's first
    axis (row / y) and column 1 the mask's second axis (col / x) -- i.e. the
    same order as ``patches_loc_dedup`` after the [:,[1,0]] swap.  ``mask`` is
    a (H, W) 0/1 tissue mask.  Prediction pixels and the mask may not be
    exactly aligned (e.g. padding offsets), so both are mapped onto the
    HistoSweep super-pixel grid via ``//patch_size``, where the mask is
    constant.  Out-of-bounds points are treated as filtered.
    """
    positions = np.asarray(positions)
    rows = positions[:, 0].astype(np.int64)  # mask first axis (row / y)
    cols = positions[:, 1].astype(np.int64)  # mask second axis (col / x)
    h, w = mask.shape
    in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    keep = np.zeros(positions.shape[0], dtype=bool)
    if in_bounds.any():
        sy = np.clip((rows[in_bounds] // patch_size) * patch_size, 0, h - 1)
        sx = np.clip((cols[in_bounds] // patch_size) * patch_size, 0, w - 1)
        keep[in_bounds] = mask[sy, sx] > 0
    return keep


def load_data_pred_TCGA_highres(datasets, main_dir, interim_dir, patch_sizes,
                                from_file=False, file_dir=None, num_workers=4):
    """Load high-res TCGA prediction data for one or more samples.

    Mirror of ``load_data_pred_TCGA`` (predict_TCGA.py) adapted to the
    high-resolution embeddings produced by extract_features_highres_optimized.py
    (``<interim>/UNI_multiscale_patches/emb_<ps>/<sample>.npz``).  It returns,
    per sample, the same ``(emb, father_lowres, father_patches,
    patches_loc_before_pad)`` tuple that ``load_data2_pred`` provides, so the
    predict_highres_TCGA loop consumes it unchanged.  Bulk expression is NOT
    loaded here (the high-res model is called with ``bulk=None``) and each
    sample is one item (no superpatch splitting).

    ``emb`` and ``patches_loc_before_pad`` are dicts keyed by patch_size.
    ``patches_loc_before_pad`` follows the ``load_data2_pred`` convention:
    (row, col) = (y, x) order, in original-image coordinates (the extraction
    padding offset is removed).  ``father_patches[ps]`` maps every fine patch to
    the index of its nearest coarse (father) patch center.
    """
    if not from_file:
        emb_folder = f"{main_dir}/{interim_dir}/UNI_multiscale_patches"
        data = []
        for dataset in tqdm(datasets):
            print(f"Loading data for {dataset}...")
            embs = {}
            patches_loc_before_pad = {}
            father_patches = {}
            for patch_size in patch_sizes:
                tmp = np.load(f"{emb_folder}/emb_{patch_size}/{dataset}.npz")
                keep_flag = tmp['flag']
                embs[patch_size] = tmp['emb'][keep_flag, :]
                patch_loc_before_pad = tmp['pos'][keep_flag, :] - [
                    tmp['pad'][0] // 2, tmp['pad'][1] // 2
                ]
                patches_loc_before_pad[patch_size] = patch_loc_before_pad[:,[1,0]]
            # Father-child relationship: for each fine patch, index of the
            # nearest coarse (father) patch center (same as load_data_pred_TCGA).
            patch_size_min = min(patch_sizes)
            for patch_size in patch_sizes:
                if patch_size != min(patch_sizes):
                    father_patches[patch_size] = find_nearby_spots(
                        patches_loc_before_pad[patch_size] + patch_size // 2,
                        patches_loc_before_pad[patch_size_min] + patch_size_min // 2,
                        k=1, workers=num_workers,
                    )
            # No additional low-res father scale for 2-scale high-res prediction.
            father_lowres = np.zeros(0, dtype=np.int64)
            data_tmp = (embs, father_lowres, father_patches, patches_loc_before_pad)
            data.append(data_tmp)
            utils.save_pickle(data_tmp, f'{file_dir}/{dataset.replace(".h5", "")}.pkl')
    else:
        data = []
        for dataset in datasets:
            dataset = dataset.replace(".h5", "")
            data_tmp = utils.load_pickle(f'{file_dir}/{dataset}.pkl')
            data.append(data_tmp)
    return data


parser = argparse.ArgumentParser(description='AURORA Prediction Script')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--interim_path', '-i', type=str, default='AURORA_interim', help='Interim data directory')
parser.add_argument('--pred_path', '-t', type=str, default='AURORA_pred', help='Directory to store predicted data')
# parser.add_argument('--patch_sizes', nargs='+', type=int, default=[3584, 896, 224], help='List of patch sizes')
# parser.add_argument('--metadata_path', '-m', type=str, default='Sample_metadata.csv', help='Sample metadata CSV file name')
# parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
# parser.add_argument('--plot', action='store_true', help='Plot intermediate results for checking')
# parser.add_argument('--log', action='store_true', help='Enable logging to TensorBoard')
parser.add_argument('--parameter_json', '-j', type=str, default='parameters.json', help='JSON file with parameters')
parser.add_argument('--model_pth', type=str, default='AURORA_model_weights.pth', help='Path to the pretrained model file')
parser.add_argument('--sample_sheet', type=str, default=None, help='Dataset directory name')
# parser.add_argument('--bulk', type=str, default='bulk_data', help='Bulk data file name')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for prediction')
# parser.add_argument('--save_embedding', action='store_true', help='Flag to save embeddings')
parser.add_argument('--uint8_output', action='store_true', help='Flag to save output in uint8 format')
parser.add_argument('--compress_output', action='store_true', help='Flag to compress output files')
parser.add_argument('--histosweep_mask_suffix', type=str, default='HistoSweep',
                    help="Suffix of the HistoSweep mask file (before '_mask_rescaled.tif') in "
                         "wsis_rescaled/. Predictions at filtered points are set to 0. "
                         "Set to '' to disable mask application.")
parser.add_argument('--histosweep_patch_size', type=int, default=14,
                    help='HistoSweep super-pixel size used to map prediction pixels onto the mask grid.')
args = parser.parse_args()

main_dir = args.project_path
## read json
with open(f"{main_dir}/{args.parameter_json}", 'r') as f:
    parameters = json.load(f)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_dir = f"{main_dir}/{args.pred_path}/processed_pkl_test_highres"
os.makedirs(dataset_dir, exist_ok=True)
if os.path.exists(dataset_dir):
    dataset_exist = [x.replace(".pkl","") for x in os.listdir(dataset_dir) if x.endswith(".pkl")]
else:
    dataset_exist = []

gene_list_file = f"{main_dir}/{parameters.get('gene_list_file',"AURORA_interim/Gene_to_predict.csv")}"
gene_list = pd.read_csv(gene_list_file)['Genes'].values
gene_list = list(gene_list)
hvg_list = gene_list
bulk_norm_factor = pd.read_csv(f"{main_dir}/{parameters.get('gene_norm_factor_file','AURORA_interim/Gene_normalize_factor.csv')}",index_col=0).loc[gene_list,'0'].values
input_size = parameters.get('img_emb_dim', 200)  # Length of embedding
output_size = len(gene_list)  # Number of genes to predict
num_workers = parameters.get('num_workers', 4)
patch_sizes = parameters.get('patch_sizes', [56, 14])  # List of patch sizes
bidirectional = parameters.get('bidirectional', False)
use_bulk = parameters.get('use_bulk', True)
pred_celltype = parameters.get('pred_celltype', True)
use_context_loss = parameters.get('use_context_loss', True)

# Load the data for each set
celltype_path = f"{main_dir}/{parameters.get('celltype_file', "AURORA_interim/deconvolution/cell_types.csv")}"
if pred_celltype:
    major_cell_type = pd.read_csv(celltype_path,header=None).values.squeeze()
else:
    major_cell_type = []
meta = pd.read_csv(f"{main_dir}/{args.sample_sheet}", header = 0)
dataset_test = meta['Sample_name'].values
if 'Bulk_sample_name' in meta.columns:
    bulk_sample_names = meta['Bulk_sample_name'].values
else:
    bulk_sample_names = dataset_test

# bulk = pd.read_csv(f"{main_dir}/{args.bulk}", index_col=0)

print("Loading pretrained model")
model = AURORA(input_size, output_size, bulk_size=len(hvg_list),
                celltype_num=len(major_cell_type),latent_size=256,
                device=device,bidirectional=bidirectional,
                use_bulk=use_bulk, pred_celltype=pred_celltype)
model.to(device)
state_dict = torch.load(f"{main_dir}/{args.model_pth}", weights_only=False, map_location=device)
model.load_state_dict(state_dict)
print("Starting evaluation")
model.eval()
# os.makedirs(f"{main_dir}/{args.pred_path}/processed_pkl", exist_ok=True)
os.makedirs(f"{main_dir}/{args.pred_path}/pred", exist_ok=True)

# sample_num = 1 #args.batch_size
# idx_total = len(dataset_test)//sample_num + 1
test_batch_size = args.batch_size

for idx in range(len(dataset_test)):
    dataset_test_use = dataset_test[idx]

    # Load the HistoSweep tissue mask (optional).  Predictions at filtered
    # (background / low-quality) points are zeroed and then dropped.
    histosweep_mask = None
    if args.histosweep_mask_suffix:
        histosweep_mask_path = (
            f"{main_dir}/{args.interim_path}/UNI_multiscale_patches/wsis_rescaled/"
            f"{dataset_test_use}_{args.histosweep_mask_suffix}_mask_rescaled.tif"
        )
        if os.path.exists(histosweep_mask_path):
            histosweep_mask = load_histosweep_mask(histosweep_mask_path)
            print(f"    Loaded HistoSweep mask: {histosweep_mask_path} "
                  f"({histosweep_mask.shape})")
        else:
            print(f"    [warn] HistoSweep mask not found: {histosweep_mask_path}; "
                  "predictions left unchanged")

    with torch.no_grad():
        # running_loss = 0.0
        if set([dataset_test_use]) <= set(dataset_exist):
            print("Samples loading from disk")
            test_data = load_data_pred_TCGA_highres(
                [dataset_test_use],
                main_dir=main_dir, interim_dir=args.interim_path,
                patch_sizes=patch_sizes,
                from_file=True, file_dir=dataset_dir, num_workers=num_workers,
            )
        else:
            print("Samples loading from scratch")
            test_data = load_data_pred_TCGA_highres(
                [dataset_test_use],
                main_dir=main_dir, interim_dir=args.interim_path,
                patch_sizes=patch_sizes,
                from_file=False, file_dir=dataset_dir, num_workers=num_workers,
            )
        test_loader_tqdm = tqdm(
            np.arange(len(test_data)), desc=f"Evaluation", unit="batch", leave=False
        )
        spot_preds = {}
        major_prop_preds = {}
        for index in test_loader_tqdm:
            emb, father_lowres, father_patches, patches_loc_before_pad = test_data[index]
            # Keep the full embeddings on CPU (as tensors sharing the numpy
            # buffers where possible) and only move per-batch slices to the GPU,
            # so a large superpatch never has to fit into GPU memory at once.
            for key in emb.keys():
                emb[key] = torch.as_tensor(emb[key], dtype=torch.float32)
            father_lowres = torch.as_tensor(father_lowres, dtype=torch.int)
            for key in father_patches.keys():
                father_patches[key] = torch.as_tensor(father_patches[key], dtype=torch.int)
            # Forward pass, chunked: gather the coarse/fine hierarchy per batch
            # and run the model on `test_batch_size` patches at a time. Same
            # [fine, coarse] stacking order as before (the full tensor was
            # previously built on GPU and reversed along dim 1).
            n_fine = emb[patch_sizes[1]].shape[0]
            father_coarse = father_patches[patch_sizes[0]]
            emb_coarse = emb[patch_sizes[0]]
            emb_fine = emb[patch_sizes[1]]
            total_batch = n_fine//test_batch_size + 1
            # Pre-allocate the full outputs once and stream each batch into
            # them in the final [coarse, fine] order.  Storing every per-batch
            # tensor in a list and then re-copying through torch.cat + the
            # [:,[1,0]] swap transiently needs ~3x the output size and OOMs on
            # large WSIs (same fix as predict_highres_TCGA_optimized.py).
            output = torch.empty((n_fine, 2, output_size), dtype=torch.float32)
            major_prop_pred = torch.empty(
                (n_fine, 2, len(major_cell_type)), dtype=torch.float32
            )
            for batch in tqdm(range(total_batch), desc=f"Processing {dataset_test[index]}", unit="batch", leave=False):
                batch_start = batch*test_batch_size
                batch_end = min((batch+1)*test_batch_size, n_fine)
                if batch_start >= batch_end:
                    continue
                emb_hierrachy_batch = torch.stack(
                    [emb_fine[batch_start:batch_end,:],
                     emb_coarse[father_coarse[batch_start:batch_end],:]],
                    dim=1,
                ).to(device)
                output_batch, major_prop_pred_batch = model(emb_hierrachy_batch, None, highres=True)
                # Model returns [fine, coarse] along dim 1; store directly in
                # the final [coarse, fine] order so no full-array swap copy is
                # needed afterwards.
                output[batch_start:batch_end, 1, :] = output_batch[:, 0, :].cpu()
                output[batch_start:batch_end, 0, :] = output_batch[:, 1, :].cpu()
                major_prop_pred[batch_start:batch_end, 1, :] = major_prop_pred_batch[:, 0, :].cpu()
                major_prop_pred[batch_start:batch_end, 0, :] = major_prop_pred_batch[:, 1, :].cpu()
                del output_batch, major_prop_pred_batch
            output = output.numpy()
            major_prop_pred = major_prop_pred.numpy()

            # Remove replicate elements for patch_size > min(patch_size): for the
            # coarse levels exp_pred/prop_pred are shared by all min-size patches
            # under the same father patch, so we keep only one row per unique father
            # (same dedup as in test_highres.py). 'output'/'major_prop_pred' become
            # dicts keyed by patch_size. Expand back with:
            #   row = np.searchsorted(father_unique[ps], father_patches[ps])
            #   full = compact[row]
            father_unique_all = {}
            inverse_index_all = {}
            output_dedup = {}
            major_prop_dedup = {}
            patches_loc_dedup = {}
            for patch_size_index in range(len(patch_sizes)):
                patch_size = patch_sizes[patch_size_index]
                if patch_size != min(patch_sizes):
                    fu, ii = get_unique_indices(father_patches[patch_size].cpu(), device='cpu')
                    father_unique_all[patch_size] = fu.numpy()
                    inverse_index_all[patch_size] = ii.numpy()
                    output_dedup[patch_size] = output[ii.numpy(), patch_size_index, :]
                    major_prop_dedup[patch_size] = major_prop_pred[ii.numpy(), patch_size_index, :]
                    # One row per unique father: index the locations by the unique
                    # father values (they are indices into patches_loc_before_pad),
                    # so the positions stay aligned with output_dedup rows.
                    patches_loc_dedup[patch_size] = patches_loc_before_pad[patch_size][fu.numpy(), :]
                else:
                    output_dedup[patch_size] = output[:, patch_size_index, :]
                    major_prop_dedup[patch_size] = major_prop_pred[:, patch_size_index, :]
                    patches_loc_dedup[patch_size] = patches_loc_before_pad[patch_size]
                patches_loc_dedup[patch_size] = patches_loc_dedup[patch_size][:,[1,0]]  #! Switch back to (x,y) order for saving
            # Free the full float buffers early
            del output, major_prop_pred

            # Zero HistoSweep-filtered points so they become all-zero rows and
            # are dropped below; every remaining row stays aligned with its
            # position.  This shrinks all saved arrays to the predicted points.
            # NOTE: patches_loc_dedup was swapped to (x, y) at the [:,[1,0]]
            # above, so column 0 is the mask's FIRST axis (row / y) and column
            # 1 is the mask's SECOND axis (col / x).
            if histosweep_mask is not None:
                for patch_size in patch_sizes:
                    keep = histosweep_keep_flags(
                        patches_loc_dedup[patch_size],
                        histosweep_mask,
                        patch_size=args.histosweep_patch_size,
                    )
                    output_dedup[patch_size][~keep] = 0.0
                    major_prop_dedup[patch_size][~keep] = 0.0

            # Drop all-zero rows (e.g. HistoSweep-filtered points) so only
            # predicted points are stored; positions stay row-aligned.  A row is
            # kept when at least one gene has a nonzero prediction.
            row_keep_all = {}
            for patch_size in patch_sizes:
                row_keep = output_dedup[patch_size].any(axis=1)
                row_keep_all[patch_size] = row_keep
                if not row_keep.all():
                    output_dedup[patch_size] = output_dedup[patch_size][row_keep]
                    major_prop_dedup[patch_size] = major_prop_dedup[patch_size][row_keep]
                    patches_loc_dedup[patch_size] = patches_loc_dedup[patch_size][row_keep]

            if not args.uint8_output:
                # Make the saved object as small as possible: exp/log-prop
                # predictions become float32 CSR sparse matrices (masked-out,
                # all-zero rows are stored implicitly), and positions use the
                # smallest integer dtype. Read back with e.g.
                #   d = utils.load_pickle(path); d['exp_pred'][ps].toarray()
                from scipy.sparse import csr_matrix

                exp_pred_sparse = {
                    ps: csr_matrix(arr.astype(np.float32))
                    for ps, arr in output_dedup.items()
                }
                log_prop_pred_sparse = {
                    ps: csr_matrix(arr.astype(np.float32))
                    for ps, arr in major_prop_dedup.items()
                }
                pos_small = {
                    ps: as_smallest_int_array(loc, allow_unsigned=True)
                    for ps, loc in patches_loc_dedup.items()
                }
                utils.save_pickle(
                    {
                        'exp_pred': exp_pred_sparse,
                        'log_prop_pred': log_prop_pred_sparse,
                        'pos': pos_small,
                        'genes': gene_list,
                        'cell_types': major_cell_type,
                        'resolutions': parameters['patch_size'],
                        'sparse': True,
                    },
                    f"{main_dir}/{args.pred_path}/pred/{dataset_test_use}-highres-pred.pickle",
                )
            else:
                # ===== Compress exp_pred and prop_pred to uint8 =====
                # Keep predictions from ALL patch sizes (deduplicated per level).
                # Normalize each channel by its maximum value within each
                # patch_size level (per-patch-size per-channel maxima), keep the
                # maxima for de-normalization, then scale by 255 and cast to uint8.
                # prop_pred is a log-softmax output (all values <= 0), so we
                # exponentiate it first to get cell-type proportions in [0, 1].
                prop_probs_dedup = {ps: np.exp(arr) for ps, arr in major_prop_dedup.items()}

                # Per-patch-size max normalization + uint8 quantization.
                # All-zero rows were already dropped, so maxima only reflect
                # retained predicted points; guard a fully-dropped level.
                exp_max = {}
                prop_max = {}
                for ps in patch_sizes:
                    if output_dedup[ps].shape[0] == 0:
                        exp_max[ps] = np.zeros(output_dedup[ps].shape[1], dtype=np.float32)
                        prop_max[ps] = np.zeros(prop_probs_dedup[ps].shape[1], dtype=np.float32)
                    else:
                        exp_max[ps] = output_dedup[ps].max(axis=0)
                        prop_max[ps] = prop_probs_dedup[ps].max(axis=0)
                exp_max_safe = {ps: np.where(exp_max[ps] > 0, exp_max[ps], 1.0) for ps in patch_sizes}
                prop_max_safe = {ps: np.where(prop_max[ps] > 0, prop_max[ps], 1.0) for ps in patch_sizes}
                exp_pred_uint8 = {ps: np.rint((output_dedup[ps] / exp_max_safe[ps] * 255.0).clip(0, 255)).astype(np.uint8)
                                  for ps in patch_sizes}
                prop_pred_uint8 = {ps: np.rint((prop_probs_dedup[ps] / prop_max_safe[ps] * 255.0).clip(0, 255)).astype(np.uint8)
                                   for ps in patch_sizes}
                # Structural (memory-saving) representation shared by both output paths:
                #   (better for compression AND smaller on disk). Reverse on load with
                #   np.transpose(arr, (1, 0)).
                # * patch coordinates cast to int32 (safe: < 2^31 pixels).
                # * exp_max/prop_max cast to float32 (negligible precision loss).
                exp_pred_mem = {ps: np.ascontiguousarray(exp_pred_uint8[ps]) for ps in patch_sizes}
                prop_pred_mem = {ps: np.ascontiguousarray(prop_pred_uint8[ps]) for ps in patch_sizes}
                exp_max_mem = {ps: exp_max[ps].astype(np.float32) for ps in patch_sizes}
                prop_max_mem = {ps: prop_max[ps].astype(np.float32) for ps in patch_sizes}
                # patch_loc_mem = {ps: loc.astype(np.int32) for ps, loc in patches_loc_before_pad.items()}
                patch_loc_mem = {}
                for ps, loc in patches_loc_dedup.items():
                    if loc.shape[0] == 0:
                        patch_loc_mem[ps] = loc.astype(np.int32)
                    else:
                        patch_loc_mem[ps] = as_smallest_int_array(loc, allow_unsigned=True)
                # Free the big float buffers now: only the quantized ``*_mem``
                # dicts are needed for writing.
                del output_dedup, major_prop_dedup, prop_probs_dedup, patches_loc_dedup
                del exp_max, prop_max, exp_max_safe, prop_max_safe, exp_pred_uint8, prop_pred_uint8
                if not args.compress_output:
                    # Write to HDF5 with the same dataset layout as predict_TCGA.py.
                    # gzip keeps the file small and the write fast for high-res
                    # (millions of patches); h5py readers transparently
                    # decompress, so the format is unchanged. Remove
                    # `compression='gzip'` for byte-identical files to predict_TCGA.
                    h5_path = f"{main_dir}/{args.pred_path}/pred/{dataset_test_use}-highres-pred.h5"
                    with h5py.File(h5_path, 'w') as hf:
                        hf.create_dataset('genes', data=np.asarray(gene_list, dtype=h5py.string_dtype(encoding='utf-8')))
                        hf.create_dataset('cell_types', data=np.asarray(major_cell_type, dtype=h5py.string_dtype(encoding='utf-8')))
                        hf.create_dataset('resolutions', data=np.asarray(parameters['patch_size']))
                        for ps in patch_sizes:
                            print(f"    Writing {dataset_test_use} patch_size {ps} ...", flush=True)
                            if exp_pred_mem[ps].shape[0] > 0:
                                hf.create_dataset(f'exp_pred_{ps}', data=exp_pred_mem[ps], compression='gzip')
                                hf.create_dataset(f'prop_pred_{ps}', data=prop_pred_mem[ps], compression='gzip')
                                hf.create_dataset(f'patch_loc_{ps}', data=patch_loc_mem[ps], compression='gzip')
                            else:
                                # Fully-dropped level: keep the datasets present
                                # but empty (compression needs >0 rows).
                                hf.create_dataset(f'exp_pred_{ps}', data=exp_pred_mem[ps])
                                hf.create_dataset(f'prop_pred_{ps}', data=prop_pred_mem[ps])
                                hf.create_dataset(f'patch_loc_{ps}', data=patch_loc_mem[ps])
                            hf.create_dataset(f'exp_max_{ps}', data=exp_max_mem[ps])
                            hf.create_dataset(f'prop_max_{ps}', data=prop_max_mem[ps])
                    # Free the quantized arrays before the next superpatch
                    del exp_pred_mem, prop_pred_mem, exp_max_mem, prop_max_mem, patch_loc_mem
                    print(f"HDF5 result saved to {h5_path} ({os.path.getsize(h5_path)/1e6:.2f} MB)")
                else:
                    # Compress the uint8 result to .npz.zst with high compression rate.
                    # The arrays are packed into an in-memory .npz (zip container),
                    # then the whole byte stream is compressed with zstandard (level 18)
                    # and written to a .npz.zst file.
                    import io
                    import zstandard as zstd

                    save_dict = {
                        'genes': np.array(gene_list),
                        'cell_types': np.array(major_cell_type),
                        'resolutions': np.array(parameters['patch_size']),
                        'patch_sizes': np.array(patch_sizes),
                    }
                    for ps in patch_sizes:
                        save_dict[f'exp_pred_{ps}'] = exp_pred_mem[ps]
                        save_dict[f'prop_pred_{ps}'] = prop_pred_mem[ps]
                        save_dict[f'exp_max_{ps}'] = exp_max_mem[ps]        # [len(gene_list)] per level
                        save_dict[f'prop_max_{ps}'] = prop_max_mem[ps]      # [len(cell_types)] per level
                        save_dict[f'patch_loc_{ps}'] = patch_loc_mem[ps]
                        # if ps != min(patch_sizes):
                        #     # Keep only the retained coarse rows so these stay
                        #     # aligned with the (row-dropped) exp_pred_{ps} arrays.
                        #     rk = row_keep_all[ps]
                        #     save_dict[f'father_unique_{ps}'] = father_unique_all[ps][rk]
                        #     save_dict[f'inverse_index_{ps}'] = inverse_index_all[ps][rk]

                    buf = io.BytesIO()
                    np.savez(buf, **save_dict)
                    buf.seek(0)
                    compressed = zstd.ZstdCompressor(level=18).compress(buf.read())
                    out_path = f"{main_dir}/{args.pred_path}/pred/{dataset_test_use}-highres-pred.npz.zst"
                    with open(out_path, 'wb') as f:
                        f.write(compressed)
                    print(f"Compressed result saved to {out_path} "
                          f"({len(compressed)/1e6:.2f} MB)")

