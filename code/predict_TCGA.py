import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm 
from train_utils import get_unique_indices, as_smallest_int_array, find_nearby_spots
from model import AURORA
import argparse
import json
import h5py
import utils
def load_data_pred_TCGA(datasets, main_dir, bulk_hvg, bulk_norm_factor, bulk_metadata_name = "gdc_sample_sheet.2025-08-04.tsv",
                        bulk_file_name = "TCGA_LUAD_expression.csv",
              patch_sizes=[3584, 896, 224], from_file = False, file_dir = None, num_workers=4):
    if not from_file:
        data = []
        emb_folder = f"{main_dir}/UNI_patch"
        # bulk_folder = f"{main_dir}/bulk_RNA"
        bulk_metadata_file = f"{main_dir}/{bulk_metadata_name}"
        bulk_metadata = pd.read_csv(bulk_metadata_file, sep='\t', index_col=6)
        # remove duplicated index
        bulk_metadata = bulk_metadata.loc[~bulk_metadata.index.duplicated(keep='first')]
        # Load bulk gene expression
        bulk_file = f"{main_dir}/{bulk_file_name}"
        bulk = pd.read_csv(bulk_file, index_col=0,header=0)
        image_metadata_file = f"{main_dir}/wsis_sample_sheet_filtered.csv"
        image_metadata = pd.read_csv(image_metadata_file,header=0)
        for dataset in tqdm(datasets):
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
            sample_id = image_metadata.loc[image_metadata['File Name'].str.contains(dataset),"Sample ID"].values[0]
            bulk_sample_name = bulk_metadata.loc[sample_id,'File ID']
            # Fill in the gene expressions for the common genes
            for i, gene in enumerate(bulk_hvg):
                if gene in bulk.index:
                    bulk_filtered[i] = bulk.loc[gene,bulk_sample_name]
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

parser = argparse.ArgumentParser(description='AURORA Prediction Script')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
# parser.add_argument('--interim_path', '-i', type=str, default='AURORA_interim', help='Interim data directory')
parser.add_argument('--pred_path', '-t', type=str, default='AURORA_pred', help='Directory to store predicted data')
# parser.add_argument('--patch_sizes', nargs='+', type=int, default=[3584, 896, 224], help='List of patch sizes')
# parser.add_argument('--metadata_path', '-m', type=str, default='Sample_metadata.csv', help='Sample metadata CSV file name')
# parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
# parser.add_argument('--plot', action='store_true', help='Plot intermediate results for checking')
# parser.add_argument('--log', action='store_true', help='Enable logging to TensorBoard')
parser.add_argument('--parameter_json', '-j', type=str, default='parameters.json', help='JSON file with parameters')
parser.add_argument('--model_pth', type=str, default='AURORA_model_weights.pth', help='Path to the pretrained model file')
# parser.add_argument('--sample_sheet', type=str, default=None, help='Dataset directory name')
# parser.add_argument('--bulk', type=str, default='bulk_data', help='Bulk data file name')
parser.add_argument('--bulk_metadata_name', type=str, default='gdc_sample_sheet.2025-08-04.tsv', help='Bulk metadata file name')
parser.add_argument('--bulk_file_name', type=str, default='TCGA_LUAD_expression.csv', help='Bulk gene expression file name')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for prediction')
parser.add_argument('--uint8_output', action='store_true', help='Flag to save output in uint8 format')
parser.add_argument('--only_224', action='store_true', help='Flag to use only 224x224 patches')
args = parser.parse_args()

main_dir = args.project_path
## read json
with open(f"{main_dir}/{args.parameter_json}", 'r') as f:
    parameters = json.load(f)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_dir = f"{main_dir}/{args.pred_path}/processed_pkl"
if os.path.exists(dataset_dir):
    dataset_exist = [x.replace(".pkl","") for x in os.listdir(dataset_dir) if x.endswith(".pkl")]
else:
    dataset_exist = []

gene_list_file = f"{main_dir}/{parameters.get('gene_list_file',"AURORA_interim/Gene_to_predict.csv")}"
gene_list = pd.read_csv(gene_list_file)['Genes'].values
gene_list = list(gene_list)
hvg_list = gene_list
bulk_norm_factor = pd.read_csv(f"{main_dir}/{parameters.get('gene_norm_factor_file','AURORA_interim/Gene_normalize_factor.csv')}",index_col=0).loc[gene_list,'0'].values
input_size = parameters.get('img_emb_dim', 1536)  # Length of embedding
output_size = len(gene_list)  # Number of genes to predict
num_workers = parameters.get('num_workers', 4)
patch_sizes = parameters.get('patch_sizes', [3584, 896, 224])  # List of patch sizes
pixel_size = parameters.get('pixel_size', 0.5)
bidirectional = parameters.get('bidirectional', False)
use_bulk = parameters.get('use_bulk', True)
pred_celltype = parameters.get('pred_celltype', True)
use_context_loss = parameters.get('use_context_loss', True)

# Load the data for each set
celltype_path = f"{main_dir}/{parameters.get('celltype_file', "AURORA_interim/deconvolution/cell_types.csv")}"
major_cell_type = pd.read_csv(celltype_path,header=None).values.squeeze()
dataset_list = pd.read_csv(f"{main_dir}/wsis_sample_sheet_filtered.csv",index_col=0)
dataset_test = list(dataset_list['File Name'].values)
dataset_test = [x.split('.')[0] for x in dataset_test]

# meta = pd.read_csv(f"{main_dir}/{args.sample_sheet}", header = 0)
# dataset_test = meta['Sample_name'].values
# if 'Bulk_sample_name' in meta.columns:
#     bulk_sample_names = meta['Bulk_sample_name'].values
# else:
#     bulk_sample_names = dataset_test

print("Loading pretrained model")
model = AURORA(input_size, output_size, bulk_size=len(hvg_list),
                celltype_num=len(major_cell_type),
                device=device,bidirectional=bidirectional,
                use_bulk=use_bulk, pred_celltype=pred_celltype)
model.to(device)
state_dict = torch.load(f"{main_dir}/{args.model_pth}", weights_only=False, map_location=device)
model.load_state_dict(state_dict)
print("Starting evaluation")
model.eval()
# os.makedirs(f"{main_dir}/{args.pred_path}/processed_pkl", exist_ok=True)
os.makedirs(f"{main_dir}/{args.pred_path}/pred", exist_ok=True)

sample_num = args.batch_size
idx_total = len(dataset_test)//sample_num + 1

for idx in range(idx_total):
    idx_end = min(len(dataset_test), (idx+1)*sample_num)
    dataset_test_use = dataset_test[idx*sample_num:idx_end]
    # bulk_sample_names_use = bulk_sample_names[idx*sample_num:idx_end]
    with torch.no_grad():
        # running_loss = 0.0
        if set(list(dataset_test_use)) <= set(dataset_exist):
            print("Samples loading from disk")
            test_data = load_data_pred_TCGA(list(dataset_test_use),
                                    main_dir=main_dir, bulk_hvg=hvg_list, 
                                    bulk_norm_factor=bulk_norm_factor,
                                    bulk_metadata_name=args.bulk_metadata_name, bulk_file_name=args.bulk_file_name,
                                    patch_sizes=patch_sizes, 
                                    from_file=True, file_dir=dataset_dir, num_workers=num_workers)
        else:
            print("Samples loading from scratch")
            test_data = load_data_pred_TCGA(list(dataset_test_use), 
                                    main_dir=main_dir, bulk_hvg=hvg_list, 
                                    bulk_norm_factor=bulk_norm_factor, 
                                    bulk_metadata_name=args.bulk_metadata_name, bulk_file_name=args.bulk_file_name,
                                    patch_sizes=patch_sizes, 
                                    from_file=False, file_dir=dataset_dir, num_workers=num_workers)
        test_loader_tqdm = tqdm(
            np.arange(len(test_data)), desc=f"Evaluation", unit="batch", leave=False
        )
        for index in test_loader_tqdm:
            emb, bulk, bulk_rank, father_patches, patches_loc_before_pad = test_data[index]
            for key in emb.keys():
                emb[key] = torch.tensor(emb[key], dtype=torch.float32).to(device)
            for key in father_patches.keys():
                father_patches[key] = torch.tensor(father_patches[key], dtype=torch.int).to(device)
            bulk = torch.tensor(bulk, dtype=torch.float32).to(device)
            bulk_rank = torch.tensor(bulk_rank, dtype=torch.float32).to(device)
            # Forward pass
            emb_hierrachy = torch.stack([emb[patch_sizes[0]][father_patches[patch_sizes[0]],:],
                                                    emb[patch_sizes[1]][father_patches[patch_sizes[1]],:],
                                                    emb[patch_sizes[2]]],dim=1)
            output, major_prop_pred = model(emb_hierrachy, bulk_rank)
            output = output.cpu().numpy()
            major_prop_pred = major_prop_pred.detach().cpu().numpy()

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

            if not args.uint8_output:
                utils.save_pickle({'exp_pred': output_dedup, 'log_prop_pred': major_prop_dedup,
                                    #    'father_unique': father_unique_all, 'inverse_index': inverse_index_all,
                                    'pos': patches_loc_dedup,
                                    'genes': gene_list, 'cell_types': major_cell_type,
                                    'resolutions': parameters['patch_size']},
                                f"{main_dir}/{args.pred_path}/pred/{dataset_test_use[index]}-pred.pickle")
            else:
                # ===== Quantize to uint8 and save as HDF5 =====
                # Keep predictions from the needed patch sizes (deduplicated per
                # level; with --only_224 just the 224 level). Normalize each
                # channel by its maximum value within each patch_size level
                # (per-patch-size per-channel maxima), keep the maxima for
                # de-normalization, then scale by 255 and cast to uint8.
                # prop_pred is a log-softmax output (all values <= 0), so we
                # exponentiate it first to get cell-type proportions in [0, 1].
                ps_use = [224] if args.only_224 else patch_sizes
                prop_probs_dedup = {ps: np.exp(major_prop_dedup[ps]) for ps in ps_use}

                # Per-patch-size max normalization + uint8 quantization
                exp_max = {ps: output_dedup[ps].max(axis=0) for ps in ps_use}
                prop_max = {ps: prop_probs_dedup[ps].max(axis=0) for ps in ps_use}
                exp_max_safe = {ps: np.where(exp_max[ps] > 0, exp_max[ps], 1.0) for ps in ps_use}
                prop_max_safe = {ps: np.where(prop_max[ps] > 0, prop_max[ps], 1.0) for ps in ps_use}
                exp_pred_uint8 = {ps: np.rint((output_dedup[ps] / exp_max_safe[ps] * 255.0).clip(0, 255)).astype(np.uint8)
                                    for ps in ps_use}
                prop_pred_uint8 = {ps: np.rint((prop_probs_dedup[ps] / prop_max_safe[ps] * 255.0).clip(0, 255)).astype(np.uint8)
                                    for ps in ps_use}
                # Free the large float buffers before writing to disk
                del prop_probs_dedup, output_dedup, major_prop_dedup
                del exp_max_safe, prop_max_safe

                # Patch coordinates as the smallest int dtype that fits
                patch_loc_mem = {
                    ps: as_smallest_int_array(
                        patches_loc_dedup[ps],
                        allow_unsigned=True,
                    )
                    for ps in ps_use
                }

                # Write to HDF5 (uncompressed, contiguous datasets). h5py streams
                # each array straight from memory to disk, so no extra full-size
                # in-memory compressed buffer is needed.
                h5_path = f"{main_dir}/{args.pred_path}/pred/{dataset_test_use[index]}-pred.h5"
                resolutions = parameters['patch_size'] if not args.only_224 else [224]
                with h5py.File(h5_path, 'w') as hf:
                    hf.create_dataset('genes', data=np.asarray(gene_list, dtype=h5py.string_dtype(encoding='utf-8')))
                    hf.create_dataset('cell_types', data=np.asarray(major_cell_type, dtype=h5py.string_dtype(encoding='utf-8')))
                    hf.create_dataset('resolutions', data=np.asarray(resolutions))
                    for ps in ps_use:
                        hf.create_dataset(f'exp_pred_{ps}', data=exp_pred_uint8[ps])
                        hf.create_dataset(f'prop_pred_{ps}', data=prop_pred_uint8[ps])
                        hf.create_dataset(f'patch_loc_{ps}', data=patch_loc_mem[ps])
                        hf.create_dataset(f'exp_max_{ps}', data=exp_max[ps].astype(np.float32))
                        hf.create_dataset(f'prop_max_{ps}', data=prop_max[ps].astype(np.float32))
                        # if ps != min(patch_sizes):
                        #     hf.create_dataset(f'father_unique_{ps}', data=father_unique_all[ps])
                        #     hf.create_dataset(f'inverse_index_{ps}', data=inverse_index_all[ps])
                # Free the quantized arrays before the next superpatch
                del exp_pred_uint8, prop_pred_uint8, patch_loc_mem, exp_max, prop_max
                print(f"HDF5 result saved to {h5_path} ({os.path.getsize(h5_path)/1e6:.2f} MB)")

