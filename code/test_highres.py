import pandas as pd
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import utils 
from torch.utils.tensorboard import SummaryWriter
from train_utils import load_data2, masked_focal_kl_loss, get_unique_indices, project_spot_weight, project_spot_weight_columns
from datasets import HistologyGeneExpressionDataset, my_collate
from model import AURORA
import argparse
import json
import numpy as np
parser = argparse.ArgumentParser(description='Rescale H&E images to target pixel size and adjust ST spot locations accordingly.')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--interim_path', '-i', type=str, default='AURORA_interim', help='Interim data directory')
parser.add_argument('--train_path', '-t', type=str, default='AURORA_train', help='Directory to store training data')
# parser.add_argument('--patch_sizes', nargs='+', type=int, default=[3584, 896, 224], help='List of patch sizes')
# parser.add_argument('--metadata_path', '-m', type=str, default='Sample_metadata.csv', help='Sample metadata CSV file name')
parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
parser.add_argument('--plot', action='store_true', help='Plot intermediate results for checking')
parser.add_argument('--parameter_json', '-j', type=str, default='parameters.json', help='JSON file with parameters')
parser.add_argument('--test_samples', type=str, required=True, help='Test sample CSV file name')
parser.add_argument('--model_pth', type=str, default='AURORA_model_weights.pth', help='Path to the pretrained model file')
args = parser.parse_args()

main_dir = args.project_path


def load_patch_locations(dataset):
    """Load padded-offset-corrected locations for every prediction scale."""
    emb_folder = f"{main_dir}/{args.interim_path}/UNI_multiscale_patches"
    locations = {}
    for patch_size in patch_sizes:
        emb_data = np.load(f"{emb_folder}/emb_{patch_size}/{dataset}.npz")
        keep_flag = emb_data["flag"]
        locations[patch_size] = (
            emb_data["pos"][keep_flag, :]
            - [emb_data["pad"][0] // 2, emb_data["pad"][1] // 2]
        )
    return locations


## read json
with open(f"{main_dir}/{args.train_path}/{args.parameter_json}", 'r') as f:
    parameters = json.load(f)
# retrain = False
retrain_cellprop_only = parameters.get("retrain_cellprop_only", False)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_dir = f"{main_dir}/{args.train_path}/processed_pkl_test_highres"

gene_list_file = f"{main_dir}/{args.train_path}/{parameters.get('gene_list_file',"AURORA_interim/Gene_to_predict.csv")}"
gene_list = pd.read_csv(gene_list_file)['Genes'].values
gene_list = list(gene_list)
hvg_list = gene_list
bulk_norm_factor = pd.read_csv(f"{main_dir}/{args.train_path}/{parameters.get('gene_norm_factor_file','AURORA_interim/Gene_normalize_factor.csv')}",index_col=0).loc[gene_list,'0'].values
input_size = parameters.get('img_emb_dim', 200)  # Length of embedding
output_size = len(gene_list)  # Number of genes to predict
batch_size = parameters.get('batch_size', 4)
num_workers = parameters.get('num_workers', 4)
patch_sizes = parameters.get('patch_sizes', [56, 14])  # List of patch sizes
mse_weight = parameters.get('mse_weight', 10)
prop_weight = parameters.get('prop_weight', 0.1)
focal_gamma = parameters.get('focal_gamma', 0.5)
num_epochs = parameters.get('num_epochs', 600)
exp_epochs = parameters.get('exp_epochs', 400)
pixel_size = parameters.get('pixel_size', 0.5)
lr = parameters.get('lr', 1e-3)
lr_prop = parameters.get('lr_prop', 1e-3)
weight_decay = parameters.get('weight_decay', 1e-5)
stepLR_step_exp = parameters.get('stepLR_step_exp', 200)
stepLR_gamma_exp = parameters.get('stepLR_gamma_exp', 0.1)
stepLR_step_prop = parameters.get('stepLR_step_prop', 50)
stepLR_gamma_prop = parameters.get('stepLR_gamma_prop', 0.5)
bidirectional = parameters.get('bidirectional', False)
use_bulk = parameters.get('use_bulk', True)
pred_celltype = parameters.get('pred_celltype', True)
use_context_loss = parameters.get('use_context_loss', True)
superpatch_size = 1e8
superpatch_stride = 1e8
superpatch_minsize = 0
mse_criterion = nn.MSELoss()
cosine_similarity = nn.CosineSimilarity()
prop_criterion = masked_focal_kl_loss
test_batch_size = 50000

# Load the data for each set
test_samples_path = f"{main_dir}/{args.train_path}/{args.test_samples}"
celltype_path = f"{main_dir}/{args.train_path}/{parameters.get('celltype_file', "AURORA_interim/deconvolution/cell_types.csv")}"
# test_samples_path = f"{main_dir}/Lung/Lung_test_samples.csv"

## Generate training and testing samples
dataset_test = pd.read_csv(test_samples_path).iloc[:,0].values
print(f"Test samples: {dataset_test}")
if pred_celltype:
    major_cell_type = pd.read_csv(celltype_path,header=None).values.squeeze()
else:
    major_cell_type = []

load_fromfile_flag = False
if os.path.exists(dataset_dir):
    superpatch_exist = [x.replace(".pkl","") for x in os.listdir(dataset_dir) if x.endswith(".pkl")]
    if os.path.exists(f"{dataset_dir}/superpatch_meta.csv"):
        superpatch_meta = pd.read_csv(f"{dataset_dir}/superpatch_meta.csv")
        dataset_exist = superpatch_meta['dataset'].unique()
        if set(list(dataset_test)) <= set(dataset_exist) and os.path.exists(f"{dataset_dir}/superpatch_info.json"):
            with open(f"{dataset_dir}/superpatch_info.json", 'r') as f:
                superpatch_info = json.load(f)
            if superpatch_info['size'] == superpatch_size and superpatch_info['stride'] == superpatch_stride:
                load_fromfile_flag = True
else:
    os.makedirs(dataset_dir, exist_ok=True)

model = AURORA(input_size, output_size, bulk_size=len(hvg_list),
                celltype_num=len(major_cell_type), latent_size=256,
                device=device,bidirectional=bidirectional,
                use_bulk=use_bulk, pred_celltype=pred_celltype)
model.to(device)
state_dict = torch.load(f"{main_dir}/{args.train_path}/{args.model_pth}", weights_only=False, map_location=device)
model.load_state_dict(state_dict)
print("Starting evaluation")
model.eval()
os.makedirs(f"{main_dir}/{args.train_path}/test_pred_highres", exist_ok=True)
with torch.no_grad():
    if load_fromfile_flag:#set(list(dataset_train)) <= set(dataset_exist):
        print("Training samples loading from disk")
        test_data = load_data2(list(dataset_test), main_dir=main_dir, processed_data_dir=args.processed_data_path,
                                interim_dir=args.interim_path, train_dir=args.train_path,bulk_hvg=hvg_list, bulk_norm_factor=bulk_norm_factor,
                                plot_mask=args.plot, patch_sizes=patch_sizes, load_cell_type = pred_celltype,
                                from_file=True, file_dir=dataset_dir, num_workers=num_workers, pixel_size = pixel_size,
                                superpatch_size=superpatch_size, superpatch_stride=superpatch_stride, superpatch_minsize=superpatch_minsize)
    else:
        print("Training samples loading from sketch")
        test_data = load_data2(list(dataset_test), main_dir=main_dir, processed_data_dir=args.processed_data_path,
                                interim_dir=args.interim_path, train_dir=args.train_path,bulk_hvg=hvg_list, bulk_norm_factor=bulk_norm_factor,
                                plot_mask=args.plot, patch_sizes=patch_sizes, load_cell_type = pred_celltype,
                                from_file=False, file_dir=dataset_dir, num_workers=num_workers, pixel_size = pixel_size,
                                superpatch_size=superpatch_size, superpatch_stride=superpatch_stride, superpatch_minsize=superpatch_minsize)
    test_loader_tqdm = tqdm(
        np.arange(len(test_data)), desc=f"Evaluation", unit="batch", leave=False
    )
    loss = 0
    spot_preds = {}
    major_prop_preds = {}
    for index in test_loader_tqdm:
        dataset_name = str(dataset_test[index]).replace(".h5", "")
        patch_locations = load_patch_locations(dataset_name)
        emb, windows_mask, spot_weight, gene_expression, gene_mask, bulk, bulk_rank, father_lowres, father_patches, major_prop = test_data[index]
        for key in emb.keys():
            emb[key] = torch.tensor(emb[key], dtype=torch.float32)
        for key in windows_mask.keys():
            windows_mask[key] = torch.tensor(windows_mask[key], dtype=torch.float32)
        for key in spot_weight.keys():
            if isinstance(spot_weight[key], dict):
                # Compact sparse form {'indices', 'values'}.
                spot_weight[key] = {
                    'indices': torch.tensor(spot_weight[key]['indices'], dtype=torch.long),
                    'values': torch.tensor(spot_weight[key]['values'], dtype=torch.float32),
                }
            else:
                spot_weight[key] = torch.tensor(spot_weight[key], dtype=torch.float32)
        for key in father_patches.keys():
            father_patches[key] = torch.tensor(father_patches[key], dtype=torch.int)
        gene_expression = torch.tensor(gene_expression, dtype=torch.float32)
        gene_mask = torch.tensor(gene_mask)
        bulk = torch.tensor(bulk, dtype=torch.float32)
        bulk_rank = torch.tensor(bulk_rank, dtype=torch.float32).to(device)
        father_lowres = torch.tensor(father_lowres, dtype=torch.int32)
        major_prop = torch.tensor(major_prop, dtype=torch.float32)
        # Forward pass
        emb_hierrachy = torch.stack([emb[patch_sizes[0]][father_patches[patch_sizes[0]],:],
                                        emb[patch_sizes[1]]],dim=1)
        emb_hierrachy = emb_hierrachy[:,[1,0],:]  # Reverse the order of patch sizes for LSTM input
        output_tmp = []
        major_prop_tmp = []
        total_batch = emb_hierrachy.shape[0]//test_batch_size + 1
        for batch in tqdm(range(total_batch), desc=f"Processing {dataset_test[index]}", unit="batch", leave=False):
            batch_start = batch*test_batch_size
            batch_end = min((batch+1)*test_batch_size, emb_hierrachy.shape[0])
            emb_hierrachy_batch = emb_hierrachy[batch*test_batch_size:(batch+1)*test_batch_size,:,:]
            # latent_emb_use_batch = latent_emb_use[:, batch*test_batch_size:(batch+1)*test_batch_size,:]
            emb_hierrachy_batch = emb_hierrachy_batch.to(device)
            # latent_emb_use_batch = latent_emb_use_batch.to(device)
            output_batch, major_prop_pred_batch = model(emb_hierrachy_batch, bulk_rank, highres=True)
            output_tmp.append(output_batch.cpu())
            major_prop_tmp.append(major_prop_pred_batch.cpu())
        output = torch.cat(output_tmp, dim=0)
        major_prop_pred = torch.cat(major_prop_tmp, dim=0)
        output = output[:,[1,0],:]  # Reverse back the order of patch sizes for output
        major_prop_pred = major_prop_pred[:,[1,0],:]  # Reverse back the order of patch sizes for output
        # Collect the unique-father dedup mapping (reused for compact storage below)
        father_unique_all = {}
        inverse_index_all = {}
        for patch_size_index in range(len(patch_sizes)):
            patch_size = patch_sizes[patch_size_index]
            if patch_size != min(patch_sizes):
                #map back from LSTM output to original father patches
                father_unique, inverse_index = get_unique_indices(father_patches[patch_size],device='cpu')
                father_unique_all[patch_size] = father_unique
                inverse_index_all[patch_size] = inverse_index
                output_tmp = output[inverse_index,patch_size_index,:]
                #project patch expression to spot levels (sparse spot-weight projection)
                spot_pred = project_spot_weight_columns(spot_weight[patch_size], output_tmp, father_patches[patch_size], father_unique)
                if pred_celltype:
                    major_prop_tmp = major_prop_pred[inverse_index,patch_size_index,:]
                    major_prop_spot_pred = project_spot_weight_columns(spot_weight[patch_size], torch.exp(major_prop_tmp), father_patches[patch_size], father_unique)
                    major_prop_spot_pred = torch.log(major_prop_spot_pred+1e-10)
            else:
                spot_pred = project_spot_weight(spot_weight[patch_size], output[:,patch_size_index,:])
                if pred_celltype:
                    major_prop_spot_pred = project_spot_weight(spot_weight[patch_size], torch.exp(major_prop_pred[:,patch_size_index,:]))
                    major_prop_spot_pred = torch.log(major_prop_spot_pred+1e-10)
            spot_preds[patch_size] = spot_pred.numpy()
            major_prop_preds[patch_size] = major_prop_spot_pred.numpy()
        utils.save_pickle(spot_preds,f"{main_dir}/{args.train_path}/test_pred_highres/{dataset_test[index]}-spotpred.pickle")
        utils.save_pickle(major_prop_preds,f"{main_dir}/{args.train_path}/test_pred_highres/{dataset_test[index]}-majorprop.pickle")
        # Store with replicates removed for the coarse patch sizes
        # (patch_size > min(patch_size)): for those levels output/major_prop are
        # shared by all min-size patches under the same father patch, so we keep
        # only one row per unique father (same dedup as in the spot projection
        # above). The fine (min) level keeps all rows. The stored dicts are keyed
        # by patch_size; 'father_unique' gives the father patch per compact row.
        # To expand back to one row per min-size patch:
        #   row = np.searchsorted(father_unique[ps], father_patches[ps])
        #   full = compact[row]
        output_dedup = {}
        major_prop_dedup = {}
        patch_loc_dedup = {}
        for patch_size_index in range(len(patch_sizes)):
            patch_size = patch_sizes[patch_size_index]
            if patch_size != min(patch_sizes):
                ii = inverse_index_all[patch_size]
                father_unique = father_unique_all[patch_size]
                output_dedup[patch_size] = output[ii, patch_size_index, :].numpy().astype(np.float16)
                major_prop_dedup[patch_size] = major_prop_pred[ii, patch_size_index, :].numpy().astype(np.float16)
                patch_loc_dedup[patch_size] = patch_locations[patch_size][father_unique.numpy()]
            else:
                output_dedup[patch_size] = output[:, patch_size_index, :].numpy().astype(np.float16)
                major_prop_dedup[patch_size] = major_prop_pred[:, patch_size_index, :].numpy().astype(np.float16)
                patch_loc_dedup[patch_size] = patch_locations[patch_size]
        utils.save_pickle({'output': output_dedup, 'major_prop_pred': major_prop_dedup,
                   'patch_loc': patch_loc_dedup},
                          f"{main_dir}/{args.train_path}/test_pred_highres/{dataset_test[index]}-pred.pickle")