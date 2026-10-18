### This script is only for developers
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm 
from torch.utils.data import DataLoader
from train_utils import load_data, get_unique_indices
from datasets import HistologyGeneExpressionDataset, my_collate_adjmat
from model import GraphLSTM
import argparse
import json
import utils
parser = argparse.ArgumentParser(description='GraphLSTM Test Script')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--interim_path', '-i', type=str, default='AURORA_interim', help='Interim data directory')
parser.add_argument('--train_path', '-t', type=str, default='AURORA_train', help='Directory to store training data')
parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
parser.add_argument('--plot', action='store_true', help='Plot intermediate results for checking')
parser.add_argument('--parameter_json', '-j', type=str, default='parameters.json', help='JSON file with parameters')
parser.add_argument('--test_samples', type=str, required=True, help='Test sample CSV file name')
parser.add_argument('--model_pth', type=str, default='GraphLSTM_model_weights.pth', help='Path to the pretrained model file')
parser.add_argument('--barcode_dir', type=str, default=None, help='Directory containing barcode embeddings')
args = parser.parse_args()

main_dir = args.project_path
## read json
with open(f"{main_dir}/{args.parameter_json}", 'r') as f:
    parameters = json.load(f)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_dir = f"{main_dir}/{args.train_path}/processed_pkl_GraphLSTM"
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
pred_celltype = parameters.get('pred_celltype', True)

# Load the data for each set
celltype_path = f"{main_dir}/{parameters.get('celltype_file', "AURORA_interim/deconvolution/cell_types.csv")}"
if pred_celltype:
    major_cell_type = pd.read_csv(celltype_path,header=None).values.squeeze()
else:
    major_cell_type = []
test_samples_path = f"{main_dir}/{args.test_samples}"
dataset_test = pd.read_csv(test_samples_path)['sample'].values
dataset_test = [x.replace(".h5","") for x in dataset_test]

print("Loading test data...")
if set(list(dataset_test)) <= set(dataset_exist):
    test_data = load_data(list(dataset_test), gene_list, main_dir=main_dir, processed_data_dir=args.processed_data_path,
                            interim_dir=args.interim_path, train_dir=args.train_path,
                            bulk_hvg=hvg_list, bulk_norm_factor=bulk_norm_factor, 
                            plot_mask=args.plot, patch_sizes=patch_sizes, cell_type_list=major_cell_type, load_cell_type=pred_celltype,
                            from_file=True, file_dir=dataset_dir, num_workers=num_workers, pixel_size=pixel_size,
                            load_graph=True, barcode_dir=args.barcode_dir,smooth_cnts=True)
else:
    test_data = load_data(list(dataset_test), gene_list, main_dir=main_dir, processed_data_dir=args.processed_data_path,
                            interim_dir=args.interim_path, train_dir=args.train_path,
                            bulk_hvg=hvg_list, bulk_norm_factor=bulk_norm_factor, 
                            plot_mask=args.plot, patch_sizes=patch_sizes, cell_type_list=major_cell_type, load_cell_type=pred_celltype,
                            from_file=False, file_dir=dataset_dir, num_workers=num_workers, pixel_size=pixel_size,
                            load_graph=True, barcode_dir=args.barcode_dir,smooth_cnts=True)

test_dataset = HistologyGeneExpressionDataset(test_data, device=device, highres=False, adj_mat=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=my_collate_adjmat)

print("Loading pretrained model")
model = GraphLSTM(input_size, output_size, bulk_size=len(hvg_list),
                   celltype_num=len(major_cell_type),
                   device=device, pred_celltype=pred_celltype)
model.to(device)
state_dict = torch.load(f"{main_dir}/{args.train_path}/{args.model_pth}", weights_only=False, map_location=device)
model.load_state_dict(state_dict)
print("Starting evaluation")
model.eval()
os.makedirs(f"{main_dir}/{args.train_path}/GraphLSTM_pred", exist_ok=True)
with torch.no_grad():
    sample_idx = 0
    for emb, windows_mask, spot_weight, gene_expression, gene_mask, bulk, bulk_rank, max_exp, father_patches, major_prop, edge_index_list in tqdm(test_loader, desc="Evaluation", unit="sample"):
        index = 0  # batch_size = 1
        spot_preds = {}
        major_prop_preds = {}

        # Build hierarchical embedding stack (same as training)
        emb_hierrachy = torch.stack([emb[index][patch_sizes[0]][father_patches[index][patch_sizes[0]],:],
                                     emb[index][patch_sizes[1]][father_patches[index][patch_sizes[1]],:],
                                     emb[index][patch_sizes[2]]], dim=1).to(device)

        # Build edge_index list for each hierarchy level
        edge_index_list_model = [edge_index_list[index][patch_sizes[0]].to(device),
                                 edge_index_list[index][patch_sizes[1]].to(device),
                                 edge_index_list[index][patch_sizes[2]].to(device)]

        # Forward pass through GraphLSTM
        output, major_prop_pred = model(emb_hierrachy, bulk_rank[index].to(device), edge_index_list_model)

        # Project patch-level predictions to spot level (same hierarchy as training)
        for patch_size_index in range(len(patch_sizes)):
            patch_size = patch_sizes[patch_size_index]
            if patch_size != min(patch_sizes):
                # Map back from LSTM output to original father patches
                father_unique, inverse_index = get_unique_indices(father_patches[index][patch_size], device='cpu')
                output_tmp = output[inverse_index.to(device), patch_size_index, :]
                spot_pred = spot_weight[index][patch_size][:, father_unique].to(device) @ output_tmp
                if pred_celltype:
                    major_prop_tmp = major_prop_pred[inverse_index, patch_size_index, :]
                    major_prop_spot_pred = spot_weight[index][patch_size][:, father_unique].to(device) @ torch.exp(major_prop_tmp)
                    major_prop_spot_pred = torch.log(major_prop_spot_pred + 1e-10)
            else:
                spot_pred = spot_weight[index][patch_size].to(device) @ output[:, patch_size_index, :]
                if pred_celltype:
                    major_prop_spot_pred = spot_weight[index][patch_size].to(device) @ torch.exp(major_prop_pred[:, patch_size_index, :])
                    major_prop_spot_pred = torch.log(major_prop_spot_pred + 1e-10)

            spot_preds[patch_size] = spot_pred.cpu().numpy()
            if pred_celltype:
                major_prop_preds[patch_size] = major_prop_spot_pred.cpu().numpy()

        utils.save_pickle(spot_preds, f"{main_dir}/{args.train_path}/GraphLSTM_pred/{dataset_test[sample_idx]}-spotpred.pickle")
        utils.save_pickle(major_prop_preds, f"{main_dir}/{args.train_path}/GraphLSTM_pred/{dataset_test[sample_idx]}-majorprop.pickle")
        utils.save_pickle({'output': output.cpu().numpy(), 'major_prop_pred': major_prop_pred.cpu().numpy() if major_prop_pred is not None else None},
                          f"{main_dir}/{args.train_path}/GraphLSTM_pred/{dataset_test[sample_idx]}-pred.pickle")
        sample_idx += 1
