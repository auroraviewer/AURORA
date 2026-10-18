import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm 
from train_utils import load_data, get_unique_indices
from model import AURORA
import argparse
import json
import utils
parser = argparse.ArgumentParser(description='AURORA Test Script')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--interim_path', '-i', type=str, default=None, help='Interim data directory')
parser.add_argument('--train_path', '-t', type=str, default='AURORA_train', help='Directory to store training data')
parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
parser.add_argument('--plot', action='store_true', help='Plot intermediate results for checking')
parser.add_argument('--parameter_json', '-j', type=str, default='parameters.json', help='JSON file with parameters')
parser.add_argument('--sample_sheet', type=str, required=True, help='Training sample CSV file name')
parser.add_argument('--sample_name_col', type=int, default=0, help='Column index for sample names in sample sheet CSV')
parser.add_argument('--model_pth', type=str, default='AURORA_model_weights.pth', help='Path to the pretrained model file')
parser.add_argument('--barcode_dir', type=str, default='vanilla/UNI_emb', help='Directory for barcode embeddings')
args = parser.parse_args()

main_dir = args.project_path
if args.interim_path is None:
    args.interim_path = args.train_path

## read json
with open(f"{main_dir}/{args.train_path}/{args.parameter_json}", 'r') as f:
    parameters = json.load(f)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_dir = f"{main_dir}/{args.train_path}/processed_pkl"
if os.path.exists(dataset_dir):
    dataset_exist = [x.replace(".pkl","") for x in os.listdir(dataset_dir) if x.endswith(".pkl")]
else:
    dataset_exist = []

gene_list_file = f"{main_dir}/{args.train_path}/{parameters.get('gene_list_file','AURORA_interim/Gene_to_predict.csv')}"
gene_list = pd.read_csv(gene_list_file)['Genes'].values
gene_list = list(gene_list)
hvg_list = gene_list
bulk_norm_factor = pd.read_csv(f"{main_dir}/{args.train_path}/{parameters.get('gene_norm_factor_file','AURORA_interim/Gene_normalize_factor.csv')}",index_col=0).loc[gene_list,'0'].values
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
celltype_path = f"{main_dir}/{args.train_path}/{parameters.get('celltype_file', 'AURORA_interim/deconvolution/cell_types.csv')}"
# train.py builds its cell-type head with `len(major_cell_type)` where that list is
# empty when pred_celltype is false, so a checkpoint trained that way has a 0-wide
# decoder. Reading the CSV unconditionally here rebuilt a full-width head and
# load_state_dict failed with "copying a param with shape [0, 32] ... current model
# is [10, 32]".
# major_cell_type = pd.read_csv(celltype_path,header=None).values.squeeze()
if pred_celltype:
    major_cell_type = pd.read_csv(celltype_path,header=None).values.squeeze()
else:
    major_cell_type = []
test_samples_path = f"{main_dir}/{args.sample_sheet}"
dataset_test = pd.read_csv(test_samples_path).iloc[:, args.sample_name_col].values

print("Loading pretrained model")
model = AURORA(input_size, output_size, bulk_size=len(hvg_list),
                celltype_num=len(major_cell_type),
                device=device,bidirectional=bidirectional,
                use_bulk=use_bulk, pred_celltype=pred_celltype)
model.to(device)
state_dict = torch.load(f"{main_dir}/{args.train_path}/{args.model_pth}", weights_only=False, map_location=device)
model.load_state_dict(state_dict)
print("Starting evaluation")
model.eval()
os.makedirs(f"{main_dir}/{args.train_path}/train_pred_lowres", exist_ok=True)
with torch.no_grad():
    # running_loss = 0.0
    if set(list(dataset_test)) <= set(dataset_exist):
        test_data = load_data(list(dataset_test), gene_list, main_dir=main_dir, processed_data_dir=args.processed_data_path,
                            interim_dir=args.interim_path, train_dir=args.train_path,
                            bulk_hvg=hvg_list, bulk_norm_factor=bulk_norm_factor, 
                            plot_mask=args.plot, patch_sizes=patch_sizes, cell_type_list=major_cell_type, load_cell_type = False,
                            from_file=True, file_dir=dataset_dir, num_workers=num_workers, pixel_size = pixel_size,
                            barcode_dir = f"{main_dir}/{args.barcode_dir}")
    else:
        test_data = load_data(list(dataset_test), gene_list, main_dir=main_dir, processed_data_dir=args.processed_data_path,
                            interim_dir=args.interim_path, train_dir=args.train_path,
                            bulk_hvg=hvg_list, bulk_norm_factor=bulk_norm_factor, 
                            plot_mask=args.plot, patch_sizes=patch_sizes, cell_type_list=major_cell_type,load_cell_type = False,
                            from_file=False, file_dir=dataset_dir, num_workers=num_workers, pixel_size = pixel_size,
                            barcode_dir = f"{main_dir}/{args.barcode_dir}")
    test_loader_tqdm = tqdm(
        np.arange(len(test_data)), desc=f"Evaluation", unit="batch", leave=False
    )
    loss = 0
    spot_preds = {}
    major_prop_preds = {}
    # global_bulk_pred = torch.zeros((len(test_data), len(gene_list))).to(device)
    # global_bulk_truth = torch.zeros((len(test_data), len(gene_list))).to(device)
    for index in test_loader_tqdm:
        # mse_loss_recorder = {}
        # local_bulk_loss_recorder = {}
        # major_prop_loss_recorder = {}
        # #initialize recorders
        # for patch_size in patch_sizes:
        #     mse_loss_recorder[patch_size] = 0
        #     local_bulk_loss_recorder[patch_size] = 0
        #     major_prop_loss_recorder[patch_size] = 0
        emb, windows_mask, spot_weight, gene_expression, gene_mask, bulk, bulk_rank, max_exp, father_patches, major_prop = test_data[index]
        for key in emb.keys():
            emb[key] = torch.tensor(emb[key], dtype=torch.float32).to(device)
        for key in windows_mask.keys():
            windows_mask[key] = torch.tensor(windows_mask[key], dtype=torch.float32).to(device)
        for key in spot_weight.keys():
            spot_weight[key] = torch.tensor(spot_weight[key], dtype=torch.float32).to(device)
        for key in father_patches.keys():
            father_patches[key] = torch.tensor(father_patches[key], dtype=torch.int).to(device)
        gene_expression = torch.tensor(gene_expression, dtype=torch.float32).to(device)
        gene_mask = torch.tensor(gene_mask).to(device)
        bulk = torch.tensor(bulk, dtype=torch.float32).to(device)
        bulk_rank = torch.tensor(bulk_rank, dtype=torch.float32).to(device)
        max_exp = torch.tensor(max_exp, dtype=torch.float32).to(device)
        major_prop = torch.tensor(major_prop, dtype=torch.float32).to(device)
        # Forward pass
        emb_hierrachy = torch.stack([emb[patch_sizes[0]][father_patches[patch_sizes[0]],:],
                                                emb[patch_sizes[1]][father_patches[patch_sizes[1]],:],
                                                emb[patch_sizes[2]]],dim=1)
        output, major_prop_pred, latent_emb = model(emb_hierrachy, bulk_rank, return_latent_emb=True)
        output = output.cpu().numpy()
        # The model returns (output, None) when pred_celltype is false, so there is no
        # proportion head to save. Keep the key present (as None) so downstream readers
        # see the same shape of record either way.
        # major_prop_pred = major_prop_pred.cpu().numpy()
        major_prop_pred = (
            major_prop_pred.cpu().numpy() if major_prop_pred is not None else None
        )
        latent_emb = [s.cpu().numpy() for s in latent_emb]
        utils.save_pickle({'output':output,'major_prop_pred':major_prop_pred,'latent_emb':latent_emb},f"{main_dir}/{args.train_path}/train_pred_lowres/{dataset_test[index]}-pred.pickle")
