### This script is only for developers
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
parser.add_argument('--interim_path', '-i', type=str, default='AURORA_interim', help='Interim data directory')
parser.add_argument('--train_path', '-t', type=str, default='AURORA_train', help='Directory to store training data')
# parser.add_argument('--patch_sizes', nargs='+', type=int, default=[3584, 896, 224], help='List of patch sizes')
# parser.add_argument('--metadata_path', '-m', type=str, default='Sample_metadata.csv', help='Sample metadata CSV file name')
parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
parser.add_argument('--plot', action='store_true', help='Plot intermediate results for checking')
# parser.add_argument('--log', action='store_true', help='Enable logging to TensorBoard')
parser.add_argument('--parameter_json', '-j', type=str, default='parameters.json', help='JSON file with parameters')
parser.add_argument('--test_samples', type=str, required=True, help='Test sample CSV file name')
parser.add_argument('--model_pth', type=str, default='AURORA_model_weights.pth', help='Path to the pretrained model file')
args = parser.parse_args()

main_dir = args.project_path
## read json
with open(f"{main_dir}/{args.parameter_json}", 'r') as f:
    parameters = json.load(f)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_dir = f"{main_dir}/{args.train_path}/processed_pkl"
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
test_samples_path = f"{main_dir}/{args.test_samples}"
dataset_test = pd.read_csv(test_samples_path)['sample'].values

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
os.makedirs(f"{main_dir}/{args.train_path}/test_pred", exist_ok=True)
with torch.no_grad():
    # running_loss = 0.0
    if set(list(dataset_test)) <= set(dataset_exist):
        test_data = load_data(list(dataset_test), gene_list, main_dir=main_dir, processed_data_dir=args.processed_data_path,
                            interim_dir=args.interim_path, train_dir=args.train_path,
                            bulk_hvg=hvg_list, bulk_norm_factor=bulk_norm_factor, 
                            plot_mask=args.plot, patch_sizes=patch_sizes, cell_type_list=major_cell_type,
                            from_file=True, file_dir=dataset_dir, num_workers=num_workers, pixel_size = pixel_size)
    else:
        test_data = load_data(list(dataset_test), gene_list, main_dir=main_dir, processed_data_dir=args.processed_data_path,
                            interim_dir=args.interim_path, train_dir=args.train_path,
                            bulk_hvg=hvg_list, bulk_norm_factor=bulk_norm_factor, 
                            plot_mask=args.plot, patch_sizes=patch_sizes, cell_type_list=major_cell_type,
                            from_file=True, file_dir=dataset_dir, num_workers=num_workers, pixel_size = pixel_size)
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
        output, major_prop_pred = model(emb_hierrachy, bulk_rank)
        for patch_size_index in range(len(patch_sizes)):
            patch_size = patch_sizes[patch_size_index]
            if patch_size != min(patch_sizes):
                #map back from LSTM output to original father patches
                father_unique, inverse_index = get_unique_indices(father_patches[patch_size],device=device)
                output_tmp = output[inverse_index,patch_size_index,:]
                #project patch expression to spot levels
                spot_pred = spot_weight[patch_size][:,father_unique] @ output_tmp
                major_prop_tmp = major_prop_pred[inverse_index,patch_size_index,:]
                major_prop_spot_pred = spot_weight[patch_size][:,father_unique] @ torch.exp(major_prop_tmp)
                major_prop_spot_pred = torch.log(major_prop_spot_pred)
            else:
                spot_pred = spot_weight[patch_size] @ output[:,patch_size_index,:]
                major_prop_spot_pred = spot_weight[patch_size] @ torch.exp(major_prop_pred[:,patch_size_index,:])
                major_prop_spot_pred = torch.log(major_prop_spot_pred)
            #
            spot_preds[patch_size] = spot_pred.cpu().numpy()
            major_prop_preds[patch_size] = major_prop_spot_pred.cpu().numpy()
            # mse_loss = mse_criterion(spot_pred[:,gene_mask], gene_expression[:,gene_mask])
            # mse_loss_recorder[patch_size] += mse_loss.item()
            # spot_use = major_prop.sum(axis=1) > 0
            # major_prop_loss = prop_criterion(major_prop_spot_pred[spot_use,:], major_prop[spot_use,:])
            # major_prop_loss_recorder[patch_size] += major_prop_loss.item()
            # #
            # #
            # # Compute local/global bulk loss
            # if patch_size != max(patch_sizes):
            #     local_bulk_pred = windows_mask[patch_size] @ spot_pred
            #     local_bulk_truth = windows_mask[patch_size] @ gene_expression
            #     local_bulk_loss = 1 - cosine_similarity(local_bulk_pred[:,gene_mask], local_bulk_truth[:,gene_mask])
            #     local_bulk_loss = local_bulk_loss.mean()
            #     loss += mse_loss * mse_weight + local_bulk_loss
            #     local_bulk_loss_recorder[patch_size] += local_bulk_loss.item()
            # else:
            #     # Store global bulk prediction and truth
            #     #TODO Add global loss at all levels?
            #     #TODO change get_st only keep max for genes in gene_list
            #     gene_indices = [hvg_list.index(gene) for gene in gene_list if gene in hvg_list]
            #     global_bulk_pred[index,:] = spot_pred.sum(dim=0) * max_exp[gene_indices]
            #     global_bulk_truth[index,:] = bulk[gene_indices]
        utils.save_pickle(spot_preds,f"{main_dir}/{args.train_path}/test_pred/{dataset_test[index]}-spotpred.pickle")
        utils.save_pickle(major_prop_preds,f"{main_dir}/{args.train_path}/test_pred/{dataset_test[index]}-majorprop.pickle")
        # # Update tqdm description with the running loss
        # test_loader_tqdm.set_postfix(
        #     loss=loss.item(),
        #     rmse_level1=mse_loss_recorder[patch_sizes[0]],
        #     rmse_level2=mse_loss_recorder[patch_sizes[1]],
        #     rmse_level3=mse_loss_recorder[patch_sizes[2]],
        #     local_bulk_loss_level1=local_bulk_loss_recorder[patch_sizes[1]],
        #     local_bulk_loss_level2=local_bulk_loss_recorder[patch_sizes[2]],
        #     major_prop_level1=major_prop_loss_recorder[patch_sizes[0]],
        #     major_prop_level2=major_prop_loss_recorder[patch_sizes[1]],
        #     major_prop_level3=major_prop_loss_recorder[patch_sizes[2]],
        # )
        output = output.cpu().numpy()
        major_prop_pred = major_prop_pred.cpu().numpy()
        # print(f'{dataset_test[index]}: mse -- level1 {mse_loss_recorder[patch_sizes[0]]}; level2 {mse_loss_recorder[patch_sizes[1]]}; level3 {mse_loss_recorder[patch_sizes[2]]}.')
        # print(f'{dataset_test[index]}: major prop -- level1 {major_prop_loss_recorder[patch_sizes[0]]}; level2 {major_prop_loss_recorder[patch_sizes[1]]};level3 {major_prop_loss_recorder[patch_sizes[2]]}.')
        utils.save_pickle({'output':output,'major_prop_pred':major_prop_pred},f"{main_dir}/{args.train_path}/test_pred/{dataset_test[index]}-pred.pickle")
    # global_bulk_loss = 1 - cosine_similarity(global_bulk_pred, global_bulk_truth)
    # global_bulk_loss = global_bulk_loss.mean()
    # print(f"Global bulk loss: {global_bulk_loss}")
    # loss /= len(test_data)
    # for patch_size in patch_sizes:
    #     mse_loss_recorder[patch_size] /= len(test_data)
    #     local_bulk_loss_recorder[patch_size] /= len(test_data)
    # loss += global_bulk_loss
    # print(f"Test loss: {loss / len(test_data)}")
