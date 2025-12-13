import pandas as pd
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from train_utils import load_data, masked_focal_kl_loss, get_unique_indices
from datasets import HistologyGeneExpressionDataset, my_collate
from model import AURORA
import argparse
import json
parser = argparse.ArgumentParser(description='Rescale H&E images to target pixel size and adjust ST spot locations accordingly.')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--interim_path', '-i', type=str, default='AURORA_interim', help='Interim data directory')
parser.add_argument('--train_path', '-t', type=str, default='AURORA_train', help='Directory to store training data')
# parser.add_argument('--patch_sizes', nargs='+', type=int, default=[3584, 896, 224], help='List of patch sizes')
# parser.add_argument('--metadata_path', '-m', type=str, default='Sample_metadata.csv', help='Sample metadata CSV file name')
parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
parser.add_argument('--plot', action='store_true', help='Plot intermediate results for checking')
parser.add_argument('--log', action='store_true', help='Enable logging to TensorBoard')
parser.add_argument('--parameter_json', '-j', type=str, default='parameters.json', help='JSON file with parameters')
parser.add_argument('--train_samples', type=str, default='Sample_metadata.csv', help='CSV file with training sample metadata')
parser.add_argument('--train_sample_col', type=int, default=0, help='Column index for training sample IDs in the metadata CSV')
args = parser.parse_args()

main_dir = args.project_path
## read json
with open(f"{main_dir}/{args.parameter_json}", 'r') as f:
    parameters = json.load(f)
# retrain = False
retrain_cellprop_only = parameters.get("retrain_cellprop_only", False)
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
batch_size = parameters.get('batch_size', 4)
num_workers = parameters.get('num_workers', 4)
patch_sizes = parameters.get('patch_sizes', [3584, 896, 224])  # List of patch sizes
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
mse_criterion = nn.MSELoss()
cosine_similarity = nn.CosineSimilarity()
prop_criterion = masked_focal_kl_loss

# Load the data for each set
train_samples_path = f"{main_dir}/{args.train_samples}"
celltype_path = f"{main_dir}/{parameters.get('celltype_file', "AURORA_interim/deconvolution/cell_types.csv")}"
# test_samples_path = f"{main_dir}/Lung/Lung_test_samples.csv"

## Generate training and testing samples
sample_col = args.train_sample_col
dataset_train = pd.read_csv(train_samples_path).iloc[:,sample_col].values
major_cell_type = pd.read_csv(celltype_path,header=None).values.squeeze()
if set(list(dataset_train)) <= set(dataset_exist):
    print("Training samples loading from disk")
    train_data = load_data(list(dataset_train), gene_list, main_dir=main_dir, processed_data_dir=args.processed_data_path,
                            interim_dir=args.interim_path, train_dir=args.train_path,
                            bulk_hvg=hvg_list, bulk_norm_factor=bulk_norm_factor, 
                            plot_mask=args.plot, patch_sizes=patch_sizes, cell_type_list=major_cell_type,
                            from_file=True, file_dir=dataset_dir, num_workers=num_workers, pixel_size = pixel_size)
else:
    print("Training samples loading from scratch")
    train_data = load_data(list(dataset_train), gene_list, main_dir=main_dir, processed_data_dir=args.processed_data_path,
                            interim_dir=args.interim_path, train_dir=args.train_path,
                            bulk_hvg=hvg_list, bulk_norm_factor=bulk_norm_factor, 
                            plot_mask=args.plot, patch_sizes=patch_sizes, cell_type_list=major_cell_type,
                            from_file=False, file_dir=dataset_dir, num_workers=num_workers, pixel_size = pixel_size)
train_dataset = HistologyGeneExpressionDataset(train_data,device=device)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
model = AURORA(input_size, output_size, bulk_size=len(hvg_list),
                celltype_num=len(major_cell_type),
                device=device,bidirectional=bidirectional,
                use_bulk=use_bulk, pred_celltype=pred_celltype)
# Training loop
# best_val_loss = float("inf")
# best_model = None
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer) #CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/4)
batch_num = len(train_dataset)// batch_size + 1
# print(batch_num)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepLR_step_exp * batch_num, gamma=stepLR_gamma_exp)
# default `log_dir` is "runs" - we'll be more specific here
if args.log:
    writer = SummaryWriter(f"{main_dir}/{args.train_path}/train_log")

os.makedirs(f"{main_dir}/{args.train_path}/checkpoints", exist_ok=True)
if retrain_cellprop_only:
    model = torch.load(f"{main_dir}/{args.train_path}/checkpoints/AURORA_{exp_epochs}.pth")
    start_epoch = exp_epochs
else:
    start_epoch = 0

for para in model.celltype_decoder.parameters():
    para.requires_grad = False

for epoch in range(start_epoch,num_epochs):
    if epoch % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}")
        # print(f"Learning rate: {scheduler.get_last_lr()}")
    # First train gene expression modules, then fix them and train cell type proportion modules.
    if epoch == exp_epochs:
        for para in model.parameters():
            para.requires_grad = False
        for para in model.celltype_decoder.parameters():
            para.requires_grad = True
        optimizer = optim.Adam(model.celltype_decoder.parameters(), lr=lr_prop, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepLR_step_prop * batch_num, gamma=stepLR_gamma_prop)
    if epoch > exp_epochs:
        if epoch % 50 == 0:
            focal_gamma += 0.5
    # print(f"Learning rate (Epoch {epoch}): {scheduler.get_last_lr()}")
    model.train()
    running_loss = 0.0
    running_mse_loss = {}
    running_local_bulk_loss = {}
    running_major_prop_loss = {}
    for patch_size in patch_sizes:
        running_mse_loss[patch_size] = 0.0
        running_local_bulk_loss[patch_size] = 0.0
        running_major_prop_loss[patch_size] = 0.0
    running_global_bulk_loss = 0.0
    running_global_prop_loss = 0.0
    # Initialize tqdm progress bar for training
    train_loader_tqdm = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False
    )
    for emb, windows_mask, spot_weight, gene_expression, gene_mask, bulk, bulk_rank, max_exp, father_patches, major_prop in train_loader_tqdm:
        # Zero the parameter gradients
        optimizer.zero_grad()
        loss = 0
        mse_loss_recorder = {}
        local_bulk_loss_recorder = {}
        major_prop_recorder = {}
        local_prop_recorder = {}
        #initialize recorders
        for patch_size in patch_sizes:
            mse_loss_recorder[patch_size] = 0
            local_bulk_loss_recorder[patch_size] = 0
            major_prop_recorder[patch_size] = 0
            local_prop_recorder[patch_size] = 0

        global_bulk_pred = torch.zeros((len(emb), len(gene_list))).to(device)
        global_bulk_truth = torch.zeros((len(emb), len(gene_list))).to(device)
        global_prop_pred = torch.zeros((len(emb), len(major_cell_type))).to(device)
        global_prop_truth = torch.zeros((len(emb), len(major_cell_type))).to(device)
        for index in range(len(emb)):
            emb_hierrachy = torch.stack([emb[index][patch_sizes[0]][father_patches[index][patch_sizes[0]],:],
                                            emb[index][patch_sizes[1]][father_patches[index][patch_sizes[1]],:],
                                            emb[index][patch_sizes[2]]],dim=1)
            output, major_prop_pred = model(emb_hierrachy, bulk_rank[index])
            for patch_size_index in range(len(patch_sizes)):
                patch_size = patch_sizes[patch_size_index]
                if patch_size != min(patch_sizes):
                    #map back from LSTM output to original father patches
                    father_unique, inverse_index = get_unique_indices(father_patches[index][patch_size],device=device)
                    output_tmp = output[inverse_index,patch_size_index,:]
                    #project patch expression to spot levels
                    spot_pred = spot_weight[index][patch_size][:,father_unique] @ output_tmp
                    if pred_celltype:
                        major_prop_tmp = major_prop_pred[inverse_index,patch_size_index,:]
                        major_prop_spot_pred = spot_weight[index][patch_size][:,father_unique] @ torch.exp(major_prop_tmp)
                        major_prop_spot_pred = torch.log(major_prop_spot_pred+1e-10)
                else:
                    spot_pred = spot_weight[index][patch_size] @ output[:,patch_size_index,:]
                    if pred_celltype:
                        major_prop_spot_pred = spot_weight[index][patch_size] @ torch.exp(major_prop_pred[:,patch_size_index,:])
                        major_prop_spot_pred = torch.log(major_prop_spot_pred+1e-10)
                #
                mse_loss = mse_criterion(spot_pred[:,gene_mask[index]], gene_expression[index][:,gene_mask[index]])
                mse_loss_recorder[patch_size] += mse_loss.item()
                if pred_celltype:
                    spot_use = major_prop[index].sum(axis=1) > 0
                    major_prop_loss = prop_criterion(major_prop_spot_pred[spot_use,:], major_prop[index][spot_use,:])
                    major_prop_recorder[patch_size] += major_prop_loss.item()
                #
                # Compute local/global bulk loss
                if patch_size != max(patch_sizes):
                    local_bulk_pred = windows_mask[index][patch_size] @ spot_pred
                    local_bulk_truth = windows_mask[index][patch_size] @ gene_expression[index]
                    # window_use = windows_mask[index][patch_size].sum(axis=1) > 8
                    # local_bulk_loss = 1 - cosine_similarity(local_bulk_pred[window_use,:][:,gene_mask[index]], local_bulk_truth[window_use,:][:,gene_mask[index]])
                    local_bulk_loss = 1 - cosine_similarity(local_bulk_pred[:,gene_mask[index]], local_bulk_truth[:,gene_mask[index]])
                    local_bulk_loss = local_bulk_loss.mean()

                    if pred_celltype:
                        windows_mask_normalized = windows_mask[index][patch_size] / windows_mask[index][patch_size].sum(axis=1, keepdims=True)
                        local_prop_pred = windows_mask_normalized @ torch.exp(major_prop_spot_pred)
                        local_prop_pred = torch.log(local_prop_pred + 1e-10)
                        local_prop_truth = windows_mask_normalized @ major_prop[index]
                        windows_use = local_prop_truth.sum(axis=1) > 0
                        # local_prop_loss = prop_criterion(local_prop_pred[windows_use,:], local_prop_truth[windows_use,:])
                        local_prop_loss = prop_criterion(local_prop_pred[windows_use,:], local_prop_truth[windows_use,:], gamma=focal_gamma)
                    
                    # if epoch == 401 and index == 0:
                    #     print(f"local_prop_pred: {local_prop_pred[windows_use,:]}")
                    #     print(f"local_prop_truth: {local_prop_truth[windows_use,:]}")
                    local_bulk_loss_recorder[patch_size] += local_bulk_loss.item()
                    if pred_celltype:
                        local_prop_recorder[patch_size] += local_prop_loss.item()

                    # ## Add entropy loss
                    # probs = local_prop_pred[windows_use,:].exp()
                    # entropy = torch.sum(probs * local_prop_pred[windows_use,:], dim=1).mean()
                    if use_context_loss:
                        if epoch < exp_epochs:
                            loss += mse_loss * mse_weight + local_bulk_loss
                        else:
                            loss += mse_loss * mse_weight + local_bulk_loss + (major_prop_loss+local_prop_loss*0.1) * prop_weight
                    else:
                        if epoch < exp_epochs:
                            loss += mse_loss
                        else:
                            loss += mse_loss + major_prop_loss * prop_weight
                else:
                    # Store global bulk prediction and truth
                    #TODO Add global loss at all levels?
                    #TODO change get_st only keep max for genes in gene_list
                    gene_indices = [hvg_list.index(gene) for gene in gene_list if gene in hvg_list]
                    global_bulk_pred[index,:] = spot_pred.sum(dim=0) * max_exp[index][gene_indices]
                    global_bulk_truth[index,:] = bulk[index][gene_indices]
                    if pred_celltype:
                        global_prop_truth[index,:] = major_prop[index].mean(axis=0)
                        global_prop_pred[index,:] = torch.log(torch.exp(major_prop_spot_pred).mean(dim=0)+1e-10)

        global_bulk_loss = 1 - cosine_similarity(global_bulk_pred, global_bulk_truth)
        global_bulk_loss = global_bulk_loss.mean()
        if pred_celltype:
            # global_prop_loss = prop_criterion(global_prop_pred, global_prop_truth)
            global_prop_loss = prop_criterion(global_prop_pred, global_prop_truth,gamma=focal_gamma)
        ## Add entropy loss
        # probs = global_prop_pred.exp()
        # entropy = torch.sum(probs * global_prop_pred, dim=1).mean()

        loss /= len(emb)
        for patch_size in patch_sizes:
            mse_loss_recorder[patch_size] /= len(emb)
            local_bulk_loss_recorder[patch_size] /= len(emb)
            if pred_celltype:
                major_prop_recorder[patch_size] /= len(emb)
                local_prop_recorder[patch_size] /= len(emb)

        if use_context_loss:
            loss += global_bulk_loss
            if epoch >= exp_epochs:
                loss += global_prop_loss * 0.1 * prop_weight #+ entropy * 0.01 * prop_weight
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Accumulate loss
        running_loss += loss.item()
        for patch_size in patch_sizes:
            running_mse_loss[patch_size] += mse_loss_recorder[patch_size]
            running_local_bulk_loss[patch_size] += local_bulk_loss_recorder[patch_size]
            if pred_celltype:
                running_major_prop_loss[patch_size] += major_prop_recorder[patch_size]
                running_major_prop_loss[patch_size] += local_prop_recorder[patch_size]
        running_global_bulk_loss += global_bulk_loss.item()
        if pred_celltype:
            running_global_prop_loss += global_prop_loss.item()
        # Update tqdm description with the running loss
        train_loader_tqdm.set_postfix(
            loss=loss.item()
        )

    # ...log the running loss
    writer.add_scalar('training loss',
                    running_loss / len(train_loader),
                    epoch)
    writer.add_scalar(f'training rmse loss ({patch_sizes[0]})',
                    running_mse_loss[patch_sizes[0]] / len(train_loader),
                    epoch)
    writer.add_scalar(f'training rmse loss ({patch_sizes[1]})',
                    running_mse_loss[patch_sizes[1]] / len(train_loader),
                    epoch)
    writer.add_scalar(f'training rmse loss ({patch_sizes[2]})',
                    running_mse_loss[patch_sizes[2]] / len(train_loader),
                    epoch)
    writer.add_scalar(f'training local bulk loss ({patch_sizes[1]})',
                    running_local_bulk_loss[patch_sizes[1]] / len(train_loader),
                    epoch)
    writer.add_scalar(f'training local bulk loss ({patch_sizes[2]})',
                    running_local_bulk_loss[patch_sizes[2]] / len(train_loader),
                    epoch)
    writer.add_scalar('training global bulk loss',
                    running_global_bulk_loss / len(train_loader),
                    epoch)
    if pred_celltype:
        writer.add_scalar(f'training major prop loss ({patch_sizes[0]})',
                        running_major_prop_loss[patch_sizes[0]] / len(train_loader),
                        epoch)
        writer.add_scalar(f'training major prop loss ({patch_sizes[1]})',
                        running_major_prop_loss[patch_sizes[1]] / len(train_loader),
                        epoch)
        writer.add_scalar(f'training major prop loss ({patch_sizes[2]})',
                        running_major_prop_loss[patch_sizes[2]] / len(train_loader),
                        epoch)
        writer.add_scalar(f'training local prop loss ({patch_sizes[1]})',
                        running_major_prop_loss[patch_sizes[1]] / len(train_loader),
                        epoch)
        writer.add_scalar(f'training local prop loss ({patch_sizes[2]})',
                        running_major_prop_loss[patch_sizes[2]] / len(train_loader),
                        epoch)
        writer.add_scalar('training global prop loss',
                        running_global_prop_loss / len(train_loader),
                        epoch)
    
    if epoch % 100 == 0:
        filename = f"{main_dir}/{args.train_path}/checkpoints/AURORA_{epoch}.pth"
        torch.save(model, filename)

print("Training complete")
writer.close()

# filename = f"{main_dir}/{args.train_path}/AURORA.pth"
# torch.save(model, filename)
state_dict = model.state_dict()
torch.save(state_dict, f"{main_dir}/{args.train_path}/AURORA_model_weights.pth")
