import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torch
class PatchDataset(Dataset):
    def __init__(self, image, mask, transform, loc, spot_radius_pixel, patch_size=224, stride=224, filter_prop = 0.2):
        self.stride = stride
        if spot_radius_pixel is not None:
            self.spot_radius_pixel = int(spot_radius_pixel)
        else:
            self.spot_radius_pixel = None
        # Padding the image and mask to make sure that all pixels are covered by some superpixels
        self.row_pad = self.stride - (image.shape[0] % self.stride)
        self.col_pad = self.stride - (image.shape[1] % self.stride)
        self.loc = loc
        image = np.pad(image.copy(), ((self.row_pad//2, self.row_pad-self.row_pad//2), (self.col_pad//2, self.col_pad-self.col_pad//2), (0, 0)), mode='constant', constant_values=255)
        mask = np.pad(mask.copy(), ((self.row_pad//2, self.row_pad-self.row_pad//2), (self.col_pad//2, self.col_pad-self.col_pad//2)), mode='constant', constant_values=0)
        # print(f"    Row padding: {self.row_pad}; Column padding: {self.col_pad}; Padded image shape: {image.shape}")
        self.image = image
        self.image[mask == 0] = [255, 255, 255]
        self.mask = mask
        self.patch_size = patch_size
        self.transform = transform
        self.shape_ori = np.array(image.shape[:2])
        self.num_patches = ((self.shape_ori + patch_size -1) // self.stride)
        self.total_patches = self.num_patches[0] * self.num_patches[1]
        self.filter_prop = filter_prop
    def __len__(self):
        return self.total_patches
    def __getitem__(self, idx):
        # coordinate of the top left corner of the metapixel
        i = (idx // self.num_patches[1]) * self.stride
        j = (idx % self.num_patches[1]) * self.stride
        start_i, start_j = i, j
        end_i, end_j = min(self.shape_ori[0], i + self.patch_size), min(self.shape_ori[1], j + self.patch_size)
        patch = self.image[start_i:end_i, start_j:end_j, :]
        # Pad if necessary to ensure 224x224 size
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=patch.dtype)
            padded_patch[(self.patch_size-patch.shape[0])//2:(self.patch_size-patch.shape[0])//2+patch.shape[0], 
                         (self.patch_size-patch.shape[1])//2:(self.patch_size-patch.shape[1])//2+patch.shape[1]] = patch
            patch = padded_patch
        patch = Image.fromarray(patch.astype('uint8')).convert('RGB')
        metapixel_mask = self.mask[(i):(i+self.patch_size), (j):(j+self.patch_size)]
        keep_flag = np.sum(metapixel_mask) > self.patch_size*self.patch_size * self.filter_prop
        return self.transform(patch), (i, j), keep_flag
    def plot_mask(self):
        mask_plot = self.image.copy()
        mask_plot[self.mask == 0] = [0, 255, 0]
        mask_plot = cv2.addWeighted(self.image, 0.7, mask_plot, 0.3, 0)
        mask_plot = cv2.resize(mask_plot, (0,0), fx=0.1, fy=0.1)
        return mask_plot
    def plot_patch_loc(self, plot_spot = True):
        patch_plot = self.image.copy()
        for idx in range(self.total_patches):
            i = (idx // self.num_patches[1]) * self.stride
            j = (idx % self.num_patches[1]) * self.stride
            start_i, start_j = i, j
            end_i, end_j = min(self.shape_ori[0], i + self.patch_size), min(self.shape_ori[1], j + self.patch_size)
            metapixel_mask = self.mask[(i):(i+self.patch_size), (j):(j+self.patch_size)]
            keep_flag = np.sum(metapixel_mask) > self.patch_size*self.patch_size * self.filter_prop
            if keep_flag:
                if (idx // self.num_patches[1] + idx % self.num_patches[1]) % 2 == 0:
                    patch_plot[start_i:end_i, start_j:end_j, :] = [0, 0, 255]
                else:
                    patch_plot[start_i:end_i, start_j:end_j, :] = [0, 255, 255]
            else:
                patch_plot[start_i:end_i, start_j:end_j, :] = [0, 255, 0]
            # Color two consecutive patch to check overlapping
            # if idx == 500:
            #     patch_plot[start_i:end_i, start_j:end_j, :] = [0, 0, 255]
            # elif idx == 501:
            #     patch_plot[start_i:end_i, start_j:end_j, :] = [0, 255, 255]
        patch_plot = cv2.addWeighted(self.image, 0.7, patch_plot, 0.3, 0)
        ## Plot delimiters
        for idx in range(self.total_patches):
            i = (idx // self.num_patches[1]) * self.stride
            j = (idx % self.num_patches[1]) * self.stride
            start_i, start_j = i, j
            end_i, end_j = min(self.shape_ori[0], i + self.patch_size), min(self.shape_ori[1], j + self.patch_size)
            if (idx // self.num_patches[1] + idx % self.num_patches[1]) % 4 == 0:
                patch_plot[start_i:(start_i+self.patch_size//5), start_j:(start_j+self.patch_size//5), :] = [0, 0, 0]
                patch_plot[(end_i-self.patch_size//5):end_i, (end_j-self.patch_size//5):end_j, :] = [0, 0, 0]
            elif (idx // self.num_patches[1] + idx % self.num_patches[1]) % 4 == 1:
                patch_plot[start_i:(start_i+self.patch_size//5), start_j:(start_j+self.patch_size//5), :] = [255,255,255]
                patch_plot[(end_i-self.patch_size//5):end_i, (end_j-self.patch_size//5):end_j, :] = [255,255,255]
            elif (idx // self.num_patches[1] + idx % self.num_patches[1]) % 4 == 2:
                patch_plot[start_i:(start_i+self.patch_size//5), start_j:(start_j+self.patch_size//5), :] = [255, 0, 0]
                patch_plot[(end_i-self.patch_size//5):end_i, (end_j-self.patch_size//5):end_j, :] = [255, 0, 0]
            else:
                patch_plot[start_i:(start_i+self.patch_size//5), start_j:(start_j+self.patch_size//5), :] = [255, 0, 255]
                patch_plot[(end_i-self.patch_size//5):end_i, (end_j-self.patch_size//5):end_j, :] = [255, 0, 255]
        
        if plot_spot is not None and self.spot_radius_pixel is not None and self.loc is not None:
            spot_plot = patch_plot.copy()
            #! Notice that this the first column of loc is the col index for the image matrix, which is the x axis direction on the plot!
            loc = self.loc + [self.col_pad//2, self.row_pad//2]
            for (x, y) in loc:
                spot_plot = cv2.circle(spot_plot, (int(x), int(y)), self.spot_radius_pixel, (255, 0, 0), -1)
            patch_plot = cv2.addWeighted(patch_plot, 0.5, spot_plot, 0.5, 0)
        patch_plot[self.mask == 0] = [255, 255, 255]  # Color the masked area as white
        patch_plot = cv2.resize(patch_plot, (0,0), fx=0.1, fy=0.1)
        return patch_plot

# Modify the Dataset class to output data on the correct device
class HistologyGeneExpressionDataset(Dataset):
    def __init__(self, data, device):
        self.data = []
        self.device = device
        for embs, windows_mask, spot_weight, gene_expressions, gene_mask, bulk_filtered, bulk_rank, max_exp, father_patches, major_prop in data:
            self.data.append((embs,windows_mask,spot_weight, gene_expressions, gene_mask, bulk_filtered, bulk_rank, max_exp, father_patches, major_prop))
    #
    def __len__(self):
        return len(self.data)
    #
    def __getitem__(self, idx):
        emb, windows_mask, spot_weight, gene_expression, gene_mask, bulk, bulk_rank, max_exp, father_patches, major_prop = self.data[idx]
        for key in emb.keys():
            emb[key] = torch.tensor(emb[key], dtype=torch.float32).to(self.device)
        for key in windows_mask.keys():
            windows_mask[key] = torch.tensor(windows_mask[key], dtype=torch.float32).to(self.device)
        for key in spot_weight.keys():
            spot_weight[key] = torch.tensor(spot_weight[key], dtype=torch.float32).to(self.device)
        for key in father_patches.keys():
            father_patches[key] = torch.tensor(father_patches[key], dtype=torch.int).to(self.device)
        return emb, windows_mask, spot_weight, torch.tensor(
            gene_expression, dtype=torch.float32
        ).to(self.device), torch.tensor(gene_mask).to(self.device), torch.tensor(
            bulk, dtype=torch.float32
        ).to(self.device), torch.tensor(
            bulk_rank, dtype=torch.float32
        ).to(self.device), torch.tensor(
            max_exp, dtype=torch.float32
        ).to(self.device), father_patches, torch.tensor(
            major_prop, dtype=torch.float32
        ).to(self.device)
    
def my_collate(batch):
    emb = [item[0] for item in batch]
    windows_mask = [item[1] for item in batch]
    spot_weight = [item[2] for item in batch]
    gene_expression = [item[3] for item in batch]
    gene_mask = [item[4] for item in batch]
    bulk = [item[5] for item in batch]
    bulk_rank = [item[6] for item in batch]
    max_exp = [item[7] for item in batch]
    father_patches = [item[8] for item in batch]
    major_prop = [item[9] for item in batch]
    return emb, windows_mask, spot_weight, gene_expression, gene_mask, bulk, bulk_rank, max_exp, father_patches, major_prop

def my_collate_pred(batch):
    emb = [item[0] for item in batch]
    bulk = [item[1] for item in batch]
    bulk_rank = [item[2] for item in batch]
    father_patches = [item[3] for item in batch]
    return emb, bulk, bulk_rank, father_patches
