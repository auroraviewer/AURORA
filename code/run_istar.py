import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40)) # Example: set to 2^40 pixels
import tifffile
import numpy as np
import utils
import pandas as pd
from istar.preprocess import adjust_margins, reduce_mask
from istar.utils import save_image

def load_image(dataset, main_dir, interim_dir):
    print(f"Loading image {dataset}...")
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
    return image, mask

# get command line arguments
import argparse
parser = argparse.ArgumentParser(description='Enhance prediction by iStar.')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--interim_path', '-i', type=str, default='AURORA_interim', help='Interim data directory')
parser.add_argument('--pred_path', '-t', type=str, default='AURORA_pred', help='Directory to store predicted data')
parser.add_argument('--sample_sheet', type=str, default=None, help='Dataset directory name')
parser.add_argument('--gene_list_file', type=str, default='AURORA_interim/Gene_to_predict.csv', help='Gene list file path')
parser.add_argument('--celltype_file', type=str, default='AURORA_interim/deconvolution/cell_types.csv', help='Cell type list file path')
parser.add_argument('--patch_size', type=int, default=224, help='Patch size for prediction')
# parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
args = parser.parse_args()

main_dir = args.project_path
meta = pd.read_csv(f"{main_dir}/{args.sample_sheet}", header = 0)
dataset_test = meta['Sample_name'].values
gene_list = pd.read_csv(f"{main_dir}/{args.gene_list_file}")['Genes'].values
major_cell_type = pd.read_csv(f"{main_dir}/{args.celltype_file}",header=None).values.squeeze()
major_cell_type = [x.replace(" ","_") for x in major_cell_type]
major_cell_type = [x.replace("-","_") for x in major_cell_type]


os.makedirs(f"{main_dir}/{args.pred_path}/istar_pred", exist_ok=True)
for dataset in dataset_test:
    # dataset = args.dataset #"TCGA-AO-A0JE-01A-01-TSA"
    image, mask = load_image(dataset, main_dir, args.interim_path)
    # print(image.shape)
    mask = mask.astype('uint8')
    pad = 256
    os.makedirs(f"{main_dir}/{args.pred_path}/istar_pred/{dataset}", exist_ok=True)
    image = adjust_margins(image, pad=pad, pad_value=255)
    tifffile.imwrite(f"{main_dir}/{args.pred_path}/istar_pred/{dataset}/he.tif", image, compression='zlib')
    mask = adjust_margins(mask, pad=pad, pad_value=mask.min())
    tifffile.imwrite(f"{main_dir}/{args.pred_path}/istar_pred/{dataset}/mask.tif", mask, compression='zlib')
    mask = reduce_mask(mask, factor=16)
    save_image(mask, f'{main_dir}/{args.pred_path}/istar_pred/{dataset}/mask-small.png')

    result = utils.load_pickle(f"{main_dir}/{args.pred_path}/pred/{dataset}-pred.pickle")
    pred = result['exp_pred'][:,2,:].squeeze()
    patch_loc_before_pad = result['patches_loc_before_pad'][args.patch_size][:,[1,0]]
    patch_keep = np.logical_and(patch_loc_before_pad[:,0] > 0, patch_loc_before_pad[:,1] > 0)
    patch_keep = np.logical_and(patch_loc_before_pad[:,0] + 224 < image.shape[1], patch_keep)
    patch_keep = np.logical_and(patch_loc_before_pad[:,1] + 224 < image.shape[0], patch_keep)

    pred_df = pd.DataFrame(pred, columns=gene_list)
    pred = result['prop_pred'][:,2,:].squeeze()
    pred = np.exp(pred)
    pred_df = pd.concat([pred_df, pd.DataFrame(pred, columns=major_cell_type)], axis=1)
    pred_df = pred_df.loc[patch_keep,:]
    # save pred_df to tsv
    pred_df.to_csv(f"{main_dir}/{args.pred_path}/istar_pred/{dataset}/cnts.tsv", sep="\t", index=True)
    gene_names = pred_df.columns
    # save gene names to txt
    with open(f"{main_dir}/{args.pred_path}/istar_pred/{dataset}/gene-names.txt", "w") as f:
        for gene in gene_names:
            f.write(gene + "\n")

    patch_loc_before_pad = patch_loc_before_pad[patch_keep,:]
    locs = pd.DataFrame(patch_loc_before_pad + 112, columns=['x', 'y']) # move top left corner to center
    locs['x'] = locs['x'].astype(int)
    locs['y'] = locs['y'].astype(int)
    # save locs to tsv
    locs.to_csv(f"{main_dir}/{args.pred_path}/istar_pred/{dataset}/locs.tsv", sep="\t", index=True)
    os.system(f"bash $CODE_PATH/run_istar_core.sh {main_dir}/{args.pred_path}/istar_pred/{dataset}/")
