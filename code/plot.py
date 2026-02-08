import numpy as np
import pandas as pd
import os
import argparse
import utils
import matplotlib.pyplot as plt
import cv2
def load_image(dataset, main_dir):
    import tifffile
    print(f"Loading image {dataset}...")
    # Load image
    image_path = f"{main_dir}/{dataset}_rescaled.tif"
    mask_path = f"{main_dir}/{dataset}_mask_rescaled.tif"
    with tifffile.TiffFile(image_path) as tif:
        # Access the image data at different levels
        image = tif.pages[0].asarray()
    with tifffile.TiffFile(mask_path) as tif:
        # Access the image data at different levels
        mask = tif.pages[0].asarray()
    image[mask == 0] = 255
    return image, mask

parser = argparse.ArgumentParser(description='AURORA Prediction Script')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--interim_path', '-i', type=str, default='AURORA_interim', help='Interim data directory')
parser.add_argument('--pred_path', '-t', type=str, default='AURORA_pred', help='Directory to store predicted data')
parser.add_argument('--resolution', '-r', type=int, default=224, help='For iStar-enhanced prediction, use -1')
# parser.add_argument('--parameter_json', '-j', type=str, default='parameters.json', help='JSON file with parameters')
# parser.add_argument('--model_pth', type=str, default='AURORA_model_weights.pth', help='Path to the pretrained model file')
# parser.add_argument('--sample_sheet', type=str, default=None, help='Dataset directory name')
# parser.add_argument('--bulk', type=str, default='bulk_data', help='Bulk data file name')
# parser.add_argument('--batch_size', type=int, default=100, help='Batch size for prediction')
parser.add_argument('--samples', '-s', type=str, nargs='+', default=None, help='List of sample names to plot')
parser.add_argument('--genes', '-g', type=str, nargs='+', default=None, help='List of gene names to plot or a CSV file containing gene names to plot in the first column')
parser.add_argument('--celltypes', '-c', type=str, nargs='+', default=None, help='List of cell types to plot or a CSV file containing cell types to plot in the first column')
parser.add_argument('--plot_dir', '-d', type=str, default='AURORA_plots', help='Directory to save plots')
parser.add_argument('--HistoSweep_mask', action='store_true', help='Use HistoSweep mask to the plots')
args = parser.parse_args()
cmap = plt.get_cmap('turbo')

main_dir = args.project_path
interim_dir = os.path.join(main_dir, args.interim_path)
pred_dir = os.path.join(main_dir, args.pred_path)
plot_dir = os.path.join(main_dir, args.plot_dir)
os.makedirs(plot_dir, exist_ok=True)
if args.genes is None and args.celltypes is None:
    raise ValueError("Please specify at least one of --genes or --celltypes to plot.")
else:
    if '.csv' in args.genes:
        genes_to_plot = pd.read_csv(args.genes).values.squeeze().tolist()
    else:
        genes_to_plot = args.genes
    if '.csv' in args.celltypes:
        celltypes_to_plot = pd.read_csv(args.celltypes).values.squeeze().tolist()
    else:
        celltypes_to_plot = args.celltypes

for sample in args.samples:
    os.makedirs(os.path.join(plot_dir, sample), exist_ok=True)
    image, mask = load_image(sample, f"{interim_dir}/UNI_multiscale_patches/wsis_rescaled/")
    if args.HistoSweep_mask:
        mask_path = os.path.join(main_dir, args.pred_path, 'istar_pred', sample, 'HistoSweep_Output', 'mask.png')
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            print(f"HistoSweep mask not found for sample {sample}. Using default mask.")
    mask = cv2.resize(mask, (image.shape[1]//10,image.shape[0]//10))
    if args.resolution > 0:
        results = utils.load_pickle(os.path.join(pred_dir, 'pred', f"{sample}-pred.pickle"))
        patch_loc_before_pad = results['patches_loc_before_pad'][args.resolution] // 10
        patch_loc_before_pad = patch_loc_before_pad[:,[1,0]] 
        if args.resolution not in results['resolutions']:
            raise ValueError(f"Resolution {args.resolution} not found in the prediction results for sample {sample}. Please specify a resolution from {results['resolutions']} or -1 for iStar-enhanced predictions.")
        patch_size_index = results['resolutions'].index(args.resolution)
        if genes_to_plot is not None:
            ## Plot gene expression
            pred = results['exp_pred']
            for gene in genes_to_plot:
                if gene not in results['genes']:
                    print(f"Gene {gene} not found in the prediction results for sample {sample}. Skipping this gene.")
                    continue
                gene_idx = results['genes'].index(gene)
                ## rescale the size to 1/10
                exp_image = np.zeros((image.shape[0]//10, image.shape[1]//10,3))
                occurence = np.zeros((image.shape[0]//10, image.shape[1]//10))
                pred_use = pred[:,patch_size_index,gene_idx]
                pred_use = pred_use / pred_use.max()
                for i in range(len(patch_loc_before_pad)):
                    loc_patch = patch_loc_before_pad[i]
                    pred_patch = pred_use[i]
                    x_start = max(int(loc_patch[1]),0)
                    x_end = min(int(loc_patch[1] + args.resolution//10), image.shape[1]//10)
                    y_start = max(int(loc_patch[0]),0)
                    y_end = min(int(loc_patch[0] + args.resolution//10), image.shape[0]//10)
                    color_tmp = cmap(pred_patch)
                    exp_image[y_start:y_end, x_start:x_end,:] += [color_tmp[2],color_tmp[1],color_tmp[0]]
                    occurence[y_start:y_end, x_start:x_end] += 1
                occurence[occurence==0] = 1
                for k in range(3):
                    exp_image[:,:,k] = exp_image[:,:,k] / occurence
                exp_image = (exp_image*255).astype(np.uint8)
                exp_image[mask==0,:] = 0
                cv2.imwrite(f"{plot_dir}/{sample}/{sample}-{gene}-{args.resolution}pixels.png", exp_image)
        if celltypes_to_plot is not None:
            ## Plot cell type proportion
            pred = np.exp(results['prop_pred'])
            for celltype in celltypes_to_plot:
                if celltype not in results['cell_types']:
                    print(f"Cell type {celltype} not found in the prediction results for sample {sample}. Skipping this cell type.")
                    continue
                celltype_idx = results['cell_types'].tolist().index(celltype)
                ## rescale the size to 1/10
                celltype_image = np.zeros((image.shape[0]//10, image.shape[1]//10,3))
                occurence = np.zeros((image.shape[0]//10, image.shape[1]//10))
                pred_use = pred[:,patch_size_index,celltype_idx]
                for i in range(len(patch_loc_before_pad)):
                    loc_patch = patch_loc_before_pad[i]
                    pred_patch = pred_use[i]
                    x_start = max(int(loc_patch[1]),0)
                    x_end = min(int(loc_patch[1] + args.resolution//10), image.shape[1]//10)
                    y_start = max(int(loc_patch[0]),0)
                    y_end = min(int(loc_patch[0] + args.resolution//10), image.shape[0]//10)
                    color_tmp = cmap(pred_patch)
                    celltype_image[y_start:y_end, x_start:x_end,:] += [color_tmp[2],color_tmp[1],color_tmp[0]]
                    occurence[y_start:y_end, x_start:x_end] += 1
                occurence[occurence==0] = 1
                for k in range(3):
                    celltype_image[:,:,k] = celltype_image[:,:,k] / occurence
                celltype_image = (celltype_image*255).astype(np.uint8)
                celltype_image[mask==0,:] = 0
                cv2.imwrite(f"{plot_dir}/{sample}/{sample}-{celltype}_proportion-{args.resolution}pixels.png", celltype_image)
    else:
        import pickle
        mask = cv2.imread(f"{pred_dir}/istar_pred/{sample}/mask-small.png", cv2.IMREAD_GRAYSCALE)
        if args.HistoSweep_mask:
            mask_path = os.path.join(main_dir, args.pred_path, 'istar_pred', sample, 'HistoSweep_Output', 'mask.png')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                print(f"HistoSweep mask not found for sample {sample}. Using default mask.")
        ## iStar-enhanced predictions
        if genes_to_plot is not None:
            for gene in genes_to_plot:
                istar_file = f"{pred_dir}/istar_pred/{sample}/cnts-super/{gene}.pickle"
                if not os.path.exists(istar_file):
                    print(f"iStar-enhanced prediction file for gene {gene} not found for sample {sample}. Skipping this gene.")
                    continue
                with open(istar_file, 'rb') as f:
                    istar_pred = pickle.load(f)
                istar_pred = istar_pred/ istar_pred.max()  # Normalize the prediction to [0, 1]
                if mask.shape[0] != istar_pred.shape[1]:
                    mask = cv2.resize(mask, (istar_pred.shape[1], istar_pred.shape[0]))
                plot_image = np.zeros((istar_pred.shape[0], istar_pred.shape[1], 3))
                for i in range(istar_pred.shape[0]):
                    for j in range(istar_pred.shape[1]):
                        color_tmp = cmap(istar_pred[i,j])
                        plot_image[i,j,:] = [color_tmp[2]*255, color_tmp[1]*255, color_tmp[0]*255]
                plot_image = plot_image.astype(np.uint8)
                plot_image[mask==0,:] = 0
                cv2.imwrite(f"{plot_dir}/{sample}/{sample}-{gene}-iStar-enhanced.png", plot_image)
        if celltypes_to_plot is not None:
            import pickle
            istar_celltype = []
            results = utils.load_pickle(os.path.join(pred_dir, 'pred', f"{sample}-pred.pickle"))
            for cell_type in results['cell_types']:
                cell_type = cell_type.replace(" ", "_")
                cell_type = cell_type.replace("-", "_")
                istar_file = f"{pred_dir}/istar_pred/{sample}/cnts-super/{cell_type}.pickle"
                with open(istar_file, 'rb') as f:
                    istar_pred = pickle.load(f)
                istar_celltype.append(istar_pred)
            istar_celltype = np.array(istar_celltype)
            istar_celltype = istar_celltype.transpose((1, 2, 0))  # Shape: (,, num_cell_types)
            istar_celltype = istar_celltype / istar_celltype.sum(axis=2, keepdims=True)  # Normalize each cell type
            mask = cv2.resize(mask, (istar_celltype.shape[1], istar_celltype.shape[0]))
            for celltype in celltypes_to_plot:
                if celltype not in results['cell_types']:
                    print(f"Cell type {celltype} not found in the prediction results for sample {sample}. Skipping this cell type.")
                    continue
                celltype_idx = results['cell_types'].tolist().index(celltype)
                plot_image = np.zeros((istar_celltype.shape[0], istar_celltype.shape[1], 3))
                for i in range(istar_celltype.shape[0]):
                    for j in range(istar_celltype.shape[1]):
                        color_tmp = cmap(istar_celltype[i,j,celltype_idx])
                        plot_image[i,j,:] = [color_tmp[2]*255, color_tmp[1]*255, color_tmp[0]*255]
                plot_image = plot_image.astype(np.uint8)
                plot_image[mask==0,:] = 0
                cv2.imwrite(f"{plot_dir}/{sample}/{sample}-{celltype}_proportion-iStar-enhanced.png", plot_image)



