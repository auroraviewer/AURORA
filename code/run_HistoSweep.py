import os
import argparse
import sys
code_path = os.getenv('HISTOSWEEP_PATH')
sys.path.insert(0, f"{code_path}/HistoSweep")
import pandas as pd
from HistoSweep_single import run_histosweep_single

import argparse
parser = argparse.ArgumentParser(description='Enhance prediction by iStar.')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
# parser.add_argument('--interim_path', '-i', type=str, default='AURORA_interim', help='Interim data directory')
parser.add_argument('--pred_path', '-t', type=str, default='AURORA_pred', help='Directory to store predicted data')
parser.add_argument('--sample_sheet', type=str, default=None, help='Dataset directory name')
args = parser.parse_args()
meta = pd.read_csv(f"{args.project_path}/{args.sample_sheet}", header = 0)
dataset_test = meta['Sample_name'].values

#### HistoSweep parameters and processing flags ####
# Processing flags
need_scaling_flag = False 
need_preprocessing_flag = False 

# Image and pipeline parameters
pixel_size_raw = 0.5      # Microns per pixel in original image
pixel_size = 0.5          # Target pixel size (for rescaling, if needed)
patch_size = 16           # Size of each patch used in analysis
density_thresh = 100      # Density threshold
clean_background_flag = True
min_size = 10             # Parameter used to remove isolated debris and small specs outside tissue


for dataset in dataset_test:
    HE_prefix = f'{args.project_path}/{args.pred_path}/istar_pred/{dataset}' # Path prefix to your H&E image folder
    output_dir = f"{HE_prefix}/HistoSweep_Output"  #Folder for HistoSweep results
    config = {
        'HE_prefix': f"{HE_prefix}",
        'output_dir':  f"{output_dir}",
        'pixel_size_raw': pixel_size_raw,
        'density_thresh': density_thresh,
        'clean_background_flag': clean_background_flag,
        'min_size': min_size,
        'patch_size': patch_size,
        'pixel_size': pixel_size,
        'need_scaling_flag': need_scaling_flag,
        'need_preprocessing_flag': need_preprocessing_flag,
    }

    os.makedirs(config['output_dir'], exist_ok=True)
    run_histosweep_single(
        image_path = config['HE_prefix'],
        output_base_dir=config['output_dir'],
        pixel_size_raw=config['pixel_size_raw'],
        density_thresh=config['density_thresh'],
        clean_background_flag=config['clean_background_flag'],
        min_size=config['min_size'],
        patch_size=config['patch_size'],
        pixel_size=config['pixel_size'],
        need_scaling_flag=config['need_scaling_flag'],
        need_preprocessing_flag=config['need_preprocessing_flag'],
        HE_prefix=config['HE_prefix']
    )

    print(f"✅ HistoSweep processing complete for {dataset}.")

