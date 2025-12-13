import os
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--data_path', '-d', type=str, default='./wsis/', help='Data directory path')
parser.add_argument('--save_path', '-s', type=str, default='./wsis_segmentation/', help='Save directory path')
parser.add_argument('--sample_sheet', type=str, default=None, help='Dataset directory name')
parser.add_argument('--code_path', type=str, default='./code/', help='Code directory path')
parser.add_argument('--preset', type=str, default='tcga.csv', help='Preset file for segmentation')
args = parser.parse_args()
os.makedirs(f"{args.project_path}/{args.save_path}", exist_ok=True)
if args.sample_sheet is not None:
    if args.sample_sheet.endswith('.csv'):
        wsis_list = pd.read_csv(f"{args.project_path}/{args.sample_sheet}")
    elif args.sample_sheet.endswith('.tsv'):
        wsis_list = pd.read_csv(f"{args.project_path}/{args.sample_sheet}", sep="\t")

if args.sample_sheet is None:
    # All WSIs in one folder
    os.system(f"python -u {args.code_path}/create_patches_fp.py \
               --source {args.project_path}/{args.data_path} \
               --save_dir {args.project_path}/{args.save_path} \
               --seg --preset {args.project_path}/{args.preset}")
else:
    # Each WSI has its own folder (e.g. TCGA format)
    for idx in wsis_list.index:
        # sample_id = wsis_list.loc[idx, "Sample ID"]
        file_id = wsis_list.loc[idx, "Image_folder"]
        # file_name = wsis_list.loc[idx, "File Name"]
        os.system(f"python {args.code_path}/create_patches_fp.py \
                --source {args.project_path}/{args.data_path}/{file_id} \
                --save_dir {args.project_path}/{args.save_path} \
                --seg --preset {args.project_path}/{args.preset}")
