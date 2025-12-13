import pandas as pd
import os
import argparse
parser = argparse.ArgumentParser(description='Generate sample metadata CSV from raw data directories.')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--raw_data_path', '-r', type=str, default='raw_data', help='Directory containing sample subdirectories')
parser.add_argument('--output_name', '-o', type=str, default='Sample_metadata.csv', help='Output CSV file name')
args = parser.parse_args()

main_dir = args.project_path
data_dir = f"{main_dir}/{args.raw_data_path}"
all_files = os.listdir(data_dir)
metadata = pd.DataFrame(columns=["Sample","Dir_name", "Image_name"])
for file in all_files:
    sample_name = file#.split("_")[0]
    dir_name = file
    img_candidate = os.listdir(f"{data_dir}/{file}")
    img_candidate = [img for img in img_candidate if img.endswith(".tif")]
    if len(img_candidate) > 1:
        print(f"Warning: Multiple images found for sample {sample_name} in directory {dir_name}. Only the first ({img_candidate[0]}) will be recorded. Please double check!")
    if len(img_candidate) == 0:
        print(f"Warning: No images found for sample {sample_name} in directory {dir_name}.")
    image_name = img_candidate[0] if img_candidate else "No image found"
    metadata = pd.concat(
        [metadata, pd.DataFrame({"Sample": [sample_name], "Dir_name": [dir_name], "Image_name": [image_name]})],
        ignore_index=True
    )
    
metadata.to_csv(f"{main_dir}/{args.output_name}", index=False)