import pandas as pd
import os
import pyvips
from hest import VisiumReader
import argparse
parser = argparse.ArgumentParser(description='Preprocess H&E stained spatial transcriptomics data using HEST')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--raw_data_path', '-r', type=str, default='raw_data', help='Directory containing sample subdirectories')
parser.add_argument('--metadata_path', '-m', type=str, default='Sample_metadata.csv', help='Output CSV file name')
parser.add_argument('--output_path', '-o', type=str, default='processed_data', help='Directory to save processed data')
parser.add_argument('--pixel_size', type=int, default=0.5, help='Pixel size for H&E images in um/px')
parser.add_argument('--patch_size', type=int, default=224, help='Patch size for dumping patches')
args = parser.parse_args()

main_dir = args.project_path
data_dir = f"{main_dir}/{args.raw_data_path}"
metadata = pd.read_csv(f"{main_dir}/{args.metadata_path}")
os.makedirs(f"{main_dir}/{args.output_path}", exist_ok=True)
os.makedirs(f"{main_dir}/{args.output_path}/patches", exist_ok=True)
os.makedirs(f"{main_dir}/{args.output_path}/metadata", exist_ok=True)
os.makedirs(f"{main_dir}/{args.output_path}/patches_vis", exist_ok=True)
os.makedirs(f"{main_dir}/{args.output_path}/pixel_size_vis", exist_ok=True)
os.makedirs(f"{main_dir}/{args.output_path}/spatial_plots", exist_ok=True)
os.makedirs(f"{main_dir}/{args.output_path}/st", exist_ok=True)
os.makedirs(f"{main_dir}/{args.output_path}/tissue_seg", exist_ok=True)
os.makedirs(f"{main_dir}/{args.output_path}/thumbnails", exist_ok=True)
os.makedirs(f"{main_dir}/{args.output_path}/wsis", exist_ok=True)
os.makedirs(f"{main_dir}/{args.output_path}/tmp", exist_ok=True)
for idx in range(metadata.shape[0]):
    print(f"Processing sample {idx+1}/{metadata.shape[0]}: {metadata.iloc[idx]['Sample']}")
    fullres_img_path = f'{data_dir}/{metadata.iloc[idx]["Dir_name"]}/{metadata.iloc[idx]["Image_name"]}'
    bc_matrix_path = f'{data_dir}/{metadata.iloc[idx]["Dir_name"]}/filtered_feature_bc_matrix.h5'
    spatial_coord_path = f'{data_dir}/{metadata.iloc[idx]["Dir_name"]}/spatial'
    st = VisiumReader().read(
        fullres_img_path, # path to a full res image
        bc_matrix_path, # path to filtered_feature_bc_matrix.h5
        spatial_coord_path=spatial_coord_path,
        save_autoalign=True # pass this argument to visualize the fiducial autodetection
    )
    st.segment_tissue()
    st.save_spatial_plot(save_path=f"{main_dir}/{args.output_path}/tmp")
    st.dump_patches(
                f"{main_dir}/{args.output_path}/tmp",
                name=metadata.iloc[idx]["Sample"],
                target_patch_size=args.patch_size,
                target_pixel_size=args.pixel_size # pixel size of the patches in um/px after rescaling
    )
    st.save(path=f"{main_dir}/{args.output_path}/tmp",plot_pxl_size=True,bigtiff = True)
    os.system(f"mv {main_dir}/{args.output_path}/tmp/{metadata.iloc[idx]['Sample']}.h5 {main_dir}/{args.output_path}/patches/")
    os.system(f"mv {main_dir}/{args.output_path}/tmp/{metadata.iloc[idx]['Sample']}_patch_vis.png {main_dir}/{args.output_path}/patches_vis/")
    os.system(f"mv {main_dir}/{args.output_path}/tmp/aligned_adata.h5ad {main_dir}/{args.output_path}/st/{metadata.iloc[idx]['Sample']}.h5ad")
    os.system(f"mv {main_dir}/{args.output_path}/tmp/aligned_fullres_HE.tif {main_dir}/{args.output_path}/wsis/{metadata.iloc[idx]['Sample']}.tif")
    os.system(f"mv {main_dir}/{args.output_path}/tmp/downscaled_fullres.jpeg {main_dir}/{args.output_path}/thumbnails/{metadata.iloc[idx]['Sample']}_downscaled_fullres.jpeg")
    os.system(f"mv {main_dir}/{args.output_path}/tmp/metrics.json {main_dir}/{args.output_path}/metadata/{metadata.iloc[idx]['Sample']}.json")
    os.system(f"mv {main_dir}/{args.output_path}/tmp/tissue_contours.geojson {main_dir}/{args.output_path}/tissue_seg/{metadata.iloc[idx]['Sample']}_contours.geojson")
    os.system(f"mv {main_dir}/{args.output_path}/tmp/tissue_seg_vis.jpg {main_dir}/{args.output_path}/tissue_seg/{metadata.iloc[idx]['Sample']}_vis.jpg")
    os.system(f"mv {main_dir}/{args.output_path}/tmp/spatial_plots.png {main_dir}/{args.output_path}/spatial_plots/{metadata.iloc[idx]['Sample']}_spatial_plots.png")
    os.system(f"mv {main_dir}/{args.output_path}/tmp/pixel_size_vis.png {main_dir}/{args.output_path}/pixel_size_vis/{metadata.iloc[idx]['Sample']}_pixel_size_vis.png")
    os.system(f"rm -rf {main_dir}/{args.output_path}/tmp/*")