import scanpy as sc
from utils import load_pickle, save_tsv
import pandas as pd
import os
import numpy as np
import argparse
from tqdm import tqdm
import squidpy as sq
import json
parser = argparse.ArgumentParser(description='Rescale H&E images to target pixel size and adjust ST spot locations accordingly.')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--output_path', '-o', type=str, default='AURORA_interim', help='Output directory')
parser.add_argument('--metadata_path', '-m', type=str, default='Sample_metadata.csv', help='Sample metadata CSV file name')
parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
parser.add_argument('--output_folder', '-f', type=str, default='cnts_smooth', help='Output folder name for smoothed counts')
args = parser.parse_args()

main_dir = args.project_path
# folder containing all h5ad files for ST
st_folder = f"{main_dir}/{args.processed_data_path}/st/"
meta_folder = f"{main_dir}/{args.processed_data_path}/metadata/"
# folder containing all barcodes pickles
save_folder = f"{main_dir}/{args.output_path}/{args.output_folder}/"
bar_folder = f"{main_dir}/{args.output_path}/UNI_multiscale_patches/locs/"

if not os.path.exists(f"{main_dir}/gene_symbols.csv"):
    gene_symbols = sc.queries.biomart_annotations(
                "hsapiens", 
                attrs=["ensembl_gene_id", "hgnc_symbol"]
            ).set_index("ensembl_gene_id")
    gene_symbols.to_csv(f"{main_dir}/gene_symbols.csv")
else:
    gene_symbols = pd.read_csv(f"{main_dir}/gene_symbols.csv", index_col=0)

gene_symbols = gene_symbols[~gene_symbols.index.duplicated(keep='first')]
# bulk_norm_factor = bulk_norm_factor[~bulk_norm_factor.index.duplicated(keep='first')]
# save_folder = "/project/hest1k/cnts_norm_by_gene/"

meta = pd.read_csv(f"{main_dir}/{args.metadata_path}",index_col=0, header=0)
file_names = list(meta.index)

for file_name in tqdm(file_names):
    dataset = file_name
    if os.path.exists(f"{save_folder}{dataset}-cnts.tsv"):
        print(f"{file_name} already processed, skipping...")
        continue
    st_file= os.path.join(st_folder, f"{file_name}.h5ad")
    # print(f"Reading {st_file}") 
    st = sc.read_h5ad(st_file)
    # read json metadata file
    with open(f"{meta_folder}/{file_name}.json", 'r') as f:
        # Load the JSON data into a Python dictionary
        st_metadata = json.load(f)

    radius = 2 * st_metadata['inter_spot_dist'] / st_metadata['pixel_size_um_estimated']
    # Convert Ensembl gene IDs to gene symbols
    if st.var_names[0].startswith("ENSG"):
        st = st[:, st.var_names.isin(gene_symbols.index)]
        obsm_tmp = st.obsm
        st.var_names = gene_symbols.loc[st.var_names, "hgnc_symbol"].values
        st = st.to_df().T.groupby(st.var_names).sum().T
        st = sc.AnnData(st)
        st.obsm = obsm_tmp
    
    barcode_file = os.path.join(bar_folder, f"{dataset}-bars.pickle")
    # print(f"Reading {barcode_file}")
    barcode = load_pickle(barcode_file)
    barcode = [code.decode('utf-8') for code in barcode.flatten()]

    print("Filtering ...")
    st = st[barcode,:]
    if hasattr(st.X,'toarray'):
        df = pd.DataFrame(st.X.toarray(), index = st.obs_names, columns= st.var_names)
    else:
        df = pd.DataFrame(st.X, index = st.obs_names, columns= st.var_names)
    
    print("Smoothing ...")
    sq.gr.spatial_neighbors(st, radius=radius, set_diag = True, coord_type = 'generic')

    st.obsp['spatial_connectivities'] = st.obsp['spatial_connectivities'].todense()
    # adj_matrix = st.obsp['spatial_connectivities']/ np.sum(st.obsp['spatial_connectivities'], axis=1, keepdims=True)
    adj = np.asarray(st.obsp['spatial_connectivities'])
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adj_matrix = adj / row_sums
    df_smooth = pd.DataFrame(adj_matrix @ df.values, index = st.obs_names, columns= st.var_names)
    df_smooth = df_smooth.astype(np.float32)
    df_smooth = df_smooth.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12), axis=0)
    df_smooth = df_smooth.astype(np.float32)
    save_tsv(df_smooth, f"{save_folder}{dataset}-cnts.tsv")