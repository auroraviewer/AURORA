import scanpy as sc
from utils import load_pickle, save_tsv
import pandas as pd
import os
import numpy as np
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Rescale H&E images to target pixel size and adjust ST spot locations accordingly.')
parser.add_argument('--project_path', '-p', type=str, default='./', help='Main directory path')
parser.add_argument('--output_path', '-o', type=str, default='AURORA_interim', help='Output directory')
parser.add_argument('--metadata_path', '-m', type=str, default='Sample_metadata.csv', help='Sample metadata CSV file name')
parser.add_argument('--processed_data_path', '-d', type=str, default='processed_data', help='Processed data directory path')
parser.add_argument('--num_genes', type=int, default=3000, help='Number of highly expressed genes to select')
parser.add_argument('--genetype_file', type=str, default='TCGA_genetype.csv', help='File containing gene types information')
args = parser.parse_args()

main_dir = args.project_path
# folder containing all h5ad files for ST
st_folder = f"{main_dir}/{args.processed_data_path}/st/"
# folder containing all barcodes pickles
save_folder = f"{main_dir}/{args.output_path}/cnts/"
bar_folder = f"{main_dir}/{args.output_path}/UNI_multiscale_patches/locs/"
bulk_folder = f"{main_dir}/{args.output_path}/bulk/"

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

#Get summary statistics
gene_list = set()
for sample in tqdm(file_names):
    adata = sc.read_h5ad(f"{main_dir}/{args.processed_data_path}/st/{sample}.h5ad")
    if adata.var_names[0].startswith("ENSG"):
        adata = adata[:, adata.var_names.isin(gene_symbols.index)]
        adata.var_names = gene_symbols.loc[adata.var_names, "hgnc_symbol"].values
        adata = adata.to_df().T.groupby(adata.var_names).sum().T
        adata = sc.AnnData(adata)
    gene_list = gene_list.union(set(adata.var_names))

gene_list = list(gene_list)
gene_expression_sums = pd.DataFrame(0, index=gene_list, columns=file_names)
gene_expression_sums['occurance'] = 0

for sample in tqdm(file_names):
    print(f"Processing sample: {sample}")
    sample_path = f"{main_dir}/{args.processed_data_path}/st/{sample}.h5ad"
    adata = sc.read_h5ad(sample_path)
    if adata.var_names[0].startswith("ENSG"):
        adata = adata[:, adata.var_names.isin(gene_symbols.index)]
        adata.var_names = gene_symbols.loc[adata.var_names, "hgnc_symbol"].values
        adata = adata.to_df().T.groupby(adata.var_names).sum().T
        adata = sc.AnnData(adata)
    bulk_tmp = adata.X.sum(axis=0)
    if isinstance(bulk_tmp, pd.Series):
        bulk_tmp = bulk_tmp.to_frame().T
    else:
        bulk_tmp = pd.DataFrame(bulk_tmp.T, index=adata.var_names)
    for gene in gene_list:
        if gene in adata.var_names:
            gene_expression_sums.loc[gene,sample] += bulk_tmp.loc[gene].values[0]
            gene_expression_sums.loc[gene,'occurance'] += 1

# gene_expression_sums.to_csv(f"{main_dir}/Lung/gene_sample_statistics.csv")

x_occurence = gene_expression_sums['occurance']
gene_expression_sums = gene_expression_sums.drop(columns = ['occurance'])

#! Set this
gene_type = pd.read_csv(f"{main_dir}/{args.genetype_file}", index_col=0)
gene_type = gene_type.loc[gene_type.index.isin(gene_expression_sums.index), :]
x = gene_expression_sums.loc[gene_type.index, :].copy()
x = x.loc[gene_type['gene_type'] == 'protein_coding', :]
x = x[~x.index.duplicated(keep='first')]

gene_median = x.apply(lambda x: x[x>0].median(), axis = 1)
x.apply(lambda x: x/gene_median, axis = 0)
x_rank = x.rank(axis = 0, method = 'first', ascending=True)
x_rank_avg = x_rank.mean(axis=1)
x_rank_avg = x_rank_avg.sort_values(ascending = False)
gene_list = x_rank_avg.index[0:args.num_genes]
gene_list = pd.DataFrame(gene_list.values,columns=['Genes'])
gene_list.to_csv(f"{main_dir}/{args.output_path}/Gene_to_predict.csv")
gene_median.to_csv(f"{main_dir}/{args.output_path}/Gene_normalize_factor.csv")

bulk_norm_factor = pd.read_csv(f"{main_dir}/{args.output_path}/Gene_normalize_factor.csv", index_col=0, header=0)

for file_name in tqdm(file_names):
    st_file= os.path.join(st_folder, f"{file_name}.h5ad")
    # print(f"Reading {st_file}") 
    st = sc.read_h5ad(st_file)

    # Convert Ensembl gene IDs to gene symbols
    if st.var_names[0].startswith("ENSG"):
        st = st[:, st.var_names.isin(gene_symbols.index)]
        obsm_tmp = st.obsm
        st.var_names = gene_symbols.loc[st.var_names, "hgnc_symbol"].values
        st = st.to_df().T.groupby(st.var_names).sum().T
        st = sc.AnnData(st)
        st.obsm = obsm_tmp
    
    dataset = file_name
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
    
    print("Normalizing ...")
    df = df.astype(np.float32)
    df_normalized = df.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12), axis=0)
    df_normalized = df_normalized.astype(np.float32)
    save_tsv(df_normalized, f"{save_folder}{dataset}-cnts.tsv")

    # print("Smoothing ...")
    # if "TENX" in file_name:
    #     sq.gr.spatial_neighbors(st, n_neighs=6, set_diag = True, percentile=0.95)
    # else:
    #     sq.gr.spatial_neighbors(st, n_neighs=8, set_diag = True, percentile=0.95)

    # st.obsp['spatial_connectivities'] = st.obsp['spatial_connectivities'].todense()
    # adj_matrix = st.obsp['spatial_connectivities']/ np.sum(st.obsp['spatial_connectivities'], axis=1, keepdims=True)
    # df_smooth = pd.DataFrame(adj_matrix @ df.values, index = st.obs_names, columns= st.var_names)
    # df_smooth = df_smooth.astype(np.float32)
    # df_smooth = df_smooth.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12), axis=0)
    # df_smooth = df_smooth.astype(np.float32)
    # save_tsv(df_smooth, f"{save_folder}{dataset}-cnts-smooth.tsv")

    print("Generating pseudo-bulk ...")
    df = df.astype(np.float32)
    df_bulk = df.loc[:,df.columns.isin(bulk_norm_factor.index)].sum(axis=0)
    df_bulk = df_bulk.astype(np.float32)
    save_tsv(df_bulk, f"{bulk_folder}{dataset}-bulk.tsv")

    norm_factor_use = bulk_norm_factor.loc[df_bulk.index,:]
    df_bulk = df_bulk / norm_factor_use['0'].values
    df_bulk = df_bulk.rank(method = 'first', ascending=True, pct=True)
    save_tsv(df_bulk, f"{bulk_folder}{dataset}-bulk-rank.tsv")

    print("Storing the maximum expression ...")
    df = df.astype(np.float32)
    df_max = df.max(axis=0)
    df_max = df_max.astype(np.float32)
    
    save_tsv(df_max, f"{bulk_folder}{dataset}-max.tsv")

