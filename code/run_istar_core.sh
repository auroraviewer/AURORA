#!/bin/bash
set -e

prefix=$1  #"/project/iclip2/zchen/TCGA_BRCA/istar_pred/$1/" # e.g. data/demo/
epoch_num=$2

device="cuda"  # "cuda" or "cpu"
pixel_size=0.5  # desired pixel size for the whole analysis
# n_genes=1000  # number of most variable genes to impute

# preprocess histology image
echo $pixel_size > ${prefix}pixel-size.txt
echo 0.5 > ${prefix}pixel-size-raw.txt
echo 112 > ${prefix}radius.txt

# extract histology features
python -u ${ISTAR_PATH}extract_features.py ${prefix} --device=${device} --tif
# train gene expression prediction model and predict at super-resolution
python -u ${ISTAR_PATH}impute.py ${prefix} --epochs=${epoch_num} --device=${device} --tif # train model from scratch
# segment image by gene features
python -u ${ISTAR_PATH}cluster.py --filter-size=8 --min-cluster-size=20 --n-clusters=10 --mask=${prefix}mask-small.png ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/
