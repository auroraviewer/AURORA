# AURORA

## Installation

### Set up environment
For convenience, we recommend creating and activating a dedicated conda environment before installing AURORA. If you haven't installed conda yet, we suggest using [Miniforge](https://conda-forge.org/download/), a lightweight distribution of conda.
```
conda create -n aurora python=3.12
conda activate aurora
```
### Download AURORA
```
git clone https://github.com/auroraviewer/AURORA.git
cd AURORA
```
The codes for AURORA will be under `./AURORA/code`. You can set this path as an environment variable `CODE_PATH` for later usage.
### Install dependencies
```
conda install conda-forge::pyvips
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```
**Note**: Please install [pytorch](https://pytorch.org/) consistent with your `cuda` version.

### Install HEST (optional)
AURORA relies on [HEST](https://github.com/mahmoodlab/HEST?tab=readme-ov-file) to preprocess samples before **training**.
```
git clone https://github.com/mahmoodlab/HEST.git
cd HEST
pip install -e .
cd ..
```

If a GPU is available on your machine, you can accelerate the preprocessing procedure by installing [cucim](https://docs.rapids.ai/install/) on your conda environment following the suggestions from [HEST](https://github.com/mahmoodlab/HEST?tab=readme-ov-file).

### Install RCTD (optional)
```
conda install conda-forge::r-base==4.3.3
conda install conda-forge::r-devtools
conda install bioconda::r-spacexr
conda install conda-forge::r-reticulate
R
install.packages("anndata")
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("biomaRt")
install.packages("optparse")
conda install conda-forge::r-seuratobject
```
## Usage
### 0. Configure global parameters
<!-- ```
# Path to AURORA codes
export CODE_PATH=/project/iclip/zchen/AURORA_github/code
# Path to data and output
export PROJECT_PATH=/project/iclip/zchen/AURORA_github/examples
``` -->

```
# Path to AURORA codes
export CODE_PATH=[Your AURORA code path]
# Path to data and output
export PROJECT_PATH=[Your project folder]
```
**Note**: If you download AURORA using git, then you can set `CODE_PATH` as `./AURORA/code`. AURORA supposes that all codes needed are stored under `CODE_PATH` and all files needed are under `PROJECT_PATH`.

### 1. Apply AURORA on new samples (inference-only)
AURORA requires a `metadata` file (`.csv` format) to store the name of all images to analyze, together with other information. The `metadata` file should contain the following columns:
 - `Image_name`: The name of image files to predict (with extension like .svs, .tif);
 - `Sample_name`: This will be used to index all images. So, no replicates should appear in this column. This can be the same as the image name without the extension suffix. You can also use a shorter version if the image name is too long;
 - `Bulk_sample_name`: The name of bulk samples corresponding to the image files in the bulk expressions matrix. The bulk data should be stored in a `csv` file with each row representing a gene and each column representing a sample. An example of the expression matrix can be found at [TCGA_LUAD_expression.csv](/examples/TCGA_LUAD_expression.csv)
 - `Image_folder` (optional): If images are stored in separate folders, please provide the names of these folders;
 - `Pixel_size_um` (optional): If you are using a format other than `.svs`, please specify the pixel size (in µm) in this column.

An example of the required `metadata` file is at [Inference_sample.csv](examples/Inference_samples.csv). You can download a test sample from [TCGA-86-A4P8-01A-01-TSA](https://portal.gdc.cancer.gov/files/00d4fa70-79a9-4cbc-b0c5-388323a4db9f). 

#### 1.1 Tissue segmentation
(Some parts of codes are borrowed from [CLAM](https://github.com/mahmoodlab/CLAM))

You can put all H&E images within one folder (e.g. `wsis`). We also support a format similar to the one used by [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga), where images are stored in separate folders. In this case, please provide the folder names in the metadata file (`Image_folder` column).

The preset file can be downloaded from [CLAM](https://github.com/mahmoodlab/CLAM). More information about these parameters can also be found there. We provide two examples: `example/tcga.csv` (tested on TCGA-BRCA and TCGA-KIRC) and `example/lung.csv` (suitable for TCGA-LUAD).
```
python -u $CODE_PATH/segment_image.py -p $PROJECT_PATH -d wsis -s wsis_segmentation --code_path $CODE_PATH --preset lung.csv
```

#### 1.2 H&E feature extraction
First, we rescale H&E images to the same resolution (by default, 0.5 µm/pixel). By default, we assume the images are in `.svs` format, which has a built in pixel size parameter. You can also use other image format like `.tif`. But you need to manually [specify](#1-apply-aurora-on-new-samples-inference-only) the pixel size in the `metadata` file.
```
python -u $CODE_PATH/rescale.py --inference -p $PROJECT_PATH -m Inference_samples.csv -o AURORA_interim --image_path wsis --seg_path wsis_segmentation --plot
```

Then, we extract image features using [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h). Please first create a token on huggingface following this [tutorial](https://huggingface.co/docs/hub/en/security-tokens). 
```
python -u $CODE_PATH/extract_features.py --inference -p $PROJECT_PATH -m Inference_samples.csv -o AURORA_interim --image_path wsis --seg_path wsis_segmentation -b 128 --plot --token [YOUR_HF_TOKEN]
```

**Note**: To load UNI2-h from local, please first clone [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) to your local machine following the instructions [here](https://huggingface.co/MahmoodLab/UNI2-h/tree/main?clone=true). Then use the following code to extract image features:
```
python -u $CODE_PATH/extract_features.py --inference -p $PROJECT_PATH -m Inference_samples.csv -o AURORA_interim --image_path wsis --seg_path wsis_segmentation -b 128 --plot --UNI_local_path [YOUR_LOCAL_UNI_PATH]
```

Here, [YOUR_LOCAL_UNI_PATH] should be the **absolute path** to your UNI model, for example `/home/UNI2-h`.

#### 1.3 Predict gene expressions
We provided AURORA prediction models for three cancer types: lung adenocarcinoma (LUAD), breast invasive carcinoma (BRCA), and kidney renal clear cell carcinoma (KIRC). You can download the checkpoints and model parameters at [Hugging Face](https://huggingface.co/datasets/AURORAData/AURORA_prediction) and put them under your project folder (`PROJECT_PATH`). Then, specify model parameter file using `-j` and checkpoint file using `--model_pth`. The output folder can be set by `-t`.
```
python -u $CODE_PATH/predict.py -p $PROJECT_PATH -i AURORA_interim -t AURORA_pred -j model_parameter_LUAD.json --model_pth AURORA_LUAD.pth --sample_sheet Inference_samples.csv --bulk TCGA_LUAD_expression.csv
```

The predicted gene expressions will be stored as `.pickle` files under `./AURORA_pred/pred`. In each file, the results are saved as a dictionary:
    - `exp_pred`: gene expression predictions with shape (`#spots`, `#patch levels`, `#genes`), by default, the patch levels are [3584,896,224]. You can find this information from the `.json` file of the pretrained model (e.g. `model_parameter_LUAD.json`);
    - `prop_pred`: cell type proportions in **log** scales with shape (`#spots`, `#patch levels`, `#cell types`);
    - `patches_loc_before_pad`: coordinates of the top left corner of the patches on the rescaled H&E image (which is stored in `AURORA_interim/UNI_multiscale_patches/wsis_rescaled/`).

#### 1.4 Gene expression prediction enhancement
We use [iStar](https://github.com/daviddaiweizhang/istar) to enhance the resolution of predicted gene expressions. Please use the version of iStar here, since some codes are modified to suit AURORA's data format.

To download model weights for iStar, please following [istar/download_checkpoints.sh](code/istar/download_checkpoints.sh). The checkpoints should be put in `$CODE_PATH/istar/checkpoints`.
**Note**: You can also use the checkpoints [here](https://github.com/mahmoodlab/HIPT/tree/master/HIPT_4K/Checkpoints).
```
export ISTAR_PATH=$CODE_PATH/istar/
python -u $CODE_PATH/run_istar.py -p $PROJECT_PATH -i AURORA_interim -t AURORA_pred --sample_sheet Inference_samples.csv --gene_list_file LUAD_gene.csv --celltype_file LUAD_cell_types.csv --patch_size 224 --epoch_istar 50
```

The iStar-enhanced gene expression predictions will be under `./AURORA_pred/istar_pred`.

---

If you want to train your own AURORA model, please following the following steps.
### 2. Visium data preprocessing (training-only, optional)
AURORA uses the same data structure as [HEST-1K](https://huggingface.co/datasets/MahmoodLab/hest). If your training data is from HEST-1K, then you can skip this step.

Before preprocessing, place all Visium datasets used as training samples into a single directory  (`raw_data`). Each sample should reside in its own subdirectory within `raw_data`. For each Visium sample, ensure the following three [Space Ranger](https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/output-overview) output components are present: 
 - The full resolution H&E image (`.tif` format);
 - `filtered_feature_bc_matrix.h5`;
 - The `spatial` folder.

You can download example Visium samples from [10x Genomics](https://www.10xgenomics.com/datasets?configure%5BhitsPerPage%5D=50&configure%5BmaxValuesPerFacet%5D=1000&refinementList%5Bplatform%5D%5B0%5D=Visium%20Spatial&refinementList%5Bproduct.name%5D%5B0%5D=Spatial%20Gene%20Expression&page=2), such as [BRCA1](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-0-0) and [BRCA2](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-2-1-standard-1-0-0)

An example `raw_data` looks like this:
```
raw_data
├── P2_LUAD
│   ├── B_sample_4.19H06001-3.high.ADC.CytAssist.tif
│   ├── filtered_feature_bc_matrix.h5
│   ├── spatial
│   │   ├── aligned_fiducials.jpg
│   │   ├── aligned_tissue_image.jpg
│   │   ├── cytassist_image.tiff
│   │   ├── detected_tissue_image.jpg
│   │   ├── scalefactors_json.json
│   │   ├── spatial_enrichment.csv
│   │   ├── tissue_hires_image.png
│   │   ├── tissue_lowres_image.png
│   │   └── tissue_positions_list.csv
│   └── ... (other files)
├── P3_LUAD
│   ├── filtered_feature_bc_matrix.h5
│   ├── Sample_6.15H12199-5.ADC.CytAssist.tif
│   ├── spatial
│   │   ├── aligned_fiducials.jpg
│   │   ├── aligned_tissue_image.jpg
│   │   ├── cytassist_image.tiff
│   │   ├── detected_tissue_image.jpg
│   │   ├── scalefactors_json.json
│   │   ├── spatial_enrichment.csv
│   │   ├── tissue_hires_image.png
│   │   ├── tissue_lowres_image.png
│   │   └── tissue_positions_list.csv
│   └── ... (other files)
└── ... (other samples)
```
#### 2.1 Generate a sample list
Before preprocessing, we need a `.csv` file with three columns:
 - Sample names;
 - Sample folder names under `raw_data`;
 - The name of the full resolution H&E image.

An example sample list can be found [here](/examples/Training_samples.csv).

You can generate this sample list using the following script.
```
python -u $CODE_PATH/generate_sample_list.py -p $PROJECT_PATH -r raw_data -o Sample_metadata.csv
```

#### 2.2 Preprocess data
By default, we rescale all H&E images to 0.5 µm/pixel.
```
python -u $CODE_PATH/HEST_preprocess.py -p $PROJECT_PATH -r raw_data -m Sample_metadata.csv -o processed_data
```

After preprocessing, we recommend double checking the alignment results using figures generated under `processed_data/spatial_plots`. Please **remove** the samples whose H&E image cannot align with the spot coordinates from `Sample_metadata.csv`.

### 3. Prepare training samples (training-only)
With all samples in HEST-1K data format. We now extract the image features and prepare expressions for model training.

#### 3.1 H&E feature extraction
First, we rescale H&E images to the same resolution (by default, 0.5 µm/pixel). Then, we extract image features for patches at multiple resolution (by default 3584 * 3584 pixels, 896 * 896 pixels, 224 * 224 pixels).
```
python -u $CODE_PATH/rescale.py -p $PROJECT_PATH -d processed_data -m Sample_metadata.csv -o AURORA_interim --plot
```

Then, we use [UNI-2h](https://github.com/mahmoodlab/UNI) to extract image features. Please first create a token on huggingface following this [tutorial](https://huggingface.co/datasets/MahmoodLab/hest). 
```
python -u $CODE_PATH/extract_features.py -p $PROJECT_PATH -d processed_data -m Sample_metadata.csv -o AURORA_interim -b 128 --plot --token [YOUR_HF_TOKEN]
```
**Note:** Please pass your token to `extract_features` using argument `--token`.

#### 3.2 Prepare ST expressions
```
python -u $CODE_PATH/prepare_st.py -p $PROJECT_PATH -d processed_data -m Sample_metadata.csv -o AURORA_interim
```
**Note:** By default, AURORA only predicts protein coding genes. We rely on the gene annotation from TCGA to select those genes. You can download this annotation from `example` folder (`TCGA_genetype`).

#### 3.3 Cell type deconvolution (optional)
AURORA can also predict cell type proportions. To facilitate this, we use deconvolution results as supervised signals in training. By default, we use [RCTD](https://github.com/dmcable/spacexr). You also need a single-cell RNA sequencing (scRNA-seq) data with cell type annotation as reference. For example, you can find one from [3CA](https://www.weizmann.ac.il/sites/3CA).

The scRNA-seq reference should be stored in a `.RDS` file in [SeuratV5](https://satijalab.org/seurat/) format. The cell type annotation should be stored in `meta.data$cell_type`.
```
Rscript $CODE_PATH/get_deconvolution.R -p $PROJECT_PATH -d processed_data -m Sample_metadata.csv -o AURORA_interim -r Kim2020_Lung.rds
```
**Note:** You can find the path to your `python` in your environment by
```
which python
```
The result of this command should be used as value for `--python_path`.

**Note:** You can also use other deconvolution methods. Please organize your deconvolution results as separate `.csv` files and put them under `AURORA_interim/deconvolution`. Each file should be names as `[sample_name]_prop.csv`, where sample_name should be the same as that in `Sample_metadata.csv`. The first column should the spot names and the rest of the columns each represents the proportion for one cell type.

### 4. Train AURORA model (training-only)
To train AURORA, please provide a `.csv` file containing all the names of the sample used. You can use the `Sample_metadata.csv` generated in step 1.1. If this file contains multiple columns, please specify which column contains the sample names using `--train_sample_col` (by default, the first column).

AURORA model requires a `.json` file to specify its parameters. An example can be found at `example/model_parameter.json`.
```
python -u $CODE_PATH/train.py -p $PROJECT_PATH -d processed_data -i AURORA_interim -t AURORA_output -j model_parameter.json --train_samples Sample_metadata.csv --train_sample_col 0 --plot --log 
```
**Note:** If you want to do prediction using this trained model, please following Step 1 and replace the `-j` and `--model_pth` parameter with the ones you used in your training.

(The following code is only for developers)
```
python -u $CODE_PATH/test.py -p $PROJECT_PATH -d processed_data -i AURORA_interim -t AURORA_output -j model_parameter.json --model_pth AURORA_model_weights.pth --test_samples test_samples.csv --plot
```

### 5. Finetune AURORA to predict more genes (training-only)
AURORA also supports predict more gene other than the genes used in training, such as tertiary lymphoid structures marker genes. To achieve, please put the genes to predict in a `.csv` file with only one column (`example/finetune_gene`) and run the following code: 
```
python -u $CODE_PATH/finetune.py -p $PROJECT_PATH -d processed_data -i AURORA_interim -t AURORA_output -j model_parameter.json --train_samples Sample_metadata.csv --train_sample_col 0 --model_pth AURORA_model_weights.pth --finetune_gene_list finetune_gene.csv --num_epochs 200 --plot --log 
```
