# AURORA：Architecture Unveiling through RNA Omics and Routine Histology Analysis

AURORA predicts multi-resolution spatially resolved gene expression and cell-type composition directly
from H&E-stained whole-slide images. By default, AURORA predicts at 224 / 56 / 14 px for H&E images at 0.5 µm/px, which corresponds to 112 / 28 / 7 µm.


This README has two parts. Read **[Part 1](#part-1--inference-from-existing-checkpoints)** if you want to
**apply an existing model** to new slides. Read **[Part 2](#part-2--training-a-new-model)** if you
want to train a model on your own data.

---

## Examples

We provide toy examples for both inference and training modes with download codes. Every green **Example** box can be pasted as it stands; they all run from the repository root and use the same small demo dataset, so no path needs editing.

---

# Part 1 — Inference from existing checkpoints

## Install
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or your CUDA version
pip install -e ".[inference]"
```
> [!TIP]
> 🟩 **Example**: check the environment
>
> ```bash
> python -m aurora doctor
> ```

`doctor` reports which dependency groups are complete and lists anything missing.

> [!NOTE]
> **`hdf5plugin` and `zstandard` are optional**
>
> They only make the high-resolution HDF5 output smaller. Without them AURORA writes
> gzip, which `h5py` handles natively, so nothing else changes. Enable them with
> `pip install -e ".[compression]"` or
> `pip install -r requirements/optional-compression.txt`.
>
> If `pip install hdf5plugin` fails on your network, these work without a mirror:
>
> ```bash
> pip install --prefer-binary --retries 5 --timeout 60 hdf5plugin zstandard
> conda install -c conda-forge hdf5plugin zstandard
> ```
>
> A file written with zstd needs `hdf5plugin` on the machine that reads it; with gzip
> it does not. `--h5-compression gzip` selects gzip explicitly.

## 1. Sample preparation

```text
/path/to/my_data/
├── wsis/
│   └── TCGA-44-5645-01B-04-BS4.svs
├── my_samples.csv
└── my_bulk_expression.csv
```

The folder name is up to you; pass it with `--image-path` (default `wsis`).

> [!TIP]
> 🟩 **Example**: fetch the demo slide (about 120 MB)
>
> ```bash
> python examples/download_demo_data.py --what inference
> ```

> [!IMPORTANT]
> UNI2-h is required but not part of the download code due to model access control. Please register and download it from the [UNI2-h model page](https://huggingface.co/MahmoodLab/UNI2-h). Both parts of AURORA need it: either put it in `examples/demo/uni2h` for the examples below, or point `--uni-local-path` at your own copy.

The weights used in the example come from the
[AURORA prediction weights](https://huggingface.co/datasets/AURORAData/AURORA_prediction/tree/main/trained_models/LUAD)
on Hugging Face: `AURORA_LUAD_V2.1.pth` is the LUAD model. The download script
verifies what it fetched with `aurora inspect`.

> [!NOTE]
> **Downloading these files by hand**
>
>
>
> | file | source | name it |
> |---|---|---|
> | slide | [GDC](https://portal.gdc.cancer.gov/) file `21a90851-e349-4fc3-bdaa-cc3979529e45` | `wsis/TCGA-44-5645-01B-04-BS4.svs` |
> | weights | [AURORA prediction weights](https://huggingface.co/datasets/AURORAData/AURORA_prediction/blob/main/trained_models/LUAD/AURORA_LUAD_V2.1.pth) | `models/aurora.pth` |
> | bulk matrix | already in the repository | `examples/TCGA_LUAD_expression.csv` |
>
> The GDC link returns the slide under its file UUID, so the rename is required. The
> weights may keep their published name `AURORA_LUAD_V2.1.pth` instead — what matters is
> the path you give to `--model`, not the file name. Check either copy with
> `python -m aurora inspect --model <file>`.

> [!NOTE]
> **Requirements for applying AURORA to new samples**
> | | Description |
>|---|---|
>| hardware | a CUDA GPU is strongly recommended; CPU works but is slow |
>| slides | H&E whole-slide images, usually `.svs`, in one folder |
>| sample sheet | a small CSV naming your slides — see [the sample sheet](#2-write-the-sample-sheet) |
>| model | one combined `.pth`, shipped in `trained_model/` |
>| bulk matrix | a CSV of bulk expression, one column per sample |
>| UNI2-h weights | a local copy of [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |


## 2. Write the sample sheet

One CSV drives the whole run. Two columns are required:

```csv
Sample_name,Image_name
TCGA-44-5645-01B-04-BS4,TCGA-44-5645-01B-04-BS4.svs
```

- **`Sample_name`** is the identity of the sample. Every output file and folder uses
  it, so keep it short and unique.
- **`Image_name`** is the file name of the slide inside `--image-path`.

If each slide sits in its own sub-folder, add an `Image_folder` column naming it.

> [!TIP]
> 🟩 **Example**: the demo sheet, ready to use
>
> ```bash
> cat examples/demo/Inference_sample.csv
> ```

## 3. Run

> [!TIP]
> 🟩 **Example**: the whole inference chain on the demo slide
>
> ```bash
> python -m aurora run --project examples/demo --run demo_infer --samples examples/demo/Inference_sample.csv --model examples/demo/models/aurora.pth --uni-local-path examples/demo/uni2h --bulk examples/TCGA_LUAD_expression.csv
> ```

That runs the whole chain: segmentation, rescaling, high-resolution patch selection,
feature extraction and both prediction stages.

The command above uses the checkpoint that ships in `trained_model/`. To run with the
weights you downloaded instead, pass
`--model examples/demo/models/aurora.pth`.

`--samples`, `--model`, `--bulk` and `--uni-local-path` are resolved relative to where
you run the command, while `--image-path` follows `--project`.

> [!TIP]
> 🟩 **Example**: see the plan without running it
>
> ```bash
> python -m aurora run --project examples/demo --run demo_infer --samples examples/demo/Inference_sample.csv --dry-run
> ```

Every stage prints its command and its log location. If a stage fails, fix the cause
and run the same command again: completed stages are reused.

## 4. Where the results are

Everything lands under `<project>/aurora_runs/<run>/`:

```text
aurora_runs/demo_infer/
├── 03_predictions/
│   ├── lowres/pred/TCGA-44-5645-01B-04-BS4-pred.pickle
│   └── highres/pred/TCGA-44-5645-01B-04-BS4-highres-pred.h5
├── 05_plots/                      figures, once you plot
├── config/                        the exact inputs this run used
├── logs/                          one log per stage
└── state/                         what has completed
```

The low-resolution prediction is a pickle holding, per patch, the predicted expression
vector and the cell-type proportions. The high-resolution prediction is an HDF5 file by
default; pass `--no-uint8-output` to get a pickle instead.

> [!TIP]
> 🟩 **Example**: check what the demo run has completed
>
> ```bash
> python -m aurora status --project examples/demo --run demo_infer
> ```

## 5. Plot genes and cell types

Figures are a separate step, so you can plot only what you need:

| option | default | effect |
|---|---|---|
| `--gene` | — | genes to plot, space separated |
| `--celltype` | — | cell types to plot |
| `--plot-predictions` | `lowres` | pass `highres` for the refined prediction |
| `--plot-resolution` | `224` | patch level to plot |
| `--plot-downscale` | `7` | figure pixels = image pixels ÷ N (maximum 7) |

At least one gene or cell type is required.

> [!TIP]
> 🟩 **Example**: plot two genes and two cell types of the demo slide
>
> ```bash
> # Low resolution
> python -m aurora run --project examples/demo --run demo_infer --samples examples/demo/Inference_sample.csv --model examples/demo/models/aurora.pth --stage plot --gene EPCAM IL7R
> # High resolution
> python -m aurora run --project examples/demo --run demo_infer --samples examples/demo/Inference_sample.csv --model examples/demo/models/aurora.pth --stage plot --gene EPCAM IL7R --plot-predictions highres --plot-resolution 14
> ```

Figures go to `aurora_runs/demo_infer/05_plots/`. Plotting reloads the rescaled slide,
so keep the slides where they are until you are finished with the figures.

> [!IMPORTANT]
> Resolution and key correspondence for 0.5 µm/px images:
> | Resolution | Key |
> | --- | --- |
> | 112 µm | 224 |
> | 28 µm | 56 |
> | 7 µm | 14 |

## 6. Useful options

| option | default | effect |
|---|---|---|
| `--device` | `auto` | `cpu`, `cuda` or `cuda:0` |
| `--batch-size` | `128` | feature-extraction batch size |
| `--sample` | all | restrict the run to some samples |
| `--force` | off | re-run stages that already completed |
| `--keep-going` | off | continue after a stage fails |
| `--quiet` | off | do not stream stage output |

> [!TIP]
> 🟩 **Example**: list every stage of the inference chain
>
> ```bash
> python -m aurora stages --mode inference
> ```

## Reclaiming disk space

A finished run keeps a lot of intermediate material: rescaled slides, multi-scale
embeddings and scratch files. These often account for most of the disk usage, while
predictions, figures, masks, the resolved inputs, the logs and the run state are never removed.

> [!TIP]
> 🟩 **Example**: report what the demo run could free, then apply the cleanup
>
> ```bash
> python -m aurora clean --project examples/demo --run demo_infer
> python -m aurora clean --project examples/demo --run demo_infer --apply
> ```

The first command only reports; `--apply` removes the scratch tier. `--deep` also
removes the rescaled slides and embeddings, after which plotting or re-predicting means
running those stages again.

---

# Part 2 — Training a new model

Training uses your own spatial transcriptomics data, supplied in
[HEST-1K](https://huggingface.co/datasets/MahmoodLab/hest) format.

## Install

For training, install `".[training]"` instead. For raw Visium preprocessing, also
install HEST, which is not on PyPI:

```bash
git clone https://github.com/mahmoodlab/HEST.git
pip install -e HEST
pip install pyvips && conda install -c conda-forge pyvips
```

> [!TIP]
> 🟩 **Example**: check the environment
>
> ```bash
> python -m aurora doctor
> ```

`doctor` reports which dependency groups are complete and lists anything missing.

> [!NOTE]
> **`hdf5plugin` and `zstandard` are optional here too.** Training works without them;
> the high-resolution HDF5 output is simply written with gzip. Install them with
> `pip install -e ".[compression]"` when you want the smaller files, and see
> [the inference install note](#install) if `pip` cannot reach its index.

Each training sample needs a full-resolution H&E image and its spot-level expression.
If your data is already in HEST-1K format, skip to step 3.

## 1. Install the training extras

> [!TIP]
> 🟩 **Example**: the training dependencies
>
> ```bash
> pip install -e ".[training]"
> ```

## 2. Prepare raw Visium data (optional)

Put each Visium sample in its own folder under `raw_data/` **in the project
directory**, with the H&E image, `filtered_feature_bc_matrix.h5` and the `spatial/`
folder:

```text
/path/to/my_project/
├── raw_data/
│   ├── BRCA1/  BRCA1.tif  filtered_feature_bc_matrix.h5  spatial/
│   └── BRCA2/  …
├── Training_samples.csv
└── aurora_runs/            created by AURORA
```

The training sheet uses different column names from the inference sheet:

```csv
Sample,Dir_name,Image_name
BRCA1,BRCA1,V1_Breast_Cancer_Block_A_Section_1_image.tif
BRCA2,BRCA2,V1_Breast_Cancer_Block_A_Section_2_image.tif
```

> [!TIP]
> 🟩 **Example**: fetch the two demo Visium samples (about 3.5 GB) and their sheet
>
> ```bash
> python examples/download_demo_data.py --what training
> ```

> [!NOTE]
> **Downloading these files by hand**
>
> Each Visium sample needs three files from 10x Genomics, published under
> [BRCA1](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0) and [BRCA2](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-2-1-standard-1-1-0):
>
> | downloaded file | rename it to |
> |---|---|
> | `<sample>_image.tif` | keep the name — `Image_name` in the sheet has to match it |
> | `<sample>_filtered_feature_bc_matrix.h5` | `filtered_feature_bc_matrix.h5` |
> | `<sample>_spatial.tar.gz` | unpack it, then rename `…_spatial/` to `spatial/` |
>
> Put all three in `raw_data/<Sample>/`, for example `raw_data/BRCA1/`. The preprocessing
> stage opens those exact names, so the renames are what make hand-downloaded data work.

> [!TIP]
> 🟩 **Example**: convert them to HEST-1K format
>
> ```bash
> python -m aurora run --mode training --run demo_train --project examples/demo --samples examples/demo/Training_samples.csv --stage preprocess --auto-prerequisites
> ```

This stage needs HEST and `pyvips`. If your data is already in HEST-1K format, skip it.

Before going further, open `processed_data/spatial_plots` and check that the H&E image
lines up with the spot coordinates. Drop any sample that does not align: misaligned
samples spoil the whole training set.

## 3. Train

>[!Note]
> **Requirements for training a new AURORA model**
>| | Description |
>|---|---|
>| data | HEST-1K formatted samples, or raw Visium to convert |
>| gene panel | the genes to predict, or the panel bundled in a checkpoint |
>| UNI2-h | the same local copy inference uses |
>| GPU | a CUDA GPU; training on CPU is impractical |


Training starts from scratch and needs no pretrained model. You describe the model to
train by supplying three files: the gene panel (a `Genes` column), the cell types (one
name per line) and the normalisation factors (a `gene_type` column).

> [!TIP]
> 🟩 **Example**: train on the two demo samples, with a short run
>
> ```bash
> python -m aurora run --mode training --run demo_train --project examples/demo --samples examples/demo/Training_samples.csv --gene-list examples/BRCA_gene.csv --cell-types examples/BRCA_cell_types.csv --norm-factors examples/BRCA_gene_normalize_factor.csv --uni-local-path examples/demo/uni2h --no-pred-celltype --param num_epochs=10 --auto-prerequisites
> ```
>
> *Note*: the cell-type phase only runs when the cell-type head is trained, so the
> example above (`--no-pred-celltype`) stops after the expression epochs. It also
> shortens training to ten epochs with `--param num_epochs=10`; drop that flag to
> train with the defaults.

The result is a single combined checkpoint at
`aurora_runs/demo_train/checkpoints/aurora.pth`.


If your panel is the one bundled in `trained_model/AURORA_LUAD_combined.pth`, pass
`--model` instead of the three files and AURORA will use that panel.

The chain is:

```text
rescale → features_lowres → prepare_st → smooth_cnts → train_lowres
        → features_highres → prepare_highres_training → train_highres
        → combine_checkpoints
```

A full training run is long: expect hours, with high-resolution feature extraction
usually the slowest step. The default budget is two phases per resolution:

| | expression | cell type | total |
|---|---|---|---|
| low resolution | 400 epochs | 200 epochs | 600 |
| high resolution | 100 epochs | 100 epochs | 200 |

## 4. Use your trained model

Your checkpoint is an ordinary combined checkpoint, so inference works exactly as in
[Part 1](#part-1--inference-from-existing-checkpoints).

## Adjust the training run

Epochs, batch size, learning rate and the rest are ordinary settings you can change on
the command line.

> [!TIP]
> 🟩 **Example**: a longer run with a different batch size and learning rate
>
> ```bash
> python -m aurora run --mode training --run demo_train --project examples/demo --samples examples/demo/Training_samples.csv --param num_epochs=30 batch_size=8 lr=0.0005 --auto-prerequisites
> ```

| what you want | how |
|---|---|
| more or fewer epochs | `--param num_epochs=30` |
| expression only, no cell-type head | `--no-pred-celltype` |
| a different gene panel | `--gene-list my_genes.csv` |
| add genes to a trained model | `--finetune-gene-list extra_genes.csv` with `--stage finetune` |
| reset the settings to their defaults | `--refresh-params` |

Settings you pass with `--param` are written into the run, so they persist for the
following stages and later runs.

> [!TIP]
> 🟩 **Example**: see every setting the demo run will use
>
> ```bash
> python -m aurora params --project examples/demo --run demo_train
> ```

Decide on the cell-type head at the start: it changes the shape of the model, so
switching it later means retraining. Training the cell-type head also needs per-spot
cell-type proportions in `processed_data/deconvolution`, which come from the
`deconvolution` stage (RCTD, which requires R). The demo above trains expression only.

## Optional: Cell-type proportions with RCTD

Everything above trains the expression head only. AURORA can also train the **cell-type
head**, which predicts how much of each cell type is present at every spot. That head
learns from per-spot proportions, and AURORA estimates those with
[RCTD](https://github.com/dmcable/spacexr) from a labelled single-cell reference.

RCTD is R code, so it is the one part of AURORA that `pip install` cannot set up.

### 1. Install R and the RCTD packages

Install R 4.2 or newer, then run this in `R`:

```r
install.packages(c("BiocManager", "Matrix", "reticulate", "stringr", "optparse",
                   "anndata", "remotes"))
BiocManager::install("biomaRt", ask = FALSE)
remotes::install_github("dmcable/spacexr", build_vignettes = FALSE)
```

`code/get_deconvolution.R` loads `spacexr`, `anndata`, `stringr`, `Matrix`, `biomaRt`,
`reticulate` and `optparse`. AURORA starts it with `Rscript`, so `Rscript` must be on
`PATH` for the same user that runs `aurora`.

### 2. Provide a single-cell reference

RCTD needs a Seurat object saved as `.rds` whose metadata has a `cell_type` column. Those
names are the cell types the head learns to predict. The stage opens the file **relative
to the run directory**, so either copy it into the run, or reach outside it with a
relative path:

```text
/path/to/my_project/
├── reference.rds                  the labelled single-cell reference
├── raw_data/                      one folder per Visium sample
└── aurora_runs/demo_ct/           the run, created by AURORA
```

From inside a run, `../../reference.rds` reaches the project directory.

### 3. Estimate the proportions

```bash
python -m aurora run --mode training --run demo_ct --project /path/to/my_project --samples /path/to/my_project/Training_samples.csv --scrna-reference ../../reference.rds --stage deconvolution --auto-prerequisites
```

`--auto-prerequisites` converts raw Visium data first if that has not happened yet. The
stage writes into `aurora_runs/demo_ct/02_features/deconvolution/`:

| file | contents |
|---|---|
| `cell_types.csv` | one cell type per line — the classes the head predicts |
| `<sample>_prop.csv` | per-spot proportions, one row per barcode |
| `<sample>_prop.pdf` | QC plots, one page per cell type |
| `RCTD_ref.rds` | the reference RCTD built, reused by later runs |

Look through the PDFs before training: a cell type that is flat everywhere usually means
the reference does not match the tissue section.

### 4. Train both heads

Leave `--pred-celltype` on — it is the default — and point `--cell-types` at the list the
stage just wrote:

```bash
python -m aurora run --mode training --run demo_ct --project /path/to/my_project --samples /path/to/my_project/Training_samples.csv --gene-list examples/BRCA_gene.csv --cell-types /path/to/my_project/aurora_runs/demo_ct/02_features/deconvolution/cell_types.csv --norm-factors examples/BRCA_gene_normalize_factor.csv --auto-prerequisites
```

The run trains the expression head for `exp_epochs` and the cell-type head for the rest:
400 + 200 epochs at low resolution, 100 + 100 at high resolution. Both heads end up in
the same checkpoint, `aurora_runs/demo_ct/checkpoints/aurora.pth`, and inference then
reports cell-type proportions alongside expression.

Decide this before you start. The head changes the shape of the model, so a checkpoint
trained with `--no-pred-celltype` cannot gain one later without retraining.

The cell types are fixed by the reference: to predict a different set, build a reference
whose `cell_type` column uses those names. The list you pass with `--cell-types` must
match the columns of the `<sample>_prop.csv` files.

---

# Reference

## Command line

```text
aurora run             execute pipeline stages
aurora status          show what a run has completed
aurora stages          list stages and what they need
aurora params          show a training run's settings
aurora inspect         describe a combined checkpoint
aurora validate        validate a combined checkpoint
aurora clean           remove a run's intermediate files, keeping results
aurora doctor          environment and dependency report
aurora hest-setup      register HEST's segmentation weights
aurora migrate-layout  move an old output tree into the run layout
```

`python -m aurora <command>` is equivalent, and `--help` after any command lists its
options.

## Common options

| flag | default | effect |
|---|---|---|
| `--project`, `-p` | `.` | directory holding your data |
| `--run` | `run` | run name; files go to `<project>/aurora_runs/<run>` |
| `--run-root` | — | an explicit run directory instead |
| `--model` | discovered | combined checkpoint |
| `--samples` | recorded | sample sheet CSV |
| `--mode` | `inference` | `inference` or `training` |
| `--stage` | whole chain | run only these stages |
| `--all` | — | run the whole chain for the mode |
| `--auto-prerequisites` | off | run missing earlier stages instead of refusing |
| `--dry-run` | off | print the commands without running them |
| `--force` | off | re-run stages that already completed |

## What to expect from a run

| | inference | training |
|---|---|---|
| typical time | minutes per slide | hours |
| GPU memory | a few GB | more, depending on batch size |
| disk | rescaled slides and embeddings dominate | the same, plus checkpoints |
| result | predictions and figures | `checkpoints/aurora.pth` |

## Downloading all examples at once
Fetch the example datasets once:

```bash
python examples/download_demo_data.py
```

It downloads about 3.7 GB into `examples/demo/`:

| File | Stage | Description |
|---|---|---|
| `wsis/TCGA-44-5645-01B-04-BS4.svs` | Inference | a LUAD slide from the [Genomic Data Commons](https://portal.gdc.cancer.gov/) |
| `models/aurora.pth` | Inference | the published LUAD weights, ready for inference |
| `Inference_sample.csv` | Inference | the sample sheet for the inference example |
| `raw_data/BRCA1`, `raw_data/BRCA2` | Training | two Visium samples from [10x Genomics](https://www.10xgenomics.com/products/spatial-gene-expression) |
| `Training_samples.csv` | Training | the sample sheet for the training example |


## Citation and licence

See [LICENSE](LICENSE). Parts of the segmentation code are adapted from
[CLAM](https://github.com/mahmoodlab/CLAM) and
[TRIDENT](https://github.com/mahmoodlab/TRIDENT).
