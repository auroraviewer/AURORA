<<<<<<< HEAD
# AURORA
=======
# AURORA：Architecture Unveiling through RNA Omics and Routine Histology Analysis
>>>>>>> parent of 9593886 (Update README.md)

AURORA predicts spatially resolved gene expression and cell-type composition directly
from H&E-stained whole-slide images.

For each slide you give it, AURORA writes one predicted expression vector and one
cell-type proportion vector per patch, plus figures. Everything is keyed by the
sample name you put in the sample sheet.

The model works in two stages:

| stage | patch sizes | what it does |
|---|---|---|
| low resolution | 3584 / 896 / 224 px | predicts expression and cell-type proportions across the slide |
| high resolution | 56 / 14 px | refines that prediction at fine detail |

The high-resolution stage refines the low-resolution one, so it never runs on its
own. Both stages live in a single `.pth` file.

This README has two parts. Read **[Part 1](#part-1--inference-only)** if you want to
apply an existing model to new slides. Read **[Part 2](#part-2--training)** if you
want to train a model on your own data.

---

## Examples

Every green **Example** box can be pasted as it stands. They all run from the
repository root and use the same small demo dataset, so no path needs editing.

Fetch that dataset once:

```bash
python examples/download_demo_data.py
```

It downloads about 3.7 GB into `examples/demo/`:

| | |
|---|---|
| `wsis/TCGA-44-5645-01B-04-BS4.svs` | a LUAD slide from the [Genomic Data Commons](https://portal.gdc.cancer.gov/) |
| `raw_data/BRCA1`, `raw_data/BRCA2` | two Visium samples from [10x Genomics](https://www.10xgenomics.com/products/spatial-gene-expression) |
| `Inference_sample.csv`, `Training_samples.csv` | sample sheets for them |
| `models/aurora_combined.pth` | the published LUAD weights, ready for inference |

Only one sample is used for inference (`TCGA-44-5645-01B-04-BS4`), and two for
training (`BRCA1`, `BRCA2`). The download resumes if it is interrupted, and re-running
it does nothing.

The model comes from the
[AURORA prediction weights](https://huggingface.co/datasets/AURORAData/AURORA_prediction/tree/main/trained_models/LUAD)
on Hugging Face: `AURORA_LUAD_V2.1.pth`, the full LUAD model. The script checks what it
downloaded with `aurora inspect`; if your copy is an older low-resolution-only file
instead, it prints the one command that turns it into the combined checkpoint AURORA
reads, rather than leaving you with a file inference cannot open.

UNI2-h is not part of that download. Both parts of AURORA need it, so put it in
`examples/demo/uni2h` once, from the
[UNI2-h model page](https://huggingface.co/MahmoodLab/UNI2-h). The demo blocks point
there.

> [!TIP]
> 🟩 **Example**: fetch just the model weights, without the slide or the Visium data
>
> ```bash
> python examples/download_demo_data.py --what model
> ```

---

## Requirements

| | what |
|---|---|
| hardware | a CUDA GPU is strongly recommended; CPU works but is slow |
| slides | H&E whole-slide images, usually `.svs`, in one folder |
| sample sheet | a small CSV naming your slides — see [the sample sheet](#2-write-the-sample-sheet) |
| model | one combined `.pth`, shipped in `trained_model/` |
| bulk matrix | a CSV of bulk expression, one column per sample |
| UNI2-h weights | a local copy of [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or your CUDA version
pip install -e ".[inference]"
```

For training, use `".[training]"` instead. For raw Visium preprocessing also install
HEST, which is not on PyPI:

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

---

# Part 1 — Inference only

## 1. Put your slides in one folder

```text
/path/to/my_data/
├── wsis/
│   ├── TCGA-44-5645-01B-04-BS4.svs
│   └── TCGA-55-7893-01A-01-BS1.svs
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

## 2. Write the sample sheet

One CSV drives the whole run. Two columns are required:

```csv
Sample_name,Image_name
TCGA-44-5645-01B-04-BS4,TCGA-44-5645-01B-04-BS4.svs
TCGA-55-7893-01A-01-BS1,TCGA-55-7893-01A-01-BS1.svs
```

- **`Sample_name`** is the identity of the sample. Every output file and folder uses
  it, so keep it short and unique.
- **`Image_name`** is the file name of the slide inside `--image-path`.

If each slide sits in its own sub-folder, add an `Image_folder` column naming that
sub-folder.

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
> python -m aurora run --project examples/demo --run demo_infer --samples examples/demo/Inference_sample.csv --model trained_model/AURORA_LUAD_combined.pth --uni-local-path examples/demo/uni2h --bulk examples/TCGA_LUAD_expression.csv
> ```

That runs the whole chain: segmentation, rescaling, high-resolution patch selection,
feature extraction and both prediction stages.

The command above uses the checkpoint that ships in `trained_model/`. To run with the
weights you downloaded instead, pass
`--model examples/demo/models/aurora_combined.pth`.

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
> python -m aurora run --project examples/demo --run demo_infer --samples examples/demo/Inference_sample.csv --model trained_model/AURORA_LUAD_combined.pth --stage plot --gene EPCAM IL7R
> ```

Figures go to `aurora_runs/demo_infer/05_plots/`. Plotting reloads the rescaled slide,
so keep the slides where they are until you are finished with the figures.

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

---

## Reclaiming disk space

A finished run keeps a lot of intermediate material: rescaled slides, multi-scale
embeddings and scratch files. These are often most of the disk usage. Predictions,
figures, masks, the resolved inputs, the logs and the run state are never removed.

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

# Part 2 — Training

Training uses your own spatial transcriptomics data. AURORA trains on
[HEST-1K](https://huggingface.co/datasets/MahmoodLab/hest) format data.

## 1. What you need

| | what |
|---|---|
| data | HEST-1K formatted samples, or raw Visium to convert |
| gene panel | the genes to predict, or the panel bundled in a checkpoint |
| UNI2-h | the same local copy inference uses |
| GPU | a CUDA GPU; training on CPU is impractical |

Each training sample needs a full-resolution H&E image and its spot-level expression.
If your data is already in HEST-1K format, skip to step 3.

## 2. Install the training extras

> [!TIP]
> 🟩 **Example**: the training dependencies
>
> ```bash
> pip install -e ".[training]"
> ```

## 3. Prepare raw Visium data (optional)

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

## Train

Training starts from scratch and needs no pretrained model. Describe the model you want
to train by giving the gene panel, the cell types and the normalisation factors — one
required column each, `Genes` for the panel and `gene_type` for the factors.

> [!TIP]
> 🟩 **Example**: train on the two demo samples, with a short run
>
> ```bash
> python -m aurora run --mode training --run demo_train --project examples/demo --samples examples/demo/Training_samples.csv --gene-list examples/BRCA_gene.csv --cell-types examples/BRCA_cell_types.csv --norm-factors examples/BRCA_gene_normalize_factor.csv --uni-local-path examples/demo/uni2h --no-pred-celltype --param num_epochs=10 --auto-prerequisites
> ```

If your panel is the one bundled in `trained_model/AURORA_LUAD_combined.pth`, you can
pass `--model` instead of the three files and AURORA uses that panel.

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

The cell-type phase only runs when the cell-type head is trained, so the example above
(`--no-pred-celltype`) stops after the expression epochs. **The examples shorten this to
ten epochs with `--param num_epochs=10`**; drop that to train with the defaults.

The result is a single combined checkpoint at
`aurora_runs/demo_train/checkpoints/aurora_combined.pth`.

## 5. Adjust the training run

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

## 6. Use your trained model

Your checkpoint is an ordinary combined checkpoint, so inference works exactly as in
Part 1.

> [!TIP]
> 🟩 **Example**: check the checkpoint the demo run produced
>
> ```bash
> python -m aurora validate --model examples/demo/aurora_runs/demo_train/checkpoints/aurora_combined.pth
> ```

> [!TIP]
> 🟩 **Example**: run inference with it
>
> ```bash
> python -m aurora run --project examples/demo --run demo_trained_infer --samples examples/demo/Inference_sample.csv --model examples/demo/aurora_runs/demo_train/checkpoints/aurora_combined.pth --uni-local-path examples/demo/uni2h --bulk examples/TCGA_LUAD_expression.csv
> ```

---

## Training troubleshooting

| message | what to do |
|---|---|
| `missing required package 'hest'` | HEST installs from source; see [Install](#install) |
| `missing required package 'pyvips'` | the native libvips library is needed too: `conda install -c conda-forge pyvips` |
| `raw Visium data directory … does not exist` | point `--raw-data` at the folder holding one sub-directory per sample |
| `the HEST patch dumps … are not usable` | the `preprocess` stage did not finish for that sample; re-run it with `--stage preprocess --force` |
| `requires a local UNI2-h checkpoint` | pass `--uni-local-path` |
| `… _prop.csv` not found | the cell-type head needs the `deconvolution` stage, or use `--no-pred-celltype` |
| a stage refuses because of samples | the run was started with a different `--sample` selection; re-run with the same selection or `--force` |

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

## Environment variables

| variable | effect |
|---|---|
| `AURORA_CODE_PATH` | where the original scripts live, if not the checkout's `code/` |

## What to expect from a run

| | inference | training |
|---|---|---|
| typical time | minutes per slide | hours |
| GPU memory | a few GB | more, depending on batch size |
| disk | rescaled slides and embeddings dominate | the same, plus checkpoints |
| result | predictions and figures | `checkpoints/aurora_combined.pth` |

## Citation and licence

See [LICENSE](LICENSE). Parts of the segmentation code are adapted from
[CLAM](https://github.com/mahmoodlab/CLAM) and
[TRIDENT](https://github.com/mahmoodlab/TRIDENT).
