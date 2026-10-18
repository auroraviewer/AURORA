# AURORA — Training-pipeline reconnaissance report

**Scope.** Static read-only analysis of `C:\Edward_Chen\appeal\AURORA-main`.
No existing file was modified; this report is the only new file.

Files analysed in full (all in `code/`):

| # | File | Role |
|---|------|------|
| 1 | `train.py` | Low-resolution (multi-scale 3584/896/224) AURORA training |
| 2 | `test.py` | Low-resolution evaluation of a trained checkpoint |
| 3 | `finetune.py` | Re-decoder fine-tuning on a new gene panel |
| 4 | `prepare_st.py` | ST pseudo-bulk / gene-panel selection (scanpy) |
| 5 | `smooth_cnt.py` | Spatial smoothing of spot counts (squidpy) |
| 6 | `generate_sample_list.py` | Build `Sample_metadata.csv` from a raw-data tree |
| 7 | `HEST_preprocess.py` | HEST `VisiumReader` preprocessing of raw Visium folders |
| 8 | `datasets.py` | `torch.utils.data.Dataset` wrappers (library, no CLI) |
| 9 | `train_utils.py` | Data loading / loss / projection utilities (library, no CLI) |

Convention below: **`main_dir` = `args.project_path`** (default `'./'`). Every
path is quoted exactly as written in the source (Python f-string / `os.path.join`
expression), so `{main_dir}` means the literal variable in that expression.

---

## 1. Per-script reference

### 1.1 `code/train.py`

**Role.** Trains `model.AURORA` on pre-computed multi-scale UNI embeddings plus
pseudo-bulk/deconvolution targets. Two-phase: gene-expression decoder for
`epoch < exp_epochs`, then (at `epoch == exp_epochs`) the gene modules are frozen
and only `model.celltype_decoder` is optimised with `lr_prop`.

#### 1.1.1 argparse (verbatim flag / short / type / default / help)

| Flag | Short | Type | Default | Help |
|---|---|---|---|---|
| `--project_path` | `-p` | `str` | `'./'` | Main directory path |
| `--interim_path` | `-i` | `str` | `'AURORA_interim'` | Interim data directory |
| `--train_path` | `-t` | `str` | `'AURORA_train'` | Directory to store training data |
| `--processed_data_path` | `-d` | `str` | `'processed_data'` | Processed data directory path |
| `--plot` | — | `store_true` | `False` | Plot intermediate results for checking |
| `--log` | — | `store_true` | `False` | Enable logging to TensorBoard |
| `--parameter_json` | `-j` | `str` | `'parameters.json'` | JSON file with parameters |
| `--train_samples` | — | `str` | `'Sample_metadata.csv'` | CSV file with training sample metadata |
| `--train_sample_col` | — | `int` | `0` | Column index for training sample IDs in the metadata CSV |
| `--smooth_cnts` | — | `store_true` | `False` | Flag to indicate whether to smooth counts before training |

Commented out (lines 18–19), *not* accepted at runtime:
`--patch_sizes` (`nargs='+'`, `int`, `[3584, 896, 224]`) and `--metadata_path`
(`-m`, `str`, `'Sample_metadata.csv'`). Patch sizes are instead read from the
parameter JSON (`parameters['patch_sizes']`).
`parser.description` is the copy-pasted rescale description
(`'Rescale H&E images to target pixel size and adjust ST spot locations accordingly.'`).

Hyper-parameters come from `parameters.json` via `parameters.get(...)` with
defaults: `img_emb_dim=1536`, `batch_size=4`, `num_workers=4`,
`patch_sizes=[3584, 896, 224]`, `mse_weight=10`, `prop_weight=0.1`,
`focal_gamma=0.5`, `num_epochs=600`, `exp_epochs=400`, `pixel_size=0.5`,
`lr=1e-3`, `lr_prop=1e-3`, `weight_decay=1e-5`, `stepLR_step_exp=200`,
`stepLR_gamma_exp=0.1`, `stepLR_step_prop=50`, `stepLR_gamma_prop=0.5`,
`bidirectional=False`, `use_bulk=True`, `pred_celltype=True`,
`use_context_loss=True`, `retrain_cellprop_only=False`,
`gene_list_file='AURORA_interim/Gene_to_predict.csv'`,
`gene_norm_factor_file='AURORA_interim/Gene_normalize_factor.csv'`,
`celltype_file='AURORA_interim/deconvolution/cell_types.csv'`.

#### 1.1.2 Filesystem locations READ

| Location | Where |
|---|---|
| `f"{main_dir}/{args.parameter_json}"` | line 31 (`json.load`) |
| `f"{main_dir}/{args.train_path}/processed_pkl"` | line 40/42 — `dataset_dir` (dir listing, `.pkl` names) |
| `f"{main_dir}/{args.train_path}/processed_pkl_smooth"` | line 42 (when `--smooth_cnts`) |
| `f"{main_dir}/{parameters.get('gene_list_file',"AURORA_interim/Gene_to_predict.csv")}"` | line 48 (`pd.read_csv`) |
| `f"{main_dir}/{parameters.get('gene_norm_factor_file','AURORA_interim/Gene_normalize_factor.csv')}"` | line 52 (`pd.read_csv(..., index_col=0).loc[gene_list,'0']`) |
| `f"{main_dir}/{args.train_samples}"` | line 80 (`train_samples_path`, `pd.read_csv`) |
| `f"{main_dir}/{parameters.get('celltype_file', "AURORA_interim/deconvolution/cell_types.csv")}"` | line 81 (`celltype_path`, `pd.read_csv(header=None)`) |
| `f"{main_dir}/{args.train_path}/checkpoints/AURORA_{exp_epochs}.pth"` | line 127 (only if `retrain_cellprop_only`) |
| per-sample `.pkl` under `dataset_dir` | `train_utils.load_data(..., from_file=True, file_dir=dataset_dir)` |

Transitively read by `train_utils.load_data(...)` (see §1.9): per-sample
`metadata/{dataset}.json`, `UNI_multiscale_patches/wsis_rescaled/{dataset}_rescaled.tif`
and `_mask_rescaled.tif`, `wsis_rescaled/{dataset}_loc_rescaled.npy`,
`wsis_rescaled/{dataset}_barcode.pickle`, `locs/{dataset}-bars.pickle`,
`{prop_folder}/{dataset}_prop.csv`, `emb_{patch_size}/{dataset}.npz`,
`cnts[-smooth]/{dataset}-cnts.tsv`, `bulk/{dataset}-bulk.tsv`,
`bulk/{dataset}-max.tsv`.

#### 1.1.3 Filesystem locations WRITTEN / created

| Location | Where |
|---|---|
| `f"{main_dir}/{args.train_path}/train_log"` | line 123 — `SummaryWriter(...)` (only with `--log`) |
| `f"{main_dir}/{args.train_path}/checkpoints"` | line 125 — `os.makedirs(..., exist_ok=True)` |
| `f"{main_dir}/{args.train_path}/checkpoints/AURORA_{epoch}.pth"` | lines 350–351 — `torch.save(model, filename)` |
| `f"{main_dir}/{args.train_path}/AURORA_model_weights.pth"` | line 360 — `torch.save(state_dict, ...)` |
| `f"{main_dir}/{args.train_path}/AURORA_model_weights_smooth.pth"` | line 362 — same, when `--smooth_cnts` |
| `f"{file_dir}/{dataset.replace('.h5','')}.pkl"` | `train_utils.load_data` (line 440) |
| `f"{main_dir}/{train_dir}/image_check/"` + `.../{dataset}_{patch_size}patch.png` | `train_utils.load_data` (lines 211, 317), only when `--plot` |

#### 1.1.4 Checkpoint save / load

* **Whole-model pickle (periodic).** `epoch % 100 == 0` →
  `filename = f"{main_dir}/{args.train_path}/checkpoints/AURORA_{epoch}.pth"`,
  `torch.save(model, filename)`. The saved object is the **`nn.Module` itself**
  (not a `state_dict`), so the checkpoint contains pickled class references to
  `model.AURORA`. Only `model` is stored — **no epoch, no optimizer, no
  scheduler, no RNG state, no loss history**. With the default
  `num_epochs=600`, checkpoints are written at epochs 0, 100, 200, 300, 400, 500.
* **`state_dict` (final).** After the loop: `state_dict = model.state_dict()`;
  `torch.save(state_dict, f"{main_dir}/{args.train_path}/AURORA_model_weights.pth")`
  (or `..._model_weights_smooth.pth` with `--smooth_cnts`). Plain tensor dict,
  no metadata.
* **Load.** Only one load exists, gated by `parameters['retrain_cellprop_only']`:
  `model = torch.load(f"{main_dir}/{args.train_path}/checkpoints/AURORA_{exp_epochs}.pth")`
  (line 127) — whole-module load with **no `map_location`** and **no
  `weights_only=`**; the file must exist (i.e. `exp_epochs` must be a multiple of
  100 for the periodic save to have produced it, default 400 → OK).
* The commented-out block at lines 356–357 shows an older `AURORA.pth` whole-model
  save that is no longer executed.

#### 1.1.5 Behavioural notes / defects

* `writer` is used unconditionally inside the epoch loop (lines 308–347) and
  `writer.close()` at line 354, but `writer` is only created when `--log` is
  given → **`NameError: name 'writer' is not defined` at the end of epoch 0
  unless `--log` is passed.**
* `for para in model.celltype_decoder.parameters(): para.requires_grad = False`
  (lines 132–133) runs unconditionally, even when `pred_celltype=False`.
* Optimizer/scheduler are re-created at `epoch == exp_epochs` on
  `model.celltype_decoder.parameters()` only, so Adam state from phase 1 is
  discarded (intentional but undocumented).
* `focal_gamma += 0.5` every 50 epochs after `exp_epochs` — the JSON value is
  mutated in place.
* Inference/eval is not run at all: no validation split, no early stopping, no
  best-model tracking (the `best_val_loss` code is commented out).
* Nested same-type quotes inside f-strings (`parameters.get('gene_list_file',"AURORA_interim/Gene_to_predict.csv")`)
  at lines 48 and 81 require **PEP 701 (Python ≥ 3.12)**.

---

### 1.2 `code/test.py`

**Role.** Developer-only evaluator (header comment: *"This script is only for
developers"*). Loads weights, runs the model sample-by-sample (no batching), and
dumps per-sample prediction pickles.

#### 1.2.1 argparse

| Flag | Short | Type | Default | Help |
|---|---|---|---|---|
| `--project_path` | `-p` | `str` | `'./'` | Main directory path |
| `--interim_path` | `-i` | `str` | `'AURORA_interim'` | Interim data directory |
| `--train_path` | `-t` | `str` | `'AURORA_train'` | Directory to store training data |
| `--processed_data_path` | `-d` | `str` | `'processed_data'` | Processed data directory path |
| `--plot` | — | `store_true` | `False` | Plot intermediate results for checking |
| `--parameter_json` | `-j` | `str` | `'parameters.json'` | JSON file with parameters |
| `--test_samples` | — | `str` | **required** | Test sample CSV file name |
| `--model_pth` | — | `str` | `'AURORA_model_weights.pth'` | Path to the pretrained model file |
| `--output_dir` | — | `str` | `'test_pred'` | Directory to save test predictions |
| `--smooth_cnts` | — | `store_true` | `False` | Use smoothed counts for training and testing |
| `--barcode_dir` | — | `str` | `None` | Directory for barcode location pickles |
| `--patch_folder_name` | — | `str` | `'UNI_multiscale_patches'` | Folder name for patch embeddings |
| `--cnts_smooth_folder_name` | — | `str` | `'cnts_smooth'` | Folder name for smoothed counts |
| `--bulk_folder_name` | — | `str` | `'bulk'` | Folder name for bulk data |
| `--prop_folder_name` | — | `str` | `'deconvolution'` | Folder name for deconvolution data |

Commented out: `--patch_sizes`, `--metadata_path`, `--log` (lines 16–20).
`description='AURORA Test Script'`.

#### 1.2.2 Filesystem locations READ

| Location | Where |
|---|---|
| `f"{main_dir}/{args.parameter_json}"` | line 35 |
| `f"{main_dir}/{args.train_path}/processed_pkl"` (dir listing) | line 42/49 |
| `f"{main_dir}/{args.train_path}/processed_pkl_smooth"` (dir listing) | line 45/49 |
| `f"{main_dir}/{parameters.get('gene_list_file',"AURORA_interim/Gene_to_predict.csv")}"` | line 54 |
| `f"{main_dir}/{parameters.get('gene_norm_factor_file','AURORA_interim/Gene_normalize_factor.csv')}"` | line 58 |
| `f"{main_dir}/{parameters.get('celltype_file', "AURORA_interim/deconvolution/cell_types.csv")}"` | line 70 |
| `f"{main_dir}/{args.test_samples}"` | line 76 (`test_samples_path`; column **`'sample'`**) |
| `f"{main_dir}/{args.train_path}/{args.model_pth}"` | line 86 (`torch.load`) |
| per-sample `.pkl` in `dataset_dir` | `train_utils.load_data(..., from_file=True/False, file_dir=dataset_dir)` |
| plus everything `train_utils.load_data` reads (§1.9), with `load_cell_type=False` | lines 94–110 |

Note `dataset_test = [x.replace(".h5","") for x in dataset_test]` (line 78) and
the CSV column used is `'sample'` (unlike `train.py`'s index-based
`--train_sample_col`).

#### 1.2.3 Filesystem locations WRITTEN / created

| Location | Where |
|---|---|
| `f"{main_dir}/{args.train_path}/processed_pkl"` | line 43 — `os.makedirs(..., exist_ok=True)` |
| `f"{main_dir}/{args.train_path}/processed_pkl_smooth"` | line 46 — `os.makedirs(..., exist_ok=True)` |
| `f"{main_dir}/{args.train_path}/{args.output_dir}"` | line 90 — `os.makedirs(..., exist_ok=True)` |
| `f"{main_dir}/{args.train_path}/{args.output_dir}/{dataset_test[index]}-spotpred.pickle"` | line 191 (`utils.save_pickle(spot_preds, ...)`) |
| `f"{main_dir}/{args.train_path}/{args.output_dir}/{dataset_test[index]}-majorprop.pickle"` | line 193 (only if `pred_celltype`) |
| `f"{main_dir}/{args.train_path}/{args.output_dir}/{dataset_test[index]}-pred.pickle"` | line 212/214 — `{'output', 'major_prop_pred'}` or `{'output'}` |
| `f"{file_dir}/{dataset.replace('.h5','')}.pkl"` | side effect of `load_data` (cache) |
| `f"{main_dir}/{train_dir}/image_check/...png"` | side effect of `load_data` (only with `--plot`) |

#### 1.2.4 Checkpoint load

```python
model = AURORA(input_size, output_size, bulk_size=len(hvg_list),
               celltype_num=len(major_cell_type),
               device=device, bidirectional=bidirectional,
               use_bulk=use_bulk, pred_celltype=pred_celltype)
model.to(device)
state_dict = torch.load(f"{main_dir}/{args.train_path}/{args.model_pth}",
                        weights_only=False, map_location=device)
model.load_state_dict(state_dict)
```
Lines 81–87. It therefore expects the **`state_dict`** file produced by
`train.py` (`<train_path>/AURORA_model_weights.pth`), not the periodic
whole-module `.pth` files. `weights_only=False` is explicit; `map_location=device`
makes the load device-agnostic. No save of any kind.

---

### 1.3 `code/finetune.py`

**Role.** Continues training from a pre-trained weight file with a *different,
smaller* gene panel (`--finetune_gene_list`) and a freshly initialised decoder
(`AURORA(..., finetune=True)` → `model.new_decoder`). Non-matching weights are
silently dropped by a name+shape filter.

#### 1.3.1 argparse

| Flag | Short | Type | Default | Help |
|---|---|---|---|---|
| `--project_path` | `-p` | `str` | `'./'` | Main directory path |
| `--interim_path` | `-i` | `str` | `'AURORA_interim'` | Interim data directory |
| `--train_path` | `-t` | `str` | `'AURORA_train'` | Directory to store training data |
| `--processed_data_path` | `-d` | `str` | `'processed_data'` | Processed data directory path |
| `--plot` | — | `store_true` | `False` | Plot intermediate results for checking |
| `--log` | — | `store_true` | `False` | Enable logging to TensorBoard |
| `--parameter_json` | `-j` | `str` | `'parameters.json'` | JSON file with parameters |
| `--train_samples` | — | `str` | `'Sample_metadata.csv'` | CSV file with training sample metadata |
| `--train_sample_col` | — | `int` | `0` | Column index for training sample IDs in the metadata CSV |
| `--model_pth` | — | `str` | `'AURORA_model_weights.pth'` | Path to the pretrained model file |
| `--finetune_gene_list` | — | `str` | `None` | Path to the gene list for finetuning |
| `--num_epochs` | — | `int` | `200` | Number of training epochs |

Commented out: `--patch_sizes`, `--metadata_path` (lines 18–19).
`num_epochs = exp_epochs + args.num_epochs` (line 62), `start_epoch = exp_epochs`
(line 125) → the loop runs `exp_epochs … exp_epochs+200`, but the phase switch
`if epoch == exp_epochs:` triggers on the **first** iteration (gene modules
frozen, only `new_decoder` trained). It shares `train.py`'s parameter-JSON
defaults (no `retrain_cellprop_only`).

#### 1.3.2 Filesystem locations READ

| Location | Where |
|---|---|
| `f"{main_dir}/{args.parameter_json}"` | line 33 |
| `f"{main_dir}/{args.train_path}/processed_pkl_finetune"` (dir listing) | line 39/41 |
| `f"{main_dir}/{parameters.get('gene_list_file',"AURORA_interim/Gene_to_predict.csv")}"` | line 45 |
| `f"{main_dir}/{parameters.get('gene_norm_factor_file','AURORA_interim/Gene_normalize_factor.csv')}"` | line 49 |
| `f"{main_dir}/{args.finetune_gene_list}"` | line 51 (`pd.read_csv(header=None).iloc[:,0]`) — overrides `gene_list` |
| `f"{main_dir}/{args.train_samples}"` | line 80 (`train_samples_path`) |
| `f"{main_dir}/{parameters.get('celltype_file', "AURORA_interim/deconvolution/cell_types.csv")}"` | line 81 |
| `f"{main_dir}/{args.model_pth}"` | line 124 (`torch.load`) — **resolved against `main_dir`, not `train_path`** |
| per-sample `.pkl` in `f"{main_dir}/{args.train_path}/processed_pkl_finetune"` | `load_data(from_file=True, file_dir=dataset_dir)` |
| plus everything `train_utils.load_data` reads (§1.9) | lines 90–101 |

#### 1.3.3 Filesystem locations WRITTEN / created

| Location | Where |
|---|---|
| `f"{main_dir}/{args.train_path}/train_log"` | line 119 — `SummaryWriter` (only with `--log`) |
| `f"{main_dir}/{args.train_path}/checkpoints"` | line 121 — `os.makedirs(..., exist_ok=True)` |
| `f"{main_dir}/{args.train_path}/checkpoints/AURORA_finetune_{epoch}.pth"` | lines 284–285 — `torch.save(model, filename)` |
| `f"{main_dir}/{args.train_path}/AURORA_finetune_model_weights.pth"` | line 293 — `torch.save(state_dict, ...)` |
| `f"{file_dir}/{dataset.replace('.h5','')}.pkl"` | side effect of `load_data` into `processed_pkl_finetune` |
| `f"{main_dir}/{train_dir}/image_check/...png"` | side effect of `load_data` when `--plot` |

#### 1.3.4 Checkpoint save / load

* **Load (partial, shape-filtered):**
  ```python
  old_state_dict = torch.load(f"{main_dir}/{args.model_pth}", weights_only=False, map_location=device)
  new_state_dict = model.state_dict()
  pretrained_dict = {k: v for k, v in old_state_dict.items()
                     if k in new_state_dict and v.shape == new_state_dict[k].shape}
  new_state_dict.update(pretrained_dict)
  model.load_state_dict(new_state_dict)
  ```
  (lines 124–130). Layers whose name or shape differ (e.g. the new gene decoder)
  keep their random initialisation; nothing is reported about what was dropped.
  Note the path is `{main_dir}/{args.model_pth}` while `train.py` writes
  `AURORA_model_weights.pth` under `{main_dir}/{args.train_path}/` — the default
  invocation therefore fails unless the weights are copied/moved to the project
  root or `--model_pth` is given as `AURORA_train/AURORA_model_weights.pth`.
* **Save (periodic whole model):** `epoch % 100 == 0` →
  `torch.save(model, f"{main_dir}/{args.train_path}/checkpoints/AURORA_finetune_{epoch}.pth")`.
  Again: module object only, no optimizer/epoch/RNG state.
* **Save (final `state_dict`):** `torch.save(state_dict, f"{main_dir}/{args.train_path}/AURORA_finetune_model_weights.pth")`.
* Same `writer` `NameError` defect as `train.py` (lines 262–282 / 288).

---

### 1.4 `code/prepare_st.py`

**Role.** Builds pseudo-bulk statistics from all `.h5ad` ST samples, selects the
top-`--num_genes` genes by average rank, writes the gene list + per-gene median
normalisation factors, then writes per-sample normalised counts (`-cnts.tsv`),
pseudo-bulk (`-bulk.tsv`, `-bulk-rank.tsv`) and per-gene maximum (`-max.tsv`).

#### 1.4.1 argparse

| Flag | Short | Type | Default | Help |
|---|---|---|---|---|
| `--project_path` | `-p` | `str` | `'./'` | Main directory path |
| `--output_path` | `-o` | `str` | `'AURORA_interim'` | Output directory |
| `--metadata_path` | `-m` | `str` | `'Sample_metadata.csv'` | Sample metadata CSV file name |
| `--processed_data_path` | `-d` | `str` | `'processed_data'` | Processed data directory path |
| `--num_genes` | — | `int` | `3000` | Number of highly expressed genes to select |
| `--genetype_file` | — | `str` | `'TCGA_genetype.csv'` | File containing gene types information |

`parser.description` = the copy-pasted rescale description.

#### 1.4.2 Filesystem locations READ

| Location | Where |
|---|---|
| `f"{main_dir}/gene_symbols.csv"` | line 25 (`os.path.exists`) / line 32 (`pd.read_csv(index_col=0)`) |
| Ensembl→symbol map from `sc.queries.biomart_annotations("hsapiens", attrs=["ensembl_gene_id","hgnc_symbol"])` | line 26 — **network call to Biomart** (needs `pybiomart`), only when the CSV is missing |
| `f"{main_dir}/{args.metadata_path}"` | line 38 (`pd.read_csv(index_col=0, header=0)`; row index = sample names) |
| `f"{main_dir}/{args.processed_data_path}/st/{sample}.h5ad` | line 47 and, via `sample_path`, line 64–65 |
| `f"{main_dir}/{args.genetype_file}"` | line 87 (`pd.read_csv(index_col=0)`; needs a `gene_type` column) |
| `f"{main_dir}/{args.output_path}/Gene_normalize_factor.csv"` | line 103 (`pd.read_csv(index_col=0, header=0)`) |
| `os.path.join(bar_folder, f"{dataset}-bars.pickle")` with `bar_folder = f"{main_dir}/{args.output_path}/UNI_multiscale_patches/locs/"` | line 123/125 (`utils.load_pickle`) |
| `f"{bulk_folder}{sample}-max.tsv"` existence checks (`bulk_folder = f"{main_dir}/{args.output_path}/bulk/"`) | lines 44, 61, 107 (skip-already-done) |
| `st_folder = f"{main_dir}/{args.processed_data_path}/st/"` (defined line 19, used line 110) | line 19/110 |

#### 1.4.3 Filesystem locations WRITTEN / created

| Location | Where |
|---|---|
| `f"{main_dir}/gene_symbols.csv"` | line 30 (`gene_symbols.to_csv`) |
| `f"{main_dir}/{args.output_path}/Gene_to_predict.csv"` | line 100 (column `'Genes'`) |
| `f"{main_dir}/{args.output_path}/Gene_normalize_factor.csv"` | line 101 (per-gene median, column `0`) |
| `f"{save_folder}{dataset}-cnts.tsv"` with `save_folder = f"{main_dir}/{args.output_path}/cnts/"` | line 139 (`save_tsv(df_normalized, ...)`) |
| `f"{bulk_folder}{dataset}-bulk.tsv"` | line 159 |
| `f"{bulk_folder}{dataset}-bulk-rank.tsv"` | line 164 |
| `f"{bulk_folder}{dataset}-max.tsv"` | line 171 |
| commented-out outputs: `f"{main_dir}/Lung/gene_sample_statistics.csv"` (81), `f"{save_folder}{dataset}-cnts-smooth.tsv"` (153), `/project/hest1k/cnts_norm_by_gene/` override (36) | not executed |

No `os.makedirs` at all — the output directories must already exist
(`AURORA_interim/cnts/`, `AURORA_interim/bulk/`). No torch checkpoint logic.

#### 1.4.4 Behavioural notes / defects

* **Dead normalisation.** Line 94 `x.apply(lambda x: x/gene_median, axis = 0)`
  computes a normalised frame and throws it away; line 95 ranks the
  *un-normalised* `x`. Gene ranking therefore ignores the median normalisation
  that the following two lines and the `-bulk-rank.tsv` products assume.
* `gene_median` is written without the same "protein_coding" filter expectation
  downstream: `train.py`/`test.py` do `.loc[gene_list,'0']`, i.e. the CSV column
  must literally be named `0`.
* The loop over `file_names` runs **twice** (statistics pass at lines 43–53, write
  pass at 59–171), re-reading every `.h5ad`.
* `st = st[barcode,:]` intersected with the barcode pickle from
  `UNI_multiscale_patches/locs/` — the barcode pickle must exist before this runs
  (produced by `rescale.py`).
* Nested same-type quotes in f-strings? No — this file's `parameters.get` patterns
  are not present; only plain f-strings.

---

### 1.5 `code/smooth_cnt.py`

**Role.** Neighbour-smoothed copy of the counts: reads the per-sample `.h5ad`,
builds a radius-based spatial neighbour graph with `squidpy`, row-normalises the
connectivity matrix, smooths the expression matrix and min–max normalises per
gene; writes `<output_folder>/<sample>-cnts.tsv` (consumed by
`train.py --smooth_cnts` / `test.py --smooth_cnts`).

#### 1.5.1 argparse

| Flag | Short | Type | Default | Help |
|---|---|---|---|---|
| `--project_path` | `-p` | `str` | `'./'` | Main directory path |
| `--output_path` | `-o` | `str` | `'AURORA_interim'` | Output directory |
| `--metadata_path` | `-m` | `str` | `'Sample_metadata.csv'` | Sample metadata CSV file name |
| `--processed_data_path` | `-d` | `str` | `'processed_data'` | Processed data directory path |
| `--output_folder` | `-f` | `str` | `'cnts_smooth'` | Output folder name for smoothed counts |

#### 1.5.2 Filesystem locations READ

| Location | Where |
|---|---|
| `f"{main_dir}/gene_symbols.csv"` | line 26/33 |
| Biomart annotation (network) when the CSV is missing | lines 27–30 |
| `f"{main_dir}/{args.metadata_path}"` | line 39 |
| `os.path.join(st_folder, f"{file_name}.h5ad")`, `st_folder = f"{main_dir}/{args.processed_data_path}/st/"` | lines 20, 47, 49 |
| `f"{meta_folder}/{file_name}.json"`, `meta_folder = f"{main_dir}/{args.processed_data_path}/metadata/"` | lines 21, 51–53 (`inter_spot_dist`, `pixel_size_um_estimated`) |
| `os.path.join(bar_folder, f"{dataset}-bars.pickle")`, `bar_folder = f"{main_dir}/{args.output_path}/UNI_multiscale_patches/locs/"` | lines 24, 65, 67 |
| `f"{save_folder}{dataset}-cnts.tsv"` existence probe (skip-if-done), `save_folder = f"{main_dir}/{args.output_path}/{args.output_folder}/"` | lines 23, 44 |

#### 1.5.3 Filesystem locations WRITTEN / created

| Location | Where |
|---|---|
| `f"{main_dir}/gene_symbols.csv"` | line 31 (only if it did not exist) |
| `f"{save_folder}{dataset}-cnts.tsv"` | line 90 (`save_tsv(df_smooth, ...)`) |

No `os.makedirs`. No torch checkpoint logic.

**Notes.** `radius = 2 * st_metadata['inter_spot_dist'] / st_metadata['pixel_size_um_estimated']`
(line 55) → the metadata JSON must contain both keys. `sq.gr.spatial_neighbors(st,
radius=radius, set_diag=True, coord_type='generic')` requires `squidpy`.
The per-sample `-cnts.tsv` name is reused under a different folder, so
`train.py --smooth_cnts` looks in `{interim}/cnts_smooth/` while the default
(non-smooth) training path uses the *hard-coded* `{interim}/cnts/` written by
`prepare_st.py`.

---

### 1.6 `code/generate_sample_list.py`

**Role.** One-pass inventory of a raw-data directory → `Sample_metadata.csv` with
columns `Sample`, `Dir_name`, `Image_name`.

#### 1.6.1 argparse

| Flag | Short | Type | Default | Help |
|---|---|---|---|---|
| `--project_path` | `-p` | `str` | `'./'` | Main directory path |
| `--raw_data_path` | `-r` | `str` | `'raw_data'` | Directory containing sample subdirectories |
| `--output_name` | `-o` | `str` | `'Sample_metadata.csv'` | Output CSV file name |

`description='Generate sample metadata CSV from raw data directories.'`

#### 1.6.2 Filesystem locations READ

| Location | Where |
|---|---|
| `f"{main_dir}/{args.raw_data_path}"` — `os.listdir(data_dir)` | lines 11–12 |
| `f"{data_dir}/{file}"` — `os.listdir(...)`, filtered on `.tif` | lines 17–18 |

#### 1.6.3 Filesystem locations WRITTEN / created

| Location | Where |
|---|---|
| `f"{main_dir}/{args.output_name}"` | line 29 (`metadata.to_csv(..., index=False)`) |

No checkpoint logic. Notes: every entry of `os.listdir(data_dir)` is assumed to be
a directory (a stray file raises `NotADirectoryError`); `sample_name = file`
(the commented-out `file.split("_")[0]` is disabled); only the first `.tif` per
folder is recorded (warning printed if several); a missing image yields the
literal string `"No image found"` in `Image_name`; the output is *not* the same
schema as the `Sample_metadata.csv` that `HEST_preprocess.py` expects columns
`Sample`/`Dir_name`/`Image_name` from — it is the same schema, and
`rescale.py` additionally expects an `Image_name` index / `Pixel_size_um` column
when `--inference` is used.

---

### 1.7 `code/HEST_preprocess.py`

**Role.** Drives `hest.VisiumReader` (HEST) over the sample metadata: reads the
full-res H&E, the `filtered_feature_bc_matrix.h5` and the `spatial/` folder,
segments tissue, dumps 224-px patches, saves `.h5`/`.h5ad`/TIFF artefacts, and
re-organises the results into the AURORA `processed_data/` layout with POSIX
`mv`/`rm -rf` shell commands.

#### 1.7.1 argparse

| Flag | Short | Type | Default | Help |
|---|---|---|---|---|
| `--project_path` | `-p` | `str` | `'./'` | Main directory path |
| `--raw_data_path` | `-r` | `str` | `'raw_data'` | Directory containing sample subdirectories |
| `--metadata_path` | `-m` | `str` | `'Sample_metadata.csv'` | Output CSV file name |
| `--output_path` | `-o` | `str` | `'processed_data'` | Directory to save processed data |
| `--pixel_size` | — | `int` | `0.5` | Pixel size for H&E images in um/px |
| `--patch_size` | — | `int` | `224` | Patch size for dumping patches |

`description='Preprocess H&E stained spatial transcriptomics data using HEST'`.
`--pixel_size` is declared `type=int` while its default is the float `0.5`
(argparse does not apply `type` to defaults), so passing `--pixel_size 0.5`
raises `invalid int value`.

#### 1.7.2 Filesystem locations READ

| Location | Where |
|---|---|
| `f"{main_dir}/{args.metadata_path}"` | line 17 (`pd.read_csv`) |
| `f'{data_dir}/{metadata.iloc[idx]["Dir_name"]}/{metadata.iloc[idx]["Image_name"]}'` | line 31 (`fullres_img_path`) |
| `f'{data_dir}/{metadata.iloc[idx]["Dir_name"]}/filtered_feature_bc_matrix.h5'` | line 32 (`bc_matrix_path`) |
| `f'{data_dir}/{metadata.iloc[idx]["Dir_name"]}/spatial'` | line 33 (`spatial_coord_path`) |
| `data_dir = f"{main_dir}/{args.raw_data_path}"` | lines 16, 31–33 |

#### 1.7.3 Filesystem locations WRITTEN / created

`os.makedirs(..., exist_ok=True)` (lines 18–28):
`f"{main_dir}/{args.output_path}"`, `.../patches`, `.../metadata`, `.../patches_vis`,
`.../pixel_size_vis`, `.../spatial_plots`, `.../st`, `.../tissue_seg`,
`.../thumbnails`, `.../wsis`, `.../tmp`.

Per-sample staging (all under `f"{main_dir}/{args.output_path}/tmp"`), created by
HEST calls and then shell-moved:

| Produced in `tmp` (created by) | Moved to (line) |
|---|---|
| `f"{main_dir}/{args.output_path}/tmp"` (spatial plot, line 41) | stays |
| `st.dump_patches(f"{main_dir}/{args.output_path}/tmp", name=Sample, ...)` (line 43) | `{output_path}/patches/` (line 49) |
| `<Sample>_patch_vis.png` | `{output_path}/patches_vis/` (line 50) |
| `aligned_adata.h5ad` (`st.save(..., plot_pxl_size=True, bigtiff=True)`, line 48) | `{output_path}/st/<Sample>.h5ad` (line 51) |
| `aligned_fullres_HE.tif` | `{output_path}/wsis/<Sample>.tif` (line 52) |
| `downscaled_fullres.jpeg` | `{output_path}/thumbnails/<Sample>_downscaled_fullres.jpeg` (line 53) |
| `metrics.json` | `{output_path}/metadata/<Sample>.json` (line 54) |
| `tissue_contours.geojson` | `{output_path}/tissue_seg/<Sample>_contours.geojson` (line 55) |
| `tissue_seg_vis.jpg` | `{output_path}/tissue_seg/<Sample>_vis.jpg` (line 56) |
| `spatial_plots.png` | `{output_path}/spatial_plots/<Sample>_spatial_plots.png` (line 57) |
| `pixel_size_vis.png` | `{output_path}/pixel_size_vis/<Sample>_pixel_size_vis.png` (line 58) |
| final step: `rm -rf {main_dir}/{args.output_path}/tmp/*` (line 59) | — |

No torch checkpoint logic. `import pyvips` (line 3) is never used by this script;
`hest` provides `VisiumReader`. The `os.system` moves assume a POSIX shell
(`mv`, `rm -rf`) → Windows-unsafe.

---

### 1.8 `code/datasets.py`

**Role.** Dataset/sampler/collate library. **No `argparse`, no `if __name__`,
no filesystem paths of its own.** Imports:
`numpy`, `cv2`, `torch.utils.data.{Dataset,Sampler}`, `typing.Iterator`,
`PIL.Image`, `torch`, and the local `train_utils.SuperpatchCollection`.

Classes / functions:

| Object | Notes |
|---|---|
| `PatchDataset(image, mask, transform, loc, spot_radius_pixel, patch_size=224, stride=224, filter_prop=0.2, high_res=False)` | pads image/mask to a multiple of `stride`, whites out masked pixels, returns `(transform(patch), (i, j), keep_flag)`; in `high_res` mode `keep_flag` is a **16×16** boolean grid on 14-px tokens. All arrays are passed in memory — no paths. |
| `PrefilteredHighResPatchDataset(PatchDataset)` | requires `high_res=True`; token grid `token_size=14`; computes `fine_valid`, an integral-image window sum to keep only active patches, `source_indices`/`active_positions`, `fine_contribution_counts`; overrides `__len__`, `__getitem__`, adds `next_origin_token_row()`. Pure in-memory. |
| `RegularGridPatchDataset(PrefilteredHighResPatchDataset)` | compatibility alias. |
| `RowSpanBatchSampler(dataset, *, batch_size, max_row_span_tokens)` | batches of consecutive active patches with a bounded source-row span; consumes `dataset.active_positions` / `dataset.token_size`. |
| `HistologyGeneExpressionDataset(data, device, highres=False, adj_mat=False)` | stores either an explicit list of 10/11-tuples or a lazy `SuperpatchCollection` (then `device` is forced to `'cpu'`); converts dicts/arrays to tensors (handles the compact `{'indices','values'}` sparse `spot_weight` form) and returns the 10-tuple consumed by `train.py`/`test.py`. |
| `my_collate(batch)` | 10-tuple collate for training. |
| `my_collate_pred(batch)` | 4-tuple collate for prediction (`emb, bulk, bulk_rank, father_patches`). |
| `my_collate_adjmat(batch)` | 11-tuple collate (adds `adj_mat`). |

**Checkpoint logic: none.**

**Filesystem coupling.** The only file IO reachable through this module is
`SuperpatchCollection.__getitem__`, i.e. `utils.load_pickle(f"{self.file_dir}/{name}.pkl")`
in `train_utils.py`. Because `datasets.py` imports `train_utils` at module scope,
importing `datasets` (as `extract_features.py` and
`extract_features_highres*.py` do for `PatchDataset`) pulls in the whole heavy
`train_utils` import list (`sklearn`, `scipy`, `tifffile`, `pandas`, `tqdm`,
`cv2`, `utils`→`h5py`/`yaml`).

---

### 1.9 `code/train_utils.py`

**Role.** Data-assembly + loss library. **No `argparse`, no `__main__`,
no CLI.** Module-level imports: `json`, `sklearn.cluster.MiniBatchKMeans`,
`tqdm`, `tifffile`, `os`, `numpy`, `pandas`, `cv2`, `gc`, `threading`, `queue`,
`collections.OrderedDict`, `scipy.spatial.KDTree`,
`concurrent.futures.ProcessPoolExecutor`, `functools.partial`, `torch`,
`torch.nn.functional as F`, local `utils`, `json` (again), and mid-file
`from scipy.sparse import coo_matrix` (line 1256).
Lazy imports: `import os` (lines 210, 1265), `import matplotlib as mpl` (line 942,
only in the `plot_mask` branch of `load_data2`).

Functions/classes and their CLI parameters (keyword arguments, not argparse):

| Function | Key parameters | Purpose |
|---|---|---|
| `load_image(dataset, main_dir, processed_data_dir, interim_dir, target_radius=0.5)` | — | reads metadata JSON + rescaled image/mask TIFFs |
| `plot_image_with_spot`, `plot_image_with_patch` | — | in-memory CV2 overlays |
| `get_unique_indices(tensor, device)` | — | unique values + first-occurrence indices |
| `query_batch`, `query_kdtree_parallel`, `find_nearby_spots(spot_coord, query, k=4, workers=4)` | `k`, `workers` | KD-tree nearest-neighbour queries via `ProcessPoolExecutor` |
| `square_intersection_area` | — | spot/patch overlap area |
| `js_divergence`, `reverse_kl`, `js_reverse_mix`, `wasserstein_loss`, `masked_focal_kl_loss(log_pred, target_probs, gamma=2.0, mask_thresh=0.2)` | `gamma`, `mask_thresh` | losses (only `masked_focal_kl_loss` is used by `train.py`/`finetune.py`) |
| `load_data(...)` | `patch_sizes=[3584,896,224]`, `from_file`, `file_dir`, `num_workers`, `pixel_size`, `barcode_dir`, `load_graph`, `rand_count`, `smooth_cnts`, `patch_folder_name='UNI_multiscale_patches'`, `cnts_smooth_folder_name='cnts_smooth'`, `bulk_folder_name='bulk'`, `prop_folder_name='deconvolution'`, `plot_mask`, `load_cell_type` | low-res multi-scale dataset, 10-tuple (11 with `load_graph`) |
| `load_data_pred(...)` | `patch_sizes`, `from_file`, `file_dir`, `num_workers` | embedding-only prediction data (4-tuple) |
| `SuperpatchCollection(superpatch_meta, file_dir, cache_size=2)` | `cache_size` | lazy per-superpatch `.pkl` loader with LRU cache |
| `PrefetchLoader(loader, prefetch=4)` | `prefetch` | threaded prefetch wrapper |
| `project_spot_weight(spot_weight_sparse, output, spot_chunk=128)` / `project_spot_weight_columns(..., spot_chunk=128)` | `spot_chunk` | sparse spot projection (used by `train_highres.py`) |
| `load_data2(...)` | `patch_sizes=[56,14]`, `superpatch_size=5000`, `superpatch_stride=4000`, `superpatch_minsize=1000`, `superpatch_minlowres_patch=10`, `global_size=16`, `local_size=4`, `cache_embeddings=True`, `cache_size=4`, … | high-res superpatch dataset (returns `SuperpatchCollection`) |
| `load_data2_pred(...)` | `superpatch_size=5000`, `superpatch_stride=4000`, `superpatch_minsize=1000`, `superpatch_minlowres_patch=10`, `cache_embeddings=True`, `cache_size=2`, … | high-res prediction superpatches |
| `load_data_withadj(...)` | `patch_size=224`, `smooth_cnts=True`, … | single-scale variant with a kNN patch graph (note: contains references to undefined `patch_mask`/`plot_color` in its `plot_mask` branch) |
| `as_smallest_int_array(values, *, allow_unsigned=True)` | `allow_unsigned` | smallest lossless integer dtype (used by `predict*.py`) |

#### 1.9.1 Filesystem locations READ (literal expressions)

| Location | Function |
|---|---|
| `f"{main_dir}/{processed_data_dir}/metadata/{dataset}.json"` | `load_image` (line 23) |
| `f"{main_dir}/{interim_dir}/UNI_multiscale_patches/wsis_rescaled/{dataset}_rescaled.tif"` | `load_image` (line 30) |
| `f"{main_dir}/{interim_dir}/UNI_multiscale_patches/wsis_rescaled/{dataset}_mask_rescaled.tif"` | `load_image` (line 31) |
| `f"{main_dir}/{interim_dir}/UNI_multiscale_patches/wsis_rescaled/{dataset}_rescaled.tif"` | `load_data2_pred` (line 1086, via `tifffile.imread`) |
| `f"{emb_folder}/wsis_rescaled/{dataset}_loc_rescaled.npy"` (`emb_folder = f"{main_dir}/{interim_dir}/{patch_folder_name}"`) | `load_data` (227), `load_data_withadj` (1282) |
| `f"{emb_folder}/wsis_rescaled/{dataset}_barcode.pickle"` | `load_data` (228), `load_data_withadj` (1283) |
| `f"{prop_folder}/{dataset}_prop.csv"` (`prop_folder = f"{main_dir}/{interim_dir}/{prop_folder_name}"`) | `load_data` (233), `load_data_withadj` (1288) |
| `f"{main_dir}/{interim_dir}/UNI_multiscale_patches/locs/{dataset}-bars.pickle"` | `load_data` (236), `load_data_withadj` (1291) |
| `f"{barcode_dir}/{dataset}-bars.pickle"` (when `barcode_dir` is given) | `load_data` (238), `load_data_withadj` (1293) |
| `f"{emb_folder}/emb_{patch_size}/{dataset}.npz"` | `load_data` (252), `load_data_withadj` (1304) |
| `f"{emb_folder}/emb_{patch_size}/{dataset}.npz"`, `f"{emb_folder}/emb_224/{dataset}.npz"` | `load_data_pred` (476), `load_data2` (779/793), `load_data2_pred` (1122/1136/1193) |
| `f"{cnt_folder}/{dataset}-cnts.tsv"` (`cnt_folder = f"{main_dir}/{interim_dir}/cnts"` or `f"{main_dir}/{interim_dir}/{cnts_smooth_folder_name}"`) | `load_data` (397), `load_data_withadj` (1369) |
| `f"{bulk_folder}/{dataset}-bulk.tsv"` (`bulk_folder = f"{main_dir}/{interim_dir}/{bulk_folder_name}"` or `f"{main_dir}/{interim_dir}/bulk"`) | `load_data` (412), `load_data2` (1004), `load_data_withadj` (1381) |
| `f"{bulk_folder}/{dataset}-max.tsv"` | `load_data` (425), `load_data_withadj` (1392) |
| `f"{file_dir}/{dataset}.pkl"` (`from_file=True`) | `load_data` (450), `load_data_pred` (512), `load_data_withadj` (1409) |
| `f"{file_dir}/superpatch_meta.csv"` (resume index) | `load_data2` (720/722/1044), `load_data2_pred` (1065/1067/1246) |
| `f"{file_dir}/{dataset}_superpatch_{superpatch_id}.pkl"` existence probes | `load_data2` (761/807), `load_data2_pred` (1104/1150) |
| `f"{main_dir}/{train_dir}/train_pred_lowres/{dataset}-pred.pickle"` | `load_data2` (780, via `utils.load_pickle`) |
| `f"{self.file_dir}/{name}.pkl"` | `SuperpatchCollection.__getitem__` (540, via `utils.load_pickle`) |
| commented-out alternatives: `.../train_pred_lowres/{dataset}.h5-pred.pickle` (781), `f"{main_dir}/TCGA_LUAD/..."` block (459–468), `f"{pred_dir}/pred/{dataset}-pred.pickle"` (1124) | not executed |

#### 1.9.2 Filesystem locations WRITTEN / created (literal expressions)

| Location | Function |
|---|---|
| `f"{main_dir}/{train_dir}/image_check/"` (`os.makedirs(..., exist_ok=True)`) | `load_data` (211), `load_data2` (714), `load_data_withadj` (1266) |
| `f"{main_dir}/{train_dir}/image_check/{dataset}_{patch_size}patch.png"` (`cv2.imwrite`) | `load_data` (317), only if `plot_mask` |
| `f"{file_dir}/{dataset.replace('.h5','')}.pkl"` (`utils.save_pickle`) | `load_data` (440/445), `load_data_pred` (507), `load_data_withadj` (1404) |
| `f"{file_dir}/superpatch_info.json"` (`json.dump`) | `load_data2` (734), `load_data2_pred` (1078) |
| `f"{file_dir}/superpatch_meta.csv"` (`pd.DataFrame(...).to_csv`, rewritten after **every dataset** and at the end) | `load_data2` (730/1037), `load_data2_pred` (1075/1239) |
| `f"{main_dir}/{train_dir}/image_check/{dataset}_superpatch_{superpatch_id}_{patch_size}patch.png"` | `load_data2` (914), only if `plot_mask` |
| `f"{main_dir}/{train_dir}/image_check/{dataset}_superpatch_{superpatch_id}_spotweight_{patch_size}patch.png"` | `load_data2` (962), only if `plot_mask` (also lazily imports `matplotlib`) |
| `f"{main_dir}/{train_dir}/image_check/{dataset}_superpatch_{superpatch_id}_father_{patch_size}patch.png"` | `load_data2` (988), only if `plot_mask` |
| `f"{file_dir}/{name}.pkl"` (`utils.save_pickle`, `name = f"{dataset}_superpatch_{superpatch_id}"`) | `load_data2` (1020), `load_data2_pred` (1222) |

No `torch.save` / `torch.load` anywhere in this module (losses and projections
only) — **no checkpoint logic.** Note the two `load_data*` families write into
`file_dir` while `load_image` reads from `interim_dir`; the guard used to fill
`max_exp` (`load_data`, lines 430–432) tests membership in `bulk.index` rather
than `max_exp.index`, so a gene present in `-bulk.tsv` but absent from
`-max.tsv` raises `KeyError` (the analogous block for `bulk_filtered` is correct).

---

## 2. Import matrix

Legend: **X** = imported at module import time (directly, or transitively through
a local module); **l** = imported lazily inside a function/branch of that file or
of a transitively imported local module (**not** executed by a plain `import`);
**–** = not reachable.

Columns are top-level third-party modules (the name after `import` / `from`), then
the local modules named in the request.

| File | numpy | pandas | torch | torch_geom. | tqdm | cv2 | PIL | h5py | yaml | scipy | sklearn | tifffile | matplotlib | numba | tensorboard | scanpy | squidpy | pyvips | hest | train_utils | datasets | model | utils | image_utils | highres_* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `train.py` | X | X | X | X | X | X | X | X | X | X | X | X | – | – | **X** | – | – | – | – | X | X | X | X | – | – |
| `test.py` | X | X | X | X | X | X | X | X | X | X | X | X | – | – | – | – | – | – | – | X | – | X | X | – | – |
| `finetune.py` | X | X | X | X | X | X | X | X | X | X | X | X | – | – | **X** | – | – | – | – | X | X | X | X | – | – |
| `prepare_st.py` | X | X | – | – | X | – | X | X | X | – | – | – | – | – | – | **X** | – | – | – | – | – | – | X | – | – |
| `smooth_cnt.py` | X | X | – | – | X | – | X | X | X | – | – | – | – | – | – | **X** | **X** | – | – | – | – | – | X | – | – |
| `generate_sample_list.py` | – | X | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| `HEST_preprocess.py` | – | X | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | **X** | **X** | – | – | – | – | – | – |
| `datasets.py` | X | X | X | – | X | X | X | X | X | X | X | X | – | – | – | – | – | – | – | X | – | – | X | – | – |
| `train_utils.py` | X | X | X | – | X | X | X | X | X | X | X | X | l | – | – | – | – | – | – | – | – | – | X | – | – |

Justification of the non-obvious cells:

* **`torch_geometric` for `train.py` / `test.py` / `finetune.py`**: `model.py`
  executes `from torch_geometric.nn import GraphConv, GCNConv` at **module level,
  line 250** (mid-file, not inside a class or a `try`). Any import of `model`
  therefore fails without PyG, even though the AURORA inference path never builds
  a `GAT`/`GCN`.
* **`PIL`, `h5py`, `yaml`, `scipy`, `sklearn`, `tifffile` in `prepare_st.py` /
  `smooth_cnt.py`**: transitive through the local `utils` module only
  (`PIL`, `h5py`, `yaml`). `scipy`/`sklearn`/`tifffile` are **not** in those two
  files' closure.
* **`pandas`/`numpy`/`tqdm`/`sklearn`/`scipy`/`tifffile` in `datasets.py`**: only
  the first three are direct; the rest arrive via `from train_utils import
  SuperpatchCollection` (which also drags in `utils` → `h5py`, `yaml`, `PIL`).
* **`matplotlib` (l) in `train_utils.py`**: `import matplotlib as mpl` sits inside
  the `plot_mask` branch of `load_data2` (line 942); `load_data`'s own plotting
  branch is pure CV2, and `train.py`/`finetune.py`/`test.py` call `load_data`
  only, so matplotlib stays unimported on those paths.
* **`torch.utils.tensorboard` → `tensorboard`**: `train.py` line 8,
  `finetune.py` line 8 (`from torch.utils.tensorboard import SummaryWriter`).
  `test.py` does **not** import it (the `--log` argument is commented out).

**Classification requested by the brief (with corrections):**

* **Needed ONLY for training:** `tensorboard` (with its `protobuf`/`setuptools`
  dependencies), pulled in by `train.py`, `finetune.py` (and `train_highres.py`,
  `train_GraphLSTM.py`, `test_highres.py`). Nothing else in the nine-file matrix is
  training-exclusive: `sklearn`, `scipy`, `tifffile`, `matplotlib`, `torch`,
  `cv2`, `h5py`, `yaml`, `pandas`, `numpy`, `tqdm`, `torch_geometric` are all
  reachable from inference entry points as well (see §3).
* **Needed for inference too (contra the “training-only” hint):**
  `sklearn` (direct `KMeans`/`MiniBatchKMeans` import in `extract_features.py` and
  `train_utils.py`, `GaussianMixture` in `HistoSweep/textureAnalysis.py`),
  `matplotlib` (module-level in `extract_features.py`, `extract_features_highres.py`,
  `plot.py`, `HistoSweep/utils.py`, `HistoSweep/generateMask.py`),
  `scanpy`/`geopandas`/`rasterio` (`rescale.py` top-level, lines 7–9),
  `numba`/`psutil`/`skimage` (`run_HistoSweep.py`'s lazily imported HistoSweep
  modules).
* **Not needed for either of the two lists** (reachable only from other tools):
  `pybiomart` (indirect, via `scanpy.queries.biomart_annotations` in
  `prepare_st.py` / `smooth_cnt.py` during the **offline preprocessing** stage),
  `squidpy` (`smooth_cnt.py`), `pyvips` + `hest` (`HEST_preprocess.py`; `pyvips`
  also at top of `HistoSweep/utils.py`, stubbed by `run_HistoSweep.py`),
  `pytorch_lightning` / `lightning` (`istar/impute.py`, `istar/train.py`),
  `umap-learn` (`istar/reduce_dim.py`), `seaborn` (`istar/enrich.py`),
  `brokenaxes` (`HistoSweep/additionalPlots.py`), `torchvision` /
  `webdataset` / `requests` (`istar/hipt_*.py`, `istar/hipT_heatmap_utils.py`).

---

## 3. Inference-reachable closure

Method: for each entry point the module-level `import`/`from` statements were read
directly, then each local module was followed recursively. Lazy (function-scope)
imports are marked **(lazy)**; process-level invocations are marked
**(subprocess)** and are listed because they execute the same code in a child
interpreter (they are *not* import edges).

### 3.1 Per-entry-point local modules

| Entry point | Local modules imported (module level unless noted) |
|---|---|
| `predict.py` | `model`, `train_utils`, `utils` |
| `predict_highres.py` | `model`, `train_utils`, `utils`; `io`, `zstandard` **(lazy**, line 265–266**)** |
| `rescale.py` | `utils`, `image_utils`; `openslide` **(lazy**, only when `--inference`, line 33**)**; `datetime` (line 151) |
| `extract_features.py` | `utils`, `image_utils`, `datasets` → `train_utils` → `utils` |
| `extract_features_highres.py` | `datasets` → `train_utils`, `image_utils`; `pyvips` **(lazy**, line 68**)**; `run_CATCH.py` via `subprocess.run(..., shell=True, executable="/bin/bash")` **(subprocess**, lines 124–131**)** |
| `segment_image.py` | *(none — no local imports)*; `create_patches_fp.py` via `os.system(f"python -u ...")` **(subprocess**, lines 21–34**)** |
| `plot.py` | `utils`; `tifffile` **(lazy**, line 9**)**; `pickle` (stdlib, lazy lines 133/162) |
| `run_istar.py` | `utils`, `istar.preprocess` → `istar.utils`, `istar.image` |
| `run_HistoSweep.py` | *(none at module level)*; `HistoSweep.computeMetrics`, `HistoSweep.densityFiltering`, `HistoSweep.textureAnalysis`, `HistoSweep.ratioFiltering`, `HistoSweep.generateMask`, `HistoSweep.saveParameters` **(all lazy**, inserted at `sys.path[0]`, lines 195–200**)** → `HistoSweep/utils.py`; `pandas` **(lazy**, line 315**)**; `tifffile` **(lazy**, 108/190**)**; `PIL.Image` **(lazy**, 191**)**; `pyvips` **(lazy probe**, 149, with a stub module registered when absent**)** |
| `predict_TCGA.py` | `model`, `train_utils`, `utils` |
| `run_chunked_tcga.py` | *(none)* — pure orchestrator: `extract_features_highres_optimized.py`, `run_HistoSweep.py`, `predict_highres_TCGA_optimized.py` **(subprocess**, lines 224/266/281**)** |

Second-level local modules reached:

* `train_utils` → `utils`
* `datasets` → `train_utils` → `utils`
* `model` → (torch, torch_geometric only)
* `image_utils` → (no local imports)
* `istar.preprocess` → `istar.utils`, `istar.image`
* `create_patches_fp` → `wsi_core.WholeSlideImage` → `wsi_core.wsi_utils`, `wsi_core.util_classes`; also `wsi_core.wsi_utils`, `wsi_core.batch_process_utils`
* `extract_features_highres_optimized` → `highres_inference_pipeline`, `highres_gpu_strip_pipeline` → `highres_grid_pipeline`, `image_utils_optimized` → `highres_feature_ops`; `datasets` **(lazy**, lines 740/1004/1093/1095**)**; `highres_gpu_strip_pipeline` **(lazy**, 1010**)**; `pyvips`, `yaml`, `tifffile`, `cv2`, `matplotlib`, `sklearn`, `pandas`, `torch.utils.data`, `tqdm` all **(lazy**, lines 490–565 / 730–731 / 998–1001**)** — `torchvision` is *not* used here.
* `predict_highres_TCGA_optimized` → `train_utils`, `model`, `utils`; `tifffile` **(lazy**, 16**)**; `hdf5plugin` **(lazy**, 84**)**; `scipy.sparse.csr_matrix` **(lazy**, 683**)**; `io`/`zstandard` **(lazy**, 787–788**)**

### 3.2 Transitive third-party top-level modules, per entry point

| Entry point | Third-party modules required (lazy-only in parentheses) |
|---|---|
| `predict.py` | numpy, pandas, torch, tqdm, cv2, PIL, h5py, yaml, scipy, sklearn, tifffile, **torch_geometric** |
| `predict_highres.py` | same as `predict.py` + h5py (direct) (+ zstandard lazy) |
| `rescale.py` | numpy, pandas, tqdm, tifffile, geopandas, rasterio, scanpy, cv2, PIL, h5py, yaml, **torch**, timm, huggingface_hub (all via `image_utils`/`utils`) (+ openslide lazy) |
| `extract_features.py` | numpy, pandas, torch, tqdm, tifffile, matplotlib, sklearn, cv2, PIL, h5py, yaml, scipy, timm, huggingface_hub |
| `extract_features_highres.py` | numpy, pandas, torch, tqdm, tifffile, matplotlib, sklearn, cv2, PIL, h5py, yaml, scipy, timm, huggingface_hub (the last four reach `scipy`/`h5py`/`yaml`/`sklearn`/`tifffile` via `datasets` → `train_utils` → `utils`) (+ pyvips lazy) |
| `segment_image.py` | pandas, numpy, h5py, PIL, cv2, tqdm, matplotlib, **openslide** (via the `create_patches_fp.py` subprocess → `wsi_core`) (+ scipy.stats lazy in `wsi_utils`) |
| `plot.py` | numpy, pandas, matplotlib, cv2, PIL, h5py, yaml (+ tifffile lazy) |
| `run_istar.py` | numpy, pandas, tifffile, cv2, torch, scipy, skimage, einops, PIL, h5py, yaml |
| `run_HistoSweep.py` | numpy, pandas, tifffile, PIL, matplotlib, scipy, skimage, sklearn, numba, (pyvips — stubbed when missing), (+ psutil lazy in `computeMetrics`) |
| `predict_TCGA.py` | numpy, pandas, torch, tqdm, h5py, PIL, yaml, scipy, sklearn, tifffile, **torch_geometric** |
| `run_chunked_tcga.py` | pandas only for itself; in the spawned children: the union of `extract_features_highres_optimized.py`, `run_HistoSweep.py` and `predict_highres_TCGA_optimized.py` (see the union below), i.e. also timm, huggingface_hub, torch_geometric, matplotlib, sklearn, scipy, skimage, numba, h5py, hdf5plugin/zstandard (lazy) |

### 3.3 Union over all eleven entry points

**Local modules (17):**
`model.py`, `train_utils.py`, `utils.py`, `image_utils.py`, `datasets.py`,
`image_utils_optimized.py`, `highres_feature_ops.py`,
`highres_inference_pipeline.py`, `highres_gpu_strip_pipeline.py`,
`highres_grid_pipeline.py`, `istar/preprocess.py`, `istar/utils.py`,
`istar/image.py`, `HistoSweep/{computeMetrics,densityFiltering,textureAnalysis,ratioFiltering,generateMask,saveParameters,utils}.py`,
`wsi_core/{WholeSlideImage,wsi_utils,batch_process_utils,util_classes}.py`,
plus the entry-point scripts themselves and (subprocess) `create_patches_fp.py`,
`extract_features_highres_optimized.py`, `predict_highres_TCGA_optimized.py`.

**Third-party top-level modules (union):**
`numpy`, `pandas`, `torch`, `torch_geometric`, `tqdm`, `cv2` (opencv-python),
`PIL` (Pillow), `h5py`, `yaml` (PyYAML), `scipy`, `sklearn` (scikit-learn),
`tifffile`, `matplotlib`, `timm`, `huggingface_hub`, `geopandas`, `rasterio`,
`scanpy`, `openslide` (openslide-python), `einops`, `skimage` (scikit-image),
`numba`, and the lazily required `zstandard`, `hdf5plugin`, `psutil`.
`pyvips` is required by `HistoSweep/utils.py` at import time but is replaced by an
in-process stub by `run_HistoSweep.py` when absent.

**Explicitly NOT reachable** from the eleven entry points (so they can stay
optional for a pure inference install): `tensorboard` (training only),
`pytorch_lightning`/`lightning`, `umap-learn`, `seaborn`, `brokenaxes`,
`torchvision`, `webdataset`, `requests`, `hest`, `squidpy`, `pybiomart`
(the last four belong to `HEST_preprocess.py` / `smooth_cnt.py` /
`prepare_st.py`, i.e. the offline data-preparation stage).
`imagecodecs` is not imported anywhere but is listed in `requirements.txt` as an
implicit `tifffile` codec dependency.

---

## 4. Surprises, defects and traps worth carrying forward

1. **`torch_geometric` is a hard inference dependency.** `model.py` line 250
   imports `GraphConv, GCNConv` at module scope, so `predict.py`,
   `predict_highres.py`, `predict_TCGA.py` and every `datasets`/`train_utils`
   consumer cannot even import `model` without PyG — a package absent from
   `requirements.txt`.
2. **`writer` `NameError` without `--log`.** `train.py` (308–347, 354) and
   `finetune.py` (262–282, 288) call `writer.add_scalar(...)`/`writer.close()`
   unconditionally although `writer` is only created under `if args.log:`.
   Training aborts at the end of the first epoch unless `--log` is passed.
3. **Whole-module checkpoints.** Both trainers `torch.save(model, ...)` every 100
   epochs, embedding pickled class paths rather than a `state_dict`; no optimizer,
   epoch, scheduler or RNG state is stored anywhere, so training cannot truly be
   resumed. Only the final `state_dict` file is portable/loadable by `test.py`.
4. **`finetune.py` weight-path mismatch.** It loads
   `f"{main_dir}/{args.model_pth}"` while `train.py` writes
   `f"{main_dir}/{args.train_path}/AURORA_model_weights.pth"`; the documented
   defaults do not line up.
5. **`prepare_st.py` silently drops the gene-median normalisation** (line 94 result
   discarded) while still emitting `Gene_normalize_factor.csv`, which downstream
   training and evaluation use via `.loc[gene_list,'0']` — the normalisation is
   applied at train/inference time but was never applied to the ranking.
6. **`train_utils.load_data` `max_exp` guard bug** (lines 430–432): membership is
   tested against `bulk.index`, not `max_exp.index`; a gene in `-bulk.tsv` but not
   in `-max.tsv` raises `KeyError`. Also `load_data_withadj`'s `plot_mask` branch
   references undefined `patch_mask`/`plot_color`.
7. **Python ≥ 3.12 required** by PEP 701 nested same-type quotes in f-strings:
   `train.py` lines 48/81, `test.py` lines 54/70, `finetune.py` lines 45/81.
8. **`requirements.txt` is incomplete.** Missing for the analysed code paths:
   `torch` itself, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `h5py`,
   `PyYAML`, `numba`, `squidpy`, `pyvips`, `hest`, `torch_geometric`, `psutil`,
   `hdf5plugin`. Present but only used by non-entry-point tools: `lightning`,
   `pybiomart`, `imagecodecs`.
9. **`datasets.py` drags in the heavy `train_utils` closure** (sklearn, scipy,
   tifffile, pandas, tqdm, utils→h5py/yaml) merely for a `SuperpatchCollection`
   `isinstance` check — feature extraction pays for the training-data stack.
10. **HEST_preprocess is POSIX-only and lossy on reruns**: `os.system("mv ...")`
    / `"rm -rf ..."` (lines 49–59) mean it cannot run under Windows `cmd`/`pwsh`,
    and `--pixel_size` (declared `type=int`, default `0.5`) rejects fractional
    values on the command line. `import pyvips` is unused there.
11. **Cache-key coupling across scripts.** `train.py` looks for
    `{train_path}/processed_pkl[_smooth]`, `test.py` creates and reuses the same
    two directories, `finetune.py` uses `processed_pkl_finetune`; all three write
    into whatever `file_dir` they were given, so a `test.py` run both reads and
    mutates the training cache. `train_utils.load_data2*` additionally keep their
    own `superpatch_meta.csv` resume index next to the `.pkl` files.
12. **`sys.path` shadowing is load-bearing.** `run_HistoSweep.py` prepends the
    `HistoSweep/` package dir so `from utils import ...` inside it resolves to
    `HistoSweep/utils.py`, not `code/utils.py`; `run_istar.py` relies on
    `code/istar` being importable as a namespace package (**there is no
    `code/istar/__init__.py`**), while `code/istar/train.py`, `istar/utils.py`,
    `istar/rescale.py`, `istar/extract_features.py` shadow same-named top-level
    modules. `plot.py` duplicates the `load_image` helper instead of importing it.
13. **`run_chunked_tcga.py` has no import closure of its own** — it is a
    `subprocess` orchestrator with `pandas` as its only third-party import; its
    real dependency surface is the union of the three child scripts, and it
    deletes per-chunk `emb_{ps}/{sample}.npz`, `processed_pkl_test_highres/*.pkl`
    and (optionally) HistoSweep masks after each successful chunk.
