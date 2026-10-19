# AURORA high-resolution stage — pipeline reconnaissance

Static (read-only) analysis of the high-resolution branch of AURORA.
Repository root: `<repo>`. No existing file was modified.

**Analysed in full (the six requested scripts):**

| # | Path |
|---|------|
| 1 | `code/extract_features_highres.py` (381 lines) |
| 2 | `code/prepare_highres_training.py` (129 lines) |
| 3 | `code/train_highres.py` (387 lines) |
| 4 | `code/highres_feature_ops.py` (658 lines) |
| 5 | `code/highres_grid_pipeline.py` (438 lines) |
| 6 | `code/highres_inference_pipeline.py` (564 lines) |

**Supporting modules read only to resolve cross-script references** (all under `code/`):
`image_utils.py` (the module `extract_features_highres.py` actually imports), `datasets.py`, `model.py`,
`train_utils.py`, `utils.py`, `rescale.py`, `extract_features.py`, `prepare_st.py`, `image_utils_optimized.py`,
`extract_features_highres_optimized.py`, `run_chunked_tcga.py`, plus repo-root `command.sh` and `requirements.txt`.

**Placeholders used below** (they are the literal f-string pieces; nothing is silently converted to an absolute path):

| Symbol | Definition | Default from |
|---|---|---|
| `P` | `args.project_path` | `'./'` |
| `O` | `args.output_path` (extract) | `'AURORA_interim'` |
| `I` | `args.interim_path` (train side) | `'AURORA_interim'` in `train_highres.py`, `None` in `prepare_highres_training.py` |
| `T` | `args.train_path` | `'AURORA_train'` |
| `D` | `args.processed_data_path` | `'processed_data'` |
| `SAVE` | `f"{P}/{O}/UNI_multiscale_patches"` | written by `extract_features_highres.py` |
| `EMB` | `f"{P}/{I}/UNI_multiscale_patches"` | read by `train_utils.load_data` / `load_data2` |

`SAVE` and `EMB` are the **same directory** when the same value is passed to `-o` and `-i` (as in `command.sh`,
where both are `debug`). There is no other coupling between the two flags.

---

## 1. `code/extract_features_highres.py`

Purpose: run UNI2-h (or CATCH) over `wsis_rescaled` images at 224 px with overlapping stride, aggregate the
16x16 UNI token grid onto a 14-px lattice, and emit one smoothed float16 `.npz` per patch scale
(`emb_{56}` and `emb_{14}` by default).

### 1.1 argparse arguments (lines 22–42)

| Flag | Short | Type | Default | Help |
|---|---|---|---|---|
| `--project_path` | `-p` | `str` | `'./'` | `Main directory path` |
| `--code_path` | `-c` | `str` | `'./AURORA/code/'` | `Code directory path` |
| `--catch_path` | — | `str` | `'./CATCH/'` | `CATCH code directory path` |
| `--batch_size` | `-b` | `int` | `128` | `Batch size for processing` |
| `--output_path` | `-o` | `str` | `'AURORA_interim'` | `Output directory` |
| `--patch_sizes` | — | `int`, `nargs='+'` | `[56, 14]` | `List of patch sizes` |
| `--metadata_path` | `-m` | `str` | `'Sample_metadata.csv'` | `Sample metadata CSV file name` |
| `--sample_name_col` | — | `int` | `0` | `Column index for sample names in metadata CSV` |
| `--plot` | — | `store_true` | `False` | `Plot intermediate results for checking` |
| `--num_workers` | — | `int` | `4` | `Number of workers for data loading` |
| `--inference` | — | `store_true` | `False` | `Inference mode flag` |
| `--stride` | — | `int` | `112` | `Stride for patch extraction` |
| `--UNI_local_path` | — | `str` | `None` | `Path to the local UNI model folder` |
| `--CATCH_config_templet_path` | — | `str` | `'CATCH_config_templet.yaml'` | `Path to the CATCH config file` |
| `--use-CATCH` | — | `store_true` | `False` | `Use CATCH for feature extraction` (stored as `args.use_CATCH`) |
| `--compile-mode` | — | `str` | `'max-autotune'` | `torch.compile mode for the UNI extractor ("off" disables compilation; first batch is always slow JIT)` — `choices=['max-autotune','reduce-overhead','max-autotune-no-cudagraphs','default','off']` |

Commented-out (not registered): `--processed_data_path`, `--image_path`, `--seg_path` (lines 31, 36, 37).
Derived: `main_dir = args.project_path` (46), `save_dir = f"{main_dir}/{args.output_path}/UNI_multiscale_patches"` (47),
`patch_sizes = args.patch_sizes` (48), `inference_flag = args.inference` (49). `--UNI_local_path` is mandatory:
`raise ValueError("--UNI_local_path must be provided.")` (52–53, and again 175–176).

### 1.2 Input paths READ

| Line | Literal expression | Producer |
|---|---|---|
| 60 | `pd.read_csv(f"{main_dir}/{args.metadata_path}", index_col=0, header=0)` | user (`Sample_metadata.csv`) |
| 63 | `pd.read_csv(f"{main_dir}/{args.metadata_path}", header=0)` (with `--inference`) | user |
| 72 / 221 | `os.path.exists(f"{save_dir}/emb_{min(patch_sizes)}/{image_name}.npz")` — skip guard | itself (rerun cache) |
| 75 | `tifffile.imread(f"{save_dir}/wsis_rescaled/{image_name}_rescaled.tif")` | `rescale.py` (**low-res**) |
| 76 | `tifffile.imread(f"{save_dir}/wsis_rescaled/{image_name}_mask_rescaled.tif")` | `rescale.py` (**low-res**) |
| 114 | `open(f"{args.CATCH_config_templet_path}", 'r')` (CATCH branch; resolved against CWD, not `P`) | user |
| 120 | `config['global']['weight_dir'] = args.UNI_local_path` → timm `local-dir:{token}` (`image_utils.py:101`) | UNI2-h weights on disk |
| 133 / 150 | `np.load(f"{save_dir}/emb_{patch_size}/{image_name}.npz")` (CATCH branch re-reads its own output) | itself |
| 225 | `tifffile.imread(f"{save_dir}/wsis_rescaled/{image_name}_rescaled.tif")` (UNI branch) | `rescale.py` |
| 226 | `tifffile.imread(f"{save_dir}/wsis_rescaled/{image_name}_mask_rescaled.tif")` (UNI branch) | `rescale.py` |
| 228 | `np.load(f"{save_dir}/wsis_rescaled/{image_name}_loc_rescaled.npy")` (only when **not** `--inference`) | `rescale.py` |
| 229 | `np.load(f"{save_dir}/wsis_rescaled/{image_name}_pixel_size.npy")[0]` (only when **not** `--inference`) | `rescale.py` |
| 124 | `subprocess.run(f"python -u {args.code_path}/run_CATCH.py --config {save_dir}/emb_{patch_size}/CATCH_tmp/config.yaml --catch_path {args.catch_path}", shell=True, executable="/bin/bash")` | external script |

Note: `os.makedirs(f"{save_dir}/wsis_rescaled", ...)` is **commented out** (line 57) — this script never creates
the rescaled-image directory; it must already exist from `rescale.py`.

Sample list: without `--inference` the sample names are the CSV *index* (`meta.index`, line 61); with
`--inference` they are `meta.iloc[:, args.sample_name_col].values` (line 64). `image_name = image_file.split('.')[0]` (71, 220).

### 1.3 Output paths WRITTEN / created

| Line | Literal expression |
|---|---|
| 55 | `os.makedirs(f"{save_dir}", exist_ok=True)` |
| 56 | `os.makedirs(f"{save_dir}/check_figure", exist_ok=True)` |
| 102 | `os.makedirs(f"{save_dir}/emb_{patch_size}", exist_ok=True)` (CATCH branch) |
| 103 | `os.makedirs(f"{save_dir}/emb_{patch_size}/CATCH_tmp", exist_ok=True)` |
| 110 | `image_use.write_to_file(f"{save_dir}/emb_{patch_size}/CATCH_tmp/{image_name}_rescaled.tif")` |
| 111 | `mask_use.write_to_file(f"{save_dir}/emb_{patch_size}/CATCH_tmp/{image_name}_mask_rescaled.tif")` |
| 121 | `open(f"{save_dir}/emb_{patch_size}/CATCH_tmp/config.yaml", "w")` |
| 130 | `subprocess.run(f"mv {save_dir}/emb_{patch_size}/CATCH_tmp/{image_name}_emb14.npz {save_dir}/emb_{patch_size}/{image_name}.npz", ...)` |
| 131 | `subprocess.run(f"rm -r {save_dir}/emb_{patch_size}/CATCH_tmp", ...)` |
| 146 | `plt.savefig(f"{save_dir}/check_figure/{image_name}_emb_{patch_size}_kmeans.png")` (CATCH, `--plot`) |
| 162 | `np.savez(f"{save_dir}/emb_{patch_size}/{image_name}.npz", **catch_result_corrected)` (CATCH, `patch_size != 14`) |
| 241 | `cv2.imwrite(f"{save_dir}/check_figure/{image_name}_mask_check.jpg", image_check)` (`--plot`) |
| 243 | `cv2.imwrite(f"{save_dir}/check_figure/{image_name}_highres_patch_check.jpg", image_check)` (`--plot`) |
| 301 | `os.makedirs(f"{save_dir}/emb_{patch_size}", exist_ok=True)` (UNI branch) |
| 327 | `cv2.imwrite(f"{save_dir}/check_figure/{image_name}_emb_{patch_size}_kmeans.png", kmeans_img)` (`--plot`) |
| 328–332 | `np.savez(f"{save_dir}/emb_{patch_size}/{image_name}_raw.npz", emb=patch_embeddings_curr, pos=patch_positions_curr, flag=patch_flags_curr, pad=pad)` |
| 351–355 | `np.savez(f"{save_dir}/emb_{patch_size}/{image_name}.npz", emb=smoothed_embeddings, pos=patch_positions_curr, flag=patch_flags_curr, pad=pad)` ← **the deliverable** |
| 377 | `cv2.imwrite(f"{save_dir}/check_figure/{image_name}_smoothed_emb_{patch_size}_kmeans.png", kmeans_img)` (`--plot`) |

`torch.compile` also writes an on-disk kernel cache under the default `TORCHINDUCTOR_CACHE_DIR` (mentioned in the print at line 211), not under `SAVE`.

### 1.4 Third-party imports

`torch`, `pandas`, `numpy`, `tqdm`, `tifffile`, `matplotlib` (`matplotlib.pyplot`), `scikit-learn`
(`sklearn.cluster.KMeans, MiniBatchKMeans`), `PyYAML` (`yaml`), `opencv-python` (`cv2`), and `pyvips`
(imported lazily inside the `if args.use_CATCH:` branch, line 68).
Stdlib: `os`, `subprocess`, `re`, `argparse`, `gc`, `shutil`.
Local: `datasets` (`PatchDataset`), `image_utils` (`HighResFeatureExtractor`, `get_UNI_model`,
`smooth_embedding_halves`, `_PatchAccumulator`).

### 1.5 Checkpoint handling

**None.** This script never calls `torch.save` / `torch.load` and never builds a `state_dict` of its own.
The only weights it consumes are the timm/UNI2-h weights resolved by
`get_UNI_model(args.UNI_local_path, "UNI2-h", from_local=True)` (line 177) → `timm.create_model('local-dir:{path}', pretrained=True, **timm_kwargs)`
(`image_utils.py:101`). The model is set to `eval()`, `requires_grad_(False)`, grad-checkpointing off (180–189).
Optional `torch.compile(extractor, mode=args.compile_mode, dynamic=False)` (205–209) wraps
`HighResFeatureExtractor` (a `nn.Module` holding the timm model as `self.model`); nothing is serialized.

### 1.6 state_dict keys

Not applicable (no `state_dict` is created, saved or loaded). The relevant tensor contract is the feature
shape produced by `image_utils.HighResFeatureExtractor.forward` (lines 448–478): `global_emb = [B, 1536]`,
`patch_emb = [B, 1536, 16, 16]`. This script concatenates them into `[M, 3072]`
(`combined_features = torch.cat([center_features, subpixel_features], dim=1)`, line 283) and accumulates with
`_PatchAccumulator(3072)` (line 248; class at `image_utils.py:284`, `feature_dim=3072`).

`.npz` payload contract (per `emb_{patch_size}/{image_name}.npz`):
`emb` = `float16 [N, 3072]` (smoothed, mean over overlapping contributions), `pos` = `int32 [N, 2]`
(top-left pixel of each `patch_size` block, row-major sorted), `flag` = `bool [N]` (all `True` on the UNI path,
`image_utils.py:381`), `pad` = `(dataset.row_pad, dataset.col_pad)` written as a Python tuple (line 297).
`*_raw.npz` has the same keys with the unsmoothed mean (`float16`).

### 1.7 Dependency on the LOW-RESOLUTION pipeline

| Line(s) | Dependency |
|---|---|
| 57 | `# os.makedirs(f"{save_dir}/wsis_rescaled", exist_ok=True)` — commented out; the directory is owned by the low-res stage |
| 75–76 | reads `{save_dir}/wsis_rescaled/{image_name}_rescaled.tif` / `_mask_rescaled.tif` (written by `rescale.py:139,141`) |
| 225–226 | same two reads on the UNI path (the only image source for feature extraction) |
| 228–229 | reads `{image_name}_loc_rescaled.npy` (`rescale.py:144`) and `{image_name}_pixel_size.npy` (`rescale.py:147`); `--inference` sets both to `None` (231–232) |
| 60, 63 | reads the shared `Sample_metadata.csv` that `rescale.py` / `extract_features.py` also consume |

Cross-check: `rescale.py:29` computes the identical `save_dir = f"{main_dir}/{args.output_path}/UNI_multiscale_patches"`,
so `-o` must match between the two scripts.

---

## 2. `code/prepare_highres_training.py`

Purpose: despite the "highres" name, this script **runs the low-resolution AURORA model** over the training
sample list and caches its per-scale outputs as `{sample}-pred.pickle`, which is the low-res prior consumed by
the high-res training data builder (`train_utils.py:780`).

### 2.1 argparse arguments (lines 11–22)

| Flag | Short | Type | Default | Help |
|---|---|---|---|---|
| `--project_path` | `-p` | `str` | `'./'` | `Main directory path` |
| `--interim_path` | `-i` | `str` | `None` | `Interim data directory` |
| `--train_path` | `-t` | `str` | `'AURORA_train'` | `Directory to store training data` |
| `--processed_data_path` | `-d` | `str` | `'processed_data'` | `Processed data directory path` |
| `--plot` | — | `store_true` | `False` | `Plot intermediate results for checking` |
| `--parameter_json` | `-j` | `str` | `'parameters.json'` | `JSON file with parameters` |
| `--sample_sheet` | — | `str` | **required** (`required=True`) | `Training sample CSV file name` |
| `--sample_name_col` | — | `int` | `0` | `Column index for sample names in sample sheet CSV` |
| `--model_pth` | — | `str` | `'AURORA_model_weights.pth'` | `Path to the pretrained model file` |
| `--barcode_dir` | — | `str` | `'vanilla/UNI_emb'` | `Directory for barcode embeddings` |

`main_dir = args.project_path` (24); `if args.interim_path is None: args.interim_path = args.train_path` (25–26).

### 2.2 Input paths READ

| Line | Literal expression |
|---|---|
| 29 | `open(f"{main_dir}/{args.train_path}/{args.parameter_json}", 'r')` |
| 37 | `os.listdir(f"{main_dir}/{args.train_path}/processed_pkl")` (existence/list of low-res pkls) |
| 41 | `f"{main_dir}/{args.train_path}/{parameters.get('gene_list_file','AURORA_interim/Gene_to_predict.csv')}"` |
| 45 | `f"{main_dir}/{args.train_path}/{parameters.get('gene_norm_factor_file','AURORA_interim/Gene_normalize_factor.csv')}"` |
| 57 | `f"{main_dir}/{args.train_path}/{parameters.get('celltype_file','AURORA_interim/deconvolution/cell_types.csv')}"` |
| 59 | `test_samples_path = f"{main_dir}/{args.sample_sheet}"` |
| 68 | `torch.load(f"{main_dir}/{args.train_path}/{args.model_pth}", weights_only=False, map_location=device)` |
| 81 | `barcode_dir = f"{main_dir}/{args.barcode_dir}"` → `{P}/vanilla/UNI_emb/{dataset}-bars.pickle` (`train_utils.py:238`) |

Parameter keys read from `parameters` (defaults in parentheses): `img_emb_dim` (1536), `num_workers` (4),
`patch_sizes` (`[3584, 896, 224]`), `pixel_size` (0.5), `bidirectional` (False), `use_bulk` (True),
`pred_celltype` (True), `use_context_loss` (True), `gene_list_file`, `gene_norm_factor_file`, `celltype_file` (46–58).

**Transitive reads through `train_utils.load_data`** (called at lines 76–81 or 83–88; loader body is
`train_utils.py:199–452`), with `interim_dir=args.interim_path`, `train_dir=args.train_path`,
`processed_data_dir=args.processed_data_path`, `smooth_cnts=False`, `load_cell_type=False`,
`patch_folder_name="UNI_multiscale_patches"`:

| `train_utils.py` line | Literal expression |
|---|---|
| 23 | `open(f"{main_dir}/{processed_data_dir}/metadata/{dataset}.json", 'r')` |
| 30 | `f"{main_dir}/{interim_dir}/UNI_multiscale_patches/wsis_rescaled/{dataset}_rescaled.tif"` (via `tifffile.TiffFile(image_path).pages[0].asarray()`, line 34) |
| 31 | `f"{main_dir}/{interim_dir}/UNI_multiscale_patches/wsis_rescaled/{dataset}_mask_rescaled.tif"` (line 37) |
| 216 | `cnt_folder = f"{main_dir}/{interim_dir}/cnts"` |
| 219 | `bulk_folder = f"{main_dir}/{interim_dir}/bulk"` |
| 220 | `prop_folder = f"{main_dir}/{interim_dir}/deconvolution"` |
| 227 | `locs_file = f"{emb_folder}/wsis_rescaled/{dataset}_loc_rescaled.npy"` |
| 228 | `utils.load_pickle(f"{emb_folder}/wsis_rescaled/{dataset}_barcode.pickle")` |
| 238 | `utils.load_pickle(f"{barcode_dir}/{dataset}-bars.pickle")` (because `barcode_dir is not None`) |
| 252 | `np.load(f"{emb_folder}/emb_{patch_size}/{dataset}.npz")` for `patch_size in [3584, 896, 224]` (all **low-res**) |
| 397 | `cnts_file = f"{cnt_folder}/{dataset}-cnts.tsv"` |
| 412 | `bulk_file = f"{bulk_folder}/{dataset}-bulk.tsv"` |
| 425 | `max_file = f"{bulk_folder}/{dataset}-max.tsv"` |
| 450 | `utils.load_pickle(f"{file_dir}/{dataset}.pkl")` — `from_file=True` path, `file_dir = f"{main_dir}/{args.train_path}/processed_pkl"` (line 35) |

`emb_folder = f"{main_dir}/{interim_dir}/UNI_multiscale_patches"` (`train_utils.py:214`).
`props/deconvolution` is skipped here because `load_cell_type=False` is passed (line 79).

### 2.3 Output paths WRITTEN / created

| Line | Literal expression |
|---|---|
| 72 | `os.makedirs(f"{main_dir}/{args.train_path}/train_pred_lowres", exist_ok=True)` |
| 129 | `utils.save_pickle({'output':output,'major_prop_pred':major_prop_pred,'latent_emb':latent_emb}, f"{main_dir}/{args.train_path}/train_pred_lowres/{dataset_test[index]}-pred.pickle")` |
| 211 (`train_utils`) | `os.makedirs(f"{main_dir}/{train_dir}/image_check/", exist_ok=True)` (only on the `from_file=False` path) |
| 317 (`train_utils`) | `cv2.imwrite(f"{main_dir}/{train_dir}/image_check/{dataset}_{patch_size}patch.png", ...)` when `--plot` |
| 440 (`train_utils`) | `utils.save_pickle(data_tmp, f"{file_dir}/{dataset.replace('.h5','')}.pkl")` — writes `{P}/{T}/processed_pkl/{dataset}.pkl` when the cache is missing |

Pickle payload keys: `output` (numpy, low-res gene prediction), `major_prop_pred` (numpy),
`latent_emb` (list of numpy arrays, one per LSTM position/scale), written with plain `pickle.dump`
(`utils.py:67–71`). Consumers index scale 2 (the finest, `224`): `train_utils.py:836,838`.

### 2.4 Third-party imports

`numpy`, `pandas`, `torch`, `tqdm`.
Stdlib: `os`, `argparse`, `json`.
Local: `train_utils` (`load_data`, `get_unique_indices` — the latter is imported but unused),
`model` (`AURORA`), `utils` (`save_pickle`).

### 2.5 Checkpoint handling

```python
# line 63-69
model = AURORA(input_size, output_size, bulk_size=len(hvg_list),
                celltype_num=len(major_cell_type),
                device=device, bidirectional=bidirectional,
                use_bulk=use_bulk, pred_celltype=pred_celltype)
model.to(device)
state_dict = torch.load(f"{main_dir}/{args.train_path}/{args.model_pth}", weights_only=False, map_location=device)
model.load_state_dict(state_dict)
```

* Loaded object: a plain `state_dict` (the low-res trainer `train.py` writes one).
* `torch.load`: `weights_only=False`, `map_location=device` (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, line 32).
* `load_state_dict(state_dict)` with no `strict` argument → default `strict=True`.
* Nothing is saved here. `latent_size` is left at the `AURORA` default (`256`), and `finetune=False`
  (so the module exposes `decoder`, not `new_decoder`).
* The default `--model_pth` is the **low-res** checkpoint name; `command.sh` line 13 passes `--model_pth AURORA_BRCA.pth`.

### 2.6 state_dict keys

The same `AURORA` key set as §3.6, but with `input_size = parameters.get('img_emb_dim', 1536)` (line 46),
i.e. `lstm.weight_ih_l0` has shape `(1024, 1536)` instead of `(1024, 3072)`; `latent_size = 256`;
`output_size = len(gene_list)`; `bulk_size = len(hvg_list) == len(gene_list)`;
`celltype_num = len(major_cell_type)`.

### 2.7 Dependency on the LOW-RESOLUTION pipeline

Essentially **every** input of this script is a low-res artifact:

| Line | Dependency |
|---|---|
| 35, 37 | `{P}/{T}/processed_pkl` — the **low-res** superpatch/pkl cache written by `train.py` (contrast with `train_highres.py:38`, which uses `processed_pkl_highres`) |
| 41, 45, 57 | `Gene_to_predict.csv`, `Gene_normalize_factor.csv`, `deconvolution/cell_types.csv` — all `AURORA_interim` low-res products |
| 68 | loads the **low-res** model weights (`AURORA_model_weights.pth` / `AURORA_BRCA.pth`) |
| 68 (`weights_only=False`) | required because the file is a pickled tensor mapping, not a module; low-res `train.py` writes it |
| 76–81 | `from_file=True` → `train_utils.load_data` reads the low-res pkls |
| 83–88 | `from_file=False` → reads `emb_{3584,896,224}`, `wsis_rescaled/*`, `cnts/*`, `bulk/*`, `vanilla/UNI_emb/*-bars.pickle` (all low-res or rescale outputs) |
| 122–124 | assumes exactly **3** patch sizes, indexing `emb[patch_sizes[0]]`, `emb[patch_sizes[1]]`, `emb[patch_sizes[2]]` |
| 129 | output `train_pred_lowres/{sample}-pred.pickle` is the hand-off to the high-res loader (`train_utils.py:780`) |

---

## 3. `code/train_highres.py`

Purpose: train the high-res AURORA (2-scale hierarchy `[56, 14]`, 3072-d embeddings) on superpatches built
from the high-res `.npz` files plus the low-res prior pickle.

### 3.1 argparse arguments (lines 14–26)

| Flag | Short | Type | Default | Help |
|---|---|---|---|---|
| `--project_path` | `-p` | `str` | `'./'` | `Main directory path` |
| `--interim_path` | `-i` | `str` | `'AURORA_interim'` | `Interim data directory` |
| `--train_path` | `-t` | `str` | `'AURORA_train'` | `Directory to store training data` |
| `--processed_data_path` | `-d` | `str` | `'processed_data'` | `Processed data directory path` |
| `--plot` | — | `store_true` | `False` | `Plot intermediate results for checking` |
| `--log` | — | `store_true` | `False` | `Enable logging to TensorBoard` |
| `--parameter_json` | `-j` | `str` | `'parameters.json'` | `JSON file with parameters` |
| `--train_samples` | — | `str` | `'Sample_metadata.csv'` | `CSV file with training sample metadata` |
| `--train_sample_col` | — | `int` | `0` | `Column index for training sample IDs in the metadata CSV` |

Commented out: `--patch_sizes`, `--metadata_path` (18, 19). `main_dir = args.project_path` (28).

### 3.2 Input paths READ

| Line | Literal expression |
|---|---|
| 30 | `open(f"{main_dir}/{args.train_path}/{args.parameter_json}", 'r')` |
| 38 | `dataset_dir = f"{main_dir}/{args.train_path}/processed_pkl_highres"` (+ `os.path.exists`, 93; `os.listdir`, 94) |
| 41 | `f"{main_dir}/{args.train_path}/{parameters.get('gene_list_file','AURORA_interim/Gene_to_predict.csv')}"` |
| 45 | `f"{main_dir}/{args.train_path}/{parameters.get('gene_norm_factor_file','AURORA_interim/Gene_normalize_factor.csv')}"` |
| 79 | `train_samples_path = f"{main_dir}/{args.train_samples}"` (**root-relative**, marked `# !!! change here`) |
| 81 | `f"{main_dir}/{args.train_path}/{parameters.get('celltype_file','AURORA_interim/deconvolution/cell_types.csv')}"` |
| 86 | `pd.read_csv(train_samples_path).iloc[:, sample_col].values` |
| 95–96 | `pd.read_csv(f"{dataset_dir}/superpatch_meta.csv")` |
| 98–100 | `open(f"{dataset_dir}/superpatch_info.json", 'r')` |
| 141 | `torch.load(f"{main_dir}/{args.train_path}/checkpoints_highres/AURORA_{exp_epochs}.pth")` (only when `retrain_cellprop_only`) |

Parameter keys (defaults): `retrain_cellprop_only` (False), `gene_list_file`, `gene_norm_factor_file`,
`img_emb_dim` (1536*2 = 3072), `batch_size` (4), `num_workers` (4), `patch_sizes` (`[56, 14]`),
`mse_weight` (10), `prop_weight` (0.1), `focal_gamma` (0.5), `num_epochs` (600), `exp_epochs` (400),
`pixel_size` (0.5), `lr` (1e-3), `lr_prop` (1e-3), `weight_decay` (1e-5), `stepLR_step_exp` (200),
`stepLR_gamma_exp` (0.1), `stepLR_step_prop` (50), `stepLR_gamma_prop` (0.5), `bidirectional` (False),
`use_bulk` (True), `pred_celltype` (True), `use_context_loss` (True), `superpatch_size` (5000),
`superpatch_stride` (4000), `superpatch_minsize` (1000), `superpatch_minlowres_patch` (10),
`prefetch_batches` (1), `celltype_file` (33–72).

**Transitive reads through `train_utils.load_data2`** (`train_utils.py:706–1052`, called at lines 108–112 /
115–119 with `file_dir=dataset_dir`, `interim_dir=args.interim_path`, `train_dir=args.train_path`,
`pixel_size`, `patch_sizes`, `superpatch_*`, `load_cell_type=pred_celltype`):

| `train_utils.py` line | Literal expression | Stage |
|---|---|---|
| 23 | `open(f"{main_dir}/{processed_data_dir}/metadata/{dataset}.json", 'r')` | input metadata |
| 30, 31 | `f"{main_dir}/{interim_dir}/UNI_multiscale_patches/wsis_rescaled/{dataset}_rescaled.tif"` and `..._mask_rescaled.tif` | **low-res (rescale.py)** |
| 731 | `emb_folder = f"{main_dir}/{interim_dir}/UNI_multiscale_patches"` | — |
| 779 | `np.load(f"{emb_folder}/emb_224/{dataset}.npz")` — used as the low-res anchor grid | **low-res** |
| 780 | `utils.load_pickle(f"{main_dir}/{train_dir}/train_pred_lowres/{dataset}-pred.pickle")` | **low-res predictions** |
| 793 | `np.load(f"{emb_folder}/emb_{patch_size}/{dataset}.npz")` for `patch_size in [56, 14]` | high-res (own stage) |
| 1004 | `bulk_file = f"{bulk_folder}/{dataset}-bulk.tsv"` (`bulk_folder = f"{main_dir}/{interim_dir}/bulk"`, 732) | **low-res** |
| 720, 723 | `pd.read_csv(f"{file_dir}/superpatch_meta.csv")` (resume support) | own cache |
| 761, 807 | `os.path.exists(f"{file_dir}/{dataset}_superpatch_{superpatch_id}.pkl")` / `{name}.pkl` | own cache |
| 1044 | `pd.read_csv(f"{file_dir}/superpatch_meta.csv")` (`from_file=True`) | own cache |
| 540 (`SuperpatchCollection`) | `utils.load_pickle(f"{self.file_dir}/{name}.pkl")` | own cache |

`SuperpatchCollection` is returned by `load_data2` and wrapped by
`HistologyGeneExpressionDataset(train_data, device=device, highres=True)` (line 120; dataset class at
`datasets.py:409–500`, `highres=True` forces CPU storage, `dtype_switch = torch.int32` used only for `adj_mat`).

### 3.3 Output paths WRITTEN / created

| Line | Literal expression |
|---|---|
| 104 | `os.makedirs(dataset_dir, exist_ok=True)` (only when `dataset_dir` does not exist) |
| 137 | `SummaryWriter(f"{main_dir}/{args.train_path}/train_log_highres")` (only with `--log`) |
| 139 | `os.makedirs(f"{main_dir}/{args.train_path}/checkpoints_highres", exist_ok=True)` |
| 376–377 | `filename = f"{main_dir}/{args.train_path}/checkpoints_highres/AURORA_{epoch}.pth"`; `torch.save(model, filename)` — every `epoch % 10 == 0` |
| 387 | `torch.save(state_dict, f"{main_dir}/{args.train_path}/AURORA_highres_model_weights.pth")` — final, unconditional |
| `train_utils` 714 | `os.makedirs(f"{main_dir}/{train_dir}/image_check/", exist_ok=True)` (sketch path only) |
| `train_utils` 730, 734 | `pd.DataFrame(superpatch_meta).to_csv(f"{file_dir}/superpatch_meta.csv")`; `open(f"{file_dir}/superpatch_info.json", 'w')` |
| `train_utils` 914, 962, 988 | `cv2.imwrite(f"{main_dir}/{train_dir}/image_check/{dataset}_superpatch_{superpatch_id}_{patch_size}patch.png" / `..._spotweight_{patch_size}patch.png` / `..._father_{patch_size}patch.png`, ...)` when `--plot` |
| `train_utils` 1020 | `utils.save_pickle(data_tmp, f"{file_dir}/{name}.pkl")` with `name = f"{dataset}_superpatch_{superpatch_id}"` |

### 3.4 Third-party imports

`pandas`, `torch` (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data.DataLoader`), `tqdm`,
`tensorboard` (`from torch.utils.tensorboard import SummaryWriter`).
Stdlib: `os`, `argparse`, `json`.
Local: `train_utils` (`load_data2`, `masked_focal_kl_loss`, `get_unique_indices`, `project_spot_weight`,
`project_spot_weight_columns`, `PrefetchLoader`), `datasets` (`HistologyGeneExpressionDataset`, `my_collate`),
`model` (`AURORA`).

### 3.5 Checkpoint handling

| | Periodic checkpoint | Final checkpoint |
|---|---|---|
| Line | 376–377 | 386–387 |
| Call | `torch.save(model, f"{main_dir}/{args.train_path}/checkpoints_highres/AURORA_{epoch}.pth")` | `state_dict = model.state_dict()` then `torch.save(state_dict, f"{main_dir}/{args.train_path}/AURORA_highres_model_weights.pth")` |
| Object | the **whole `AURORA` nn.Module** (pickled class + tensors + `self.device`) | plain `OrderedDict[str, Tensor]` |
| Cadence | `if epoch % 10 == 0` (epochs 0, 10, 20, …; the filename uses `epoch`, not `epoch+1`) | once, after the loop (`print("Training complete")`, 381) |
| Extra metadata | none — no optimizer, scheduler, epoch, or `parameters.json` snapshot | none — no shapes/dims stored; the consumer must rebuild `AURORA` with the exact `parameters.json` |
| Atomicity | direct write, no temp-file/rename | direct write, no temp-file/rename |

Loading inside this script (only when `retrain_cellprop_only`):

```python
# line 140-144
if retrain_cellprop_only:
    model = torch.load(f"{main_dir}/{args.train_path}/checkpoints_highres/AURORA_{exp_epochs}.pth")
    start_epoch = exp_epochs
else:
    start_epoch = 0
```

* `torch.load` is called with **no** `map_location` and **no** `weights_only` argument.
* No `load_state_dict` at all — the whole module object replaces `model`.
* `model.to(device)` was already called at line 126, *before* this reload; the reloaded module keeps whatever
  device its tensors were pickled on.
* The optimizer/scheduler at lines 130/134 are constructed from the *pre-reload* model; the first iteration of
  the retrain run hits `epoch == exp_epochs` (line 154) and rebuilds both over `model.celltype_decoder.parameters()`,
  so the stale optimizer is discarded immediately.
* The optimizer state itself is never checkpointed — resuming a run always restarts Adam moments, LR schedule
  and `focal_gamma`.

**Note on `weights_only`:** the periodic `.pth` files contain a full module object. `torch.load` defaults to
`weights_only=True` from PyTorch 2.6, so line 141 will raise `UnpicklingError` on modern torch unless the default
is overridden. `requirements.txt` does not pin `torch`, so this is version-dependent.

### 3.6 state_dict keys (derivable from `model.py:4–74`)

`AURORA.__init__` is called at lines 122–125 with `input_size = parameters.get('img_emb_dim', 1536*2)`,
`output_size = len(gene_list)`, `bulk_size = len(hvg_list)`, `celltype_num = len(major_cell_type)`,
`latent_size=256`, `device`, `bidirectional` (default `False`), `use_bulk=True`, `pred_celltype=True`,
`finetune=False`. With `latent_size = 256` the module is `nn.Sequential`-based with ReLU-free `LeakyReLU`
activations, so parameter indices skip activation positions. There are **no buffers** (no BatchNorm/Dropout/
`num_batches_tracked` anywhere in `AURORA`), so the state_dict contains exactly 18 tensors when
`bidirectional=False` and `pred_celltype=True`:

| Key | Shape | Source |
|---|---|---|
| `bulk_encoder.0.weight` | `(256, bulk_size)` | `nn.Linear(bulk_size, latent_size)` + `LeakyReLU` (`model.py:22–25`) |
| `bulk_encoder.0.bias` | `(256,)` | ″ |
| `lstm.weight_ih_l0` | `(1024, input_size)` = `(1024, 3072)` by default | `nn.LSTM(input_size, 256, batch_first=True, bidirectional=False)` (`model.py:27`) |
| `lstm.weight_hh_l0` | `(1024, 256)` | ″ |
| `lstm.bias_ih_l0` | `(1024,)` | ″ |
| `lstm.bias_hh_l0` | `(1024,)` | ″ |
| `decoder.0.weight` / `.bias` | `(128, 256)` / `(128,)` | `Linear(latent_size, latent_size//2)` at Sequential index 0 (`model.py:42–50`) |
| `decoder.2.weight` / `.bias` | `(256, 128)` / `(256,)` | Sequential index 2 |
| `decoder.4.weight` / `.bias` | `(output_size, 256)` / `(output_size,)` | Sequential index 4 (+ `Sigmoid`) |
| `celltype_decoder.0.weight` / `.bias` | `(64, 256)` / `(64,)` | `Linear(latent_size, max(latent_size//4,32))` (`model.py:63–73`) |
| `celltype_decoder.2.weight` / `.bias` | `(32, 64)` / `(32,)` | Sequential index 2 |
| `celltype_decoder.4.weight` / `.bias` | `(celltype_num, 32)` / `(celltype_num,)` | Sequential index 4 (+ `LogSoftmax`) |

If `bidirectional=True` (parameter `bidirectional`), the decoder becomes the 4-layer variant
(`model.py:29–40`) with keys at indices `0, 2, 4, 6` (`decoder.6.weight (output_size, 256)`) and
`lstm.weight_ih_l0` shape `(1024, input_size)` with additional `_reverse` parameters
(`lstm.weight_ih_l0_reverse`, `lstm.weight_hh_l0_reverse`, `lstm.bias_ih_l0_reverse`, `lstm.bias_hh_l0_reverse`).
If `finetune=True`, the key prefix is `new_decoder.*` instead of `decoder.*` (not used by this script).

**Verified against the shipped artifact** `trained_model/AURORA_highres_model_weights.pth`
(a `torch.save` v3 zip: 20,146,232 bytes on disk, 22,280-byte `data.pkl`, root object `collections.OrderedDict`,
globals `torch._utils._rebuild_tensor_v2` + `torch.FloatStorage`, 15 unique float32 storages, 20,139,920 bytes total —
i.e. the `state_dict` write of line 387, **not** a pickled module): exactly the 18 keys above, all `torch.float32`,
with `bulk_encoder.0.weight (256, 3002)`, `lstm.weight_ih_l0 (1024, 3072)`, `decoder.4.weight (3002, 256)`,
`celltype_decoder.4.weight (10, 32)`. A consumer therefore has to instantiate
`AURORA(input_size=3072, output_size=3002, bulk_size=3002, celltype_num=10, latent_size=256, bidirectional=False)`.

For reference, the low-res checkpoint read by `prepare_highres_training.py` uses the identical key names but
`lstm.weight_ih_l0` is `(1024, 1536)`; it is **not** loadable into the high-res model built here
(`input_size = 3072`), and `train_highres.py` never attempts to load it — the high-res run starts from random
initialization unless `retrain_cellprop_only=True`.

### 3.7 Dependency on the LOW-RESOLUTION pipeline

| Line | Dependency |
|---|---|
| 38 vs 35 | `processed_pkl_highres` (high-res) instead of `processed_pkl` (low-res) — the two training scripts never share this cache |
| 41, 45, 81 | `Gene_to_predict.csv`, `Gene_normalize_factor.csv`, `deconvolution/cell_types.csv` from the low-res interim dir |
| 108–112 | `load_data2(..., from_file=True)` reads the high-res superpatch cache, whose payload was built from low-res inputs |
| 115–119 | `load_data2(..., from_file=False)` rebuilds the cache and, at `train_utils.py:779`, reads `{EMB}/emb_224/{dataset}.npz` — the **low-res 224 scale** produced by `extract_features.py` |
| `train_utils.py:780` | reads `{main_dir}/{train_dir}/train_pred_lowres/{dataset}-pred.pickle` — produced by `prepare_highres_training.py`; supplies `cnts_filtered` (gene prior) and `major_prop_filtered = np.exp(lowres_cellprop)` (`train_utils.py:995, 999`) |
| `train_utils.py:836–839` | `lowres_result['major_prop_pred'][:,2,:]` and `lowres_result['output'][:,2,:]` — hard-coded scale index **2** (the finest low-res scale, `224`) |
| `train_utils.py:838` | `lowres_geneexp` becomes the training target `cnts_filtered`, with `gene_mask` all-`True` (`train_utils.py:1001`) |
| `train_utils.py:30–31` | `wsis_rescaled` TIFFs from `rescale.py` (low-res stage) for the `--plot` figures and for `superpatch_num` sizing (line 744) |
| `train_utils.py:1004` | `bulk/{dataset}-bulk.tsv` from `prepare_st.py` (low-res stage) |
| `train_utils.py:990` | `father_lowres = find_nearby_spots(lowres_loc + 224//2, patches_loc_before_pad[patch_size_min] + 7, k=1)` — the low-res (224) to high-res (14) parent map; the 224 half-width is hard-coded |

Consequence: `patch_sizes` in this script's `parameters.json` must contain **exactly two** entries
(lines 207–208 index `patch_sizes[0]` and `patch_sizes[1]`, and 216/254 reverse them), while the low-res
`parameters.json` used by `prepare_highres_training.py` needs **three** (its lines 122–124) and
`train_utils.py:836–839` hard-codes index `2`. The two parameter files are different files
(`model_parameter_BRCA.json` vs `model_parameter_BRCA_highres.json` in `command.sh`).

---

## 4. `code/highres_feature_ops.py` (memory-bounded aggregation/smoothing primitives)

### 4.1 argparse arguments
**None.** The module has no `argparse`, no `if __name__ == "__main__"` block, and takes all knobs as
constructor/function keywords (`base_patch_size=14`, `chunk_budget=100_000`, `channel_chunk=64/128`,
`first_size=16`, `second_size=4`, `validate_unique=False`).

### 4.2 Input paths READ
**None.** The module never touches the filesystem.

### 4.3 Output paths WRITTEN / created
**None.** No `open`, `savez`, `imwrite`, `makedirs`, or logging to disk.

### 4.4 Third-party imports
`cv2` (OpenCV, used for the `cv2.blur` box filters at lines 569, 591), `numpy`.
Stdlib: `__future__` (`annotations`), `dataclasses`, `typing`.
Local: none. (Notably **no `torch` import** — this is a pure NumPy/OpenCV implementation, unlike its legacy
counterpart `image_utils.smooth_embeddings`.)

### 4.5 Checkpoint handling
**None.** No `torch`, therefore no `torch.save`/`torch.load`/`state_dict`.

### 4.6 state_dict keys
Not applicable. The relevant contracts are:
* `AggregatedGrid(grid_positions int32 (N,2), sums float32 (N, feature_dim), counts int64 (N,), base_patch_size=14)`
  with the invariant `all(counts > 0)` (`__post_init__`, 33–58);
* `SplitGridFeatureAccumulator.merge(grid_positions, global_embeddings, spatial_tokens, batch_indices, token_indices)` (115–234);
* `finalize(patch_size, base_patch_size=None)` → `(embeddings float16 (N, feature_dim), positions int32 (N,2), flags bool (N))`
  via `aggregated_grid_to_output` (462–484); requires `patch_size % base_patch_size == 0` (313–316);
* `smooth_embedding_halves(...)` hard-asserts `embeddings.shape[1] == 1536*2 == 3072` (623–626) and smooths
  the two 1536-d halves with independent kernels.

### 4.7 Dependency on the LOW-RESOLUTION pipeline
**None** (no I/O at all). It only encodes the **high-res** 14-px-token / 3072-d convention
(`base_patch_size: int = 14` at line 31, `1536*2` at 623).

**Consumers** (grep-verified): `image_utils_optimized.py:9–13` imports `SplitGridFeatureAccumulator`,
`smooth_embedding_halves as _smooth_embedding_halves_fast`, `smooth_embeddings as _smooth_embeddings_fast`,
which is in turn used by `extract_features_highres_optimized.py:38–43`. **None of the other five analysed
scripts imports this module** — `extract_features_highres.py:18` imports `smooth_embedding_halves` from
`image_utils` (the legacy NumPy/`np.add.at` version, `image_utils.py:213–221`), not from here.

---

## 5. `code/highres_grid_pipeline.py` (regular-grid GPU strip accumulator)

### 5.1 argparse arguments
**None** (no CLI; a library module).

### 5.2 Input paths READ
**None** — pure in-memory. It receives already-loaded CUDA tensors
(`global_features [B, G]`, `spatial_features [B, N, S]`, `positions`, `keep_flag [B, 16, 16]`).

### 5.3 Output paths WRITTEN / created
**None.** `flush_before()` / `finish()` return `FineGridStrip` dataclass instances
(`row_start`, `global_sums float32 (rows, cols, G)`, `spatial_sums float32 (rows, cols, S)`,
`counts int32 (rows, cols) | None`) to the caller; nothing is persisted.

### 5.4 Third-party imports
`torch` (only).
Stdlib: `__future__` (`annotations`), `dataclasses`, `typing`.
Local: none.

### 5.5 Checkpoint handling
**None** (no `torch.save`/`torch.load`; accumulates `torch.Tensor`s only).

### 5.6 state_dict keys
Not applicable. Contract details worth recording:
* constructor knobs (86–98): `grid_shape`, `global_dim`, `spatial_dim`, `strip_rows`, `token_size=14`,
  `token_grid_shape=(16,16)`, `contribution_chunk=8192`, `device=None`, `track_counts=False`;
* `accumulate(...)` (195–380) rounds model outputs to **float16 first, then promotes to float32** for
  `index_add_` (307–308, 354–376) so sums reproduce the legacy fp16 transfer path;
* routing is solved on the **host** (CPU `nonzero`, integer div/remainder, 246–306) to avoid CUDA syncs;
* hard invariants: patch positions must be multiples of `token_size` (252–255), destinations must lie inside
  `grid_shape` (287–294), and rows must arrive in **row-major order** — a contribution landing in an
  already-flushed strip raises `RuntimeError("received an already-flushed regular-grid contribution")` (295–299);
* `counts` are only tracked when `track_counts=True`.

### 5.7 Dependency on the LOW-RESOLUTION pipeline
**None** for I/O. The geometry is derived exclusively from the high-res contract: `token_size = 14`
(`RegularGridStripAccumulator.__init__`, line 93) and `token_grid_shape=(16,16)` (line 94), matching the 224-px
patch / 14-px token grid of `extract_features_highres.py` (lines 238, 290).

**Consumer:** `highres_gpu_strip_pipeline.py:27–29` imports `FineGridStripAccumulator` (aliased
`_FineGridStripAccumulator`) and re-exports its own `RegularGridStripAccumulator` (line 772); that module is
used by `extract_features_highres_optimized.py:32–37`. **Not imported by any of the other five analysed scripts.**

---

## 6. `code/highres_inference_pipeline.py` (batching / transfer / CPU accumulation helpers)

### 6.1 argparse arguments
**None** (library module; all behaviour is keyword-parameterised:
`ActivePatchBatcher(source, batch_size, pin_memory=False, on_skipped_patches=None)`,
`CpuAccumulatorPipeline(fine_state, direct_states, token_size, max_pending=2, on_processed_patches=None)`,
`stage_valid_feature_batch(*, global_features, spatial_features, batch_indices, token_indices, pixel_positions, copy_stream)`).

### 6.2 Input paths READ
**None** — it consumes in-memory tensors/arrays only.

### 6.3 Output paths WRITTEN / created
**None.** It returns `ActivePatchBatch`, `CpuFeatureBatch`, and `ActiveBatchStats` objects, and mutates the
caller-supplied accumulator objects (`fine_state`, `direct_states`) via `.merge(...)` (489–504).
The only non-memory side effect is a spawned daemon thread (`threading.Thread(name="highres-cpu-accumulator")`,
434–439) and pinned-host allocations (`torch.empty(..., pin_memory=True)`, 359–370).

### 6.4 Third-party imports
`numpy`, `torch`.
Stdlib: `__future__` (`annotations`), `collections` (`deque`), `queue`, `threading`, `dataclasses`, `typing`.
Local: none.

### 6.5 Checkpoint handling
**None** (no serialization of models or tensors to disk).

### 6.6 state_dict keys
Not applicable. Behavioural contracts:
* `ActivePatchBatcher` **requires CPU tensors** — it raises `ValueError` for non-CPU `patches`/`keep_flag`/
  positions (118–121), filters patches whose `keep_flag` is entirely `False` (217, 226–227), and re-packs
  survivors across source-batch boundaries so every emitted batch except the tail has exactly `batch_size` rows
  (240–249);
* `stage_valid_feature_batch` gathers valid tokens on-device (`index_select` over the flattened
  `[B*N, S]` grid, 340–342), converts to float16, and copies to pinned host memory on a side CUDA stream with a
  readiness `torch.cuda.Event` (357–386); the event is synchronised **by the worker thread**, not the caller;
* `CpuAccumulatorPipeline` owns all accumulator mutation in one thread (`_process`, 471–516) and merges with
  `pixel_positions // token_size` for the fine state and `pixel_positions // int(patch_size)` for each direct
  state (489–504) — the same `// patch_size` convention as `extract_features_highres.py:294`;
* the bounded queue (`maxsize=max_pending=2`) is the back-pressure mechanism; worker exceptions are re-raised
  from `submit()`/`finish()` (450–453, 543).

### 6.7 Dependency on the LOW-RESOLUTION pipeline
**None** (no I/O). It encodes the high-res conventions `token_grid_shape`/`token_count` (14-px token lattice)
and the 3072-d `(global || spatial)` merge.

**Consumer:** `extract_features_highres_optimized.py:27–31`. **Not imported by any of the other five analysed scripts.**

---

## 7. Cross-script data flow

Ordered chain of directories produced and consumed by the high-resolution stage (low-res producers shown for context,
since the high-res stage cannot run without them). All paths are relative to `P = args.project_path`.

```
0. INPUTS (user)
   {P}/Sample_metadata.csv                              (metadata_path / train_samples)
   {P}/Training_samples.csv, {P}/Inference_samples.csv  (sample_sheet)
   {P}/{D}/metadata/{sample}.json                        spot_diameter (rescale + patch sizing)
   {P}/<UNI2-h weights dir>/                             --UNI_local_path (timm local-dir:)
   <CATCH_config_templet.yaml>                           CWD-relative

1. LOW-RES  code/rescale.py  -o O
   {P}/{O}/UNI_multiscale_patches/wsis_rescaled/{s}_rescaled.tif
   {P}/{O}/UNI_multiscale_patches/wsis_rescaled/{s}_mask_rescaled.tif
   {P}/{O}/UNI_multiscale_patches/wsis_rescaled/{s}_loc_rescaled.npy
   {P}/{O}/UNI_multiscale_patches/wsis_rescaled/{s}_barcode.pickle
   {P}/{O}/UNI_multiscale_patches/wsis_rescaled/{s}_pixel_size.npy
   {P}/{O}/UNI_multiscale_patches/rescale_parameters_{ts}.csv
      |
      +--> consumed by extract_features_highres.py (lines 75-76, 225-229)
      +--> consumed by extract_features.py, prepare_st.py, train_utils.load_image / load_data / load_data2
      |
2. LOW-RES  code/extract_features.py  -o O   [default patch_sizes 3584 896 224]
   {P}/{O}/UNI_multiscale_patches/emb_3584/{s}.npz
   {P}/{O}/UNI_multiscale_patches/emb_896/{s}.npz
   {P}/{O}/UNI_multiscale_patches/emb_224/{s}.npz
   {P}/{O}/UNI_multiscale_patches/locs/{s}-bars.pickle
      |
      +--> emb_224 read by train_utils.load_data2:779  (low-res anchor used by HIGH-RES training)
      |
3. LOW-RES  code/prepare_st.py  -o O  (+ get_deconvolution.R)
   {P}/{O}/cnts/{s}-cnts.tsv          {P}/{O}/cnts_smooth/{s}-cnts-smooth.tsv
   {P}/{O}/bulk/{s}-bulk.tsv          {P}/{O}/bulk/{s}-bulk-rank.tsv   {P}/{O}/bulk/{s}-max.tsv
   {P}/{O}/Gene_to_predict.csv        {P}/{O}/Gene_normalize_factor.csv
   {P}/{O}/deconvolution/{s}_prop.csv {P}/{O}/deconvolution/cell_types.csv
      |
      +--> bulk/{s}-bulk.tsv read by train_utils.load_data2:1004 (high-res training)
      |
4. LOW-RES  code/train.py  -t T
   {P}/{T}/AURORA_model_weights.pth (or AURORA_<cohort>.pth)
   {P}/{T}/processed_pkl/{s}.pkl
      |
      +--> weights loaded by prepare_highres_training.py:68 ; pkls read via load_data(from_file=True)

5. HIGH-RES  code/extract_features_highres.py  -o O  (default patch_sizes 56 14, stride 112)
   reads  {P}/{O}/UNI_multiscale_patches/wsis_rescaled/*          (step 1)
   writes {P}/{O}/UNI_multiscale_patches/emb_56/{s}.npz
          {P}/{O}/UNI_multiscale_patches/emb_56/{s}_raw.npz
          {P}/{O}/UNI_multiscale_patches/emb_14/{s}.npz
          {P}/{O}/UNI_multiscale_patches/emb_14/{s}_raw.npz
          {P}/{O}/UNI_multiscale_patches/check_figure/*.png|jpg    (--plot)

6. BRIDGE  code/prepare_highres_training.py  -t T  (must use the SAME dir as -o above)
   reads  {P}/{T}/{parameters.json}, Gene_to_predict.csv, Gene_normalize_factor.csv,
          deconvolution/cell_types.csv, {P}/{T}/processed_pkl, {P}/{T}/{model_pth},
          emb_{3584,896,224}, cnts/*, bulk/*, wsis_rescaled/*, vanilla/UNI_emb/*-bars.pickle
   writes {P}/{T}/train_pred_lowres/{sample}-pred.pickle           <-- low-res prior
          {P}/{T}/processed_pkl/{s}.pkl, {P}/{T}/image_check/*.png (sketch path only)

7. HIGH-RES  code/train_highres.py  -t T  (and -i = the SAME dir as -o above)
   reads  {P}/{T}/{parameters.json}, {P}/{T}/processed_pkl_highres/*, {P}/{train_samples}
          (note: train_samples is resolved against P, not T)
   reads  {P}/{I}/UNI_multiscale_patches/emb_224/{s}.npz          (low-res, step 2)
          {P}/{I}/UNI_multiscale_patches/emb_{56,14}/{s}.npz     (high-res, step 5)
          {P}/{T}/train_pred_lowres/{s}-pred.pickle               (step 6)
          {P}/{I}/bulk/{s}-bulk.tsv                               (step 3)
          {P}/{I}/UNI_multiscale_patches/wsis_rescaled/*          (step 1)
          {P}/{D}/metadata/{s}.json
   writes {P}/{T}/processed_pkl_highres/{s}_superpatch_{id}.pkl
          {P}/{T}/processed_pkl_highres/superpatch_meta.csv
          {P}/{T}/processed_pkl_highres/superpatch_info.json
          {P}/{T}/image_check/{s}_superpatch_{id}_*.png           (--plot)
          {P}/{T}/train_log_highres/                              (--log)
          {P}/{T}/checkpoints_highres/AURORA_{epoch}.pth          (whole module, every 10 epochs)
          {P}/{T}/AURORA_highres_model_weights.pth                (state_dict, final)

8. DOWNSTREAM CONSUMERS of {P}/{T}/AURORA_highres_model_weights.pth
   code/test_highres.py            --model_pth AURORA_highres_model_weights.pth   (command.sh:15)
   code/predict_highres.py         --model_pth AURORA_highres_model_weights.pth   (command.sh:28)
   code/run_chunked_tcga.py        --model_pth .../AURORA_highres_model_weights.pth (command.sh:46)
      ... and re-reads {P}/{I}/UNI_multiscale_patches/emb_{56,14}/*.npz, deleting
          {s}.npz / {s}_raw.npz and processed_pkl_test_highres/{s}.pkl per chunk
          (run_chunked_tcga.py:309-334)
```

**Parallel `*_optimized` stack (not part of the six, but the only consumers of modules 4–6):**

```
code/extract_features_highres_optimized.py
   -> highres_inference_pipeline.py (ActivePatchBatcher, CpuAccumulatorPipeline, stage_valid_feature_batch)
   -> highres_gpu_strip_pipeline.py (RegularGridStripAccumulator, save_sparse_npz, stream_smooth_sparse_store)
        -> highres_grid_pipeline.py  (RegularGridStripAccumulator as _FineGridStripAccumulator)
   -> image_utils_optimized.py (HighResFeatureExtractor, _SplitPatchAccumulator, get_UNI_model, smooth_embedding_halves)
        -> highres_feature_ops.py (SplitGridFeatureAccumulator, smooth_embeddings, smooth_embedding_halves)
   Writes the same emb_{56,14}/{s}.npz contract (plus *_raw.npz only with its opt-in flag).
```

---

## 8. Surprises, inconsistencies and gotchas

1. **`train_highres.py` crashes without `--log`.** `writer` is only created inside `if args.log:` (136–137) but
   is used unconditionally at 339–373 and `writer.close()` at 382 → `NameError: name 'writer' is not defined`
   after epoch 0. `command.sh` always passes `--log`.
2. **Two different checkpoint formats.** `checkpoints_highres/AURORA_{epoch}.pth` = pickled **whole module**;
   `AURORA_highres_model_weights.pth` = **state_dict only**. Only the latter is consumed by the rest of the repo
   (`test_highres.py`, `predict_highres.py`, `run_chunked_tcga.py`). The retrain path (`torch.load` at line 141,
   no `weights_only`) will fail on PyTorch ≥ 2.6 defaults, and it passes no `map_location`.
3. **No metadata in any checkpoint**: no optimizer/scheduler state, no epoch counter, no `parameters.json`
   copy. `AURORA_highres_model_weights.pth` is only loadable if the consumer rebuilds `AURORA` with the exact
   same `img_emb_dim`, `output_size`, `bulk_size`, `celltype_num`, `latent_size`, `bidirectional` — a silent
   `load_state_dict` shape error is the only failure signal.
4. **`prepare_highres_training.py` is misnamed**: it runs the *low-res* model (default `--model_pth
   AURORA_model_weights.pth`, `img_emb_dim` 1536, three patch scales) and writes the low-res prior pickle.
   It is not a high-res trainer and it does not touch `AURORA_highres_model_weights.pth`.
5. **Different cache directories with near-identical names**: `processed_pkl` (low-res, used by
   `prepare_highres_training.py:35`) vs `processed_pkl_highres` (high-res, used by `train_highres.py:38`).
6. **`--train_samples` is resolved against the project root**, not `train_path` (`f"{main_dir}/{args.train_samples}"`,
   line 79, flagged `# !!! change here`); the `train_path`-relative variant is the commented line 78.
   `command.sh` therefore passes `--train_samples Training_samples.csv` while `prepare_highres_training.py`
   takes `--sample_sheet debug/Training_samples.csv`.
7. **`--interim_path` semantics differ between the two training-side scripts**: `train_highres.py` defaults to
   `'AURORA_interim'`, `prepare_highres_training.py` defaults to `None` and silently falls back to `--train_path`
   (25–26). Passing `-o X` to the extractor therefore requires `-i X` on both.
8. **Patch-size arity is inconsistent across the chain**: `extract_features_highres.py` default `[56, 14]` (any
   number of scales), `train_highres.py` requires exactly **2** (lines 207–208, 216, 254),
   `prepare_highres_training.py` hard-codes **3** (lines 122–124) and `train_utils.py:836–839` hard-codes scale
   index `2`. One `parameters.json` cannot satisfy all of them.
9. **Pickle-name mismatch risk in the bridge**: `prepare_highres_training.py:129` writes
   `{dataset_test[index]}-pred.pickle` using the raw sample-sheet string, whereas `train_utils.py:780` reads
   `{dataset.replace('.h5','')}-pred.pickle`. A sample sheet with `.h5` suffixes produces files the trainer will
   not find (the author left the alternative naming commented at `train_utils.py:781`).
10. **The CATCH branch of `extract_features_highres.py` is not runnable from the documented layout**:
    it shells out to `{args.code_path}/run_CATCH.py` (line 124), but `run_CATCH.py` lives only in `code_dev/`,
    not in `code/` (`CODE_PATH=./AURORA/code/` in `command.sh`). It also writes a different `.npz` contract
    (CATCH keys `emb`/`pos`/`flag`/`pad` plus `raw_pos`/`raw_pad` at 156–161, no 3072-d `center||subpixel`
    concatenation, no `smooth_embedding_halves`), and it uses `/bin/bash` and `mv`/`rm` (POSIX-only).
11. **`--compile-mode` fallback is ineffective for lazy failures.** `torch.compile` is lazy; the `try/except`
    at 204–214 only catches errors raised at graph construction, not the far more common first-forward
    compilation error.
12. **`extract_features_highres.py` imports `re` (4) and `shutil` (20) but never uses them**; the whole
    plotting/`KMeans` path is dead weight unless `--plot`.
13. **`*_raw.npz` is never read by any model path** — it is only deleted by
    `run_chunked_tcga.py:320` cleanup, and `extract_features_highres_optimized.py` makes it opt-in
    (`--save-raw-npz`, help text at line 254) "to reduce I/O".
14. **Overlapping patches are averaged twice with different weights.** With `--stride 112` and 224-px patches,
    each 14-px block receives contributions from up to 4 source patches; `_PatchAccumulator.merge/finalize`
    (`image_utils.py:326–384`) averages them, and then `smooth_embedding_halves` blurs again with a 16x16
    (scale 14) / 4x4 (scale 56) box kernel (`extract_features_highres.py:334–349`).
15. **`pad` is stored as a Python tuple inside `np.savez`** (`pad=(dataset.row_pad, dataset.col_pad)`, line 297),
    so consumers do `tmp['pad'][0]//2` (`train_utils.py:861`) — an object-dtype element, not a plain int array.
16. **`flag` is always all-`True` on the UNI path** (`image_utils.py:381`), so every `tmp['emb'][keep_flag]`
    gather downstream is a no-op copy; the real sparsity filter is the superpatch windowing in `load_data2`.
17. **Cell-type phase bookkeeping double-counts**: at lines 322–328 `running_major_prop_loss[patch_size]`
    accumulates both `major_prop_recorder` and `local_prop_recorder`, and `train_utils`-style
    `running_local_prop_loss` never exists; TensorBoard tags `training major prop loss (…)` and
    `training local prop loss (…)` are then written from the same dict (359–370), so the "local prop loss"
    curve is not a local loss.
18. **`focal_gamma` mutates during training** (`focal_gamma += 0.5` at 164 for `epoch > exp_epochs`,
    `epoch % 50 == 0`) and is not saved, so a resumed run restarts the schedule.
19. **Unused variable in `train_highres.py`**: `superpatch_exist` is computed at line 94 and never used.
20. **Modules 4–6 are orphans with respect to the five pipeline scripts**: `highres_feature_ops.py`,
    `highres_grid_pipeline.py` and `highres_inference_pipeline.py` are imported only by
    `image_utils_optimized.py`, `highres_gpu_strip_pipeline.py` and `extract_features_highres_optimized.py`.
    `extract_features_highres.py` uses the legacy `image_utils` implementations
    (`_PatchAccumulator`, `smooth_embedding_halves`), which have **different signatures** from the
    `highres_feature_ops` versions (the latter adds `channel_chunk` / `validate_unique` and drops the legacy
    duplicate-position `ValueError` unless `validate_unique=True`, where duplicates are silently overwritten by
    last-write-wins in `smooth_embeddings`, lines 567–605).
