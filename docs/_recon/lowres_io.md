# AURORA low-res pipeline — CLI, filesystem I/O, and directory recon

Reconnaissance report. **No existing file was modified.** Sources analysed in full:

| # | File | Lines |
|---|------|-------|
| 1 | `code/rescale.py` | 157 |
| 2 | `code/extract_features.py` | 151 |
| 3 | `code/segment_image.py` | 34 |
| 4 | `code/utils.py` | 173 |
| 5 | `code/plot.py` | 192 |
| 6 | `code/run_istar.py` | 88 |
| 7 | `code/run_HistoSweep.py` | 403 |

Facts about adjacent files (producers/consumers only, not full audits): `code/create_patches_fp.py`, `code/HEST_preprocess.py`, `code/prepare_st.py`, `code/image_utils.py`, `code/datasets.py`, `code/train_utils.py`, `code/run_istar_core.sh`, `code/istar/*`, `code/HistoSweep/*`, `examples/*.csv`, `command.sh`, `requirements.txt`.

Verification note: pickle protocol numbers were measured on the bundled CPython **3.12.14** (`pickle.DEFAULT_PROTOCOL == 4`, `HIGHEST_PROTOCOL == 5`).

---

## 0. Shared conventions

All scripts resolve paths by string concatenation (`f"{main_dir}/{...}"` or `os.path.join`), never via `pathlib`, except `run_HistoSweep.py`.

Canonical symbols:

```
main_dir = args.project_path                    # default './'  -> causes harmless '//' in every composed path
save_dir = f"{main_dir}/{args.output_path}/UNI_multiscale_patches"   # rescale.py L29, extract_features.py L37
interim_dir = os.path.join(main_dir, args.interim_path)              # plot.py L42
pred_dir    = os.path.join(main_dir, args.pred_path)                 # plot.py L43
```

The literal directory name **`UNI_multiscale_patches` is hardcoded** in `rescale.py`, `extract_features.py`, `plot.py`, `run_istar.py` (and elsewhere in the repo). Only `run_HistoSweep.py` exposes it (`--patches_dir_name`).

With all defaults and `-p ./`, paths materialise as e.g. `.//AURORA_interim/UNI_multiscale_patches/wsis_rescaled/` (double slash from the `'./'` default). This is harmless on both POSIX and Windows.

---

## 1. `code/rescale.py`

`argparse.ArgumentParser(description='Rescale H&E images to target pixel size and adjust ST spot locations accordingly.')`

### 1.1 argparse

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--project_path` | `-p` | `str` | `'./'` | `Main directory path` |
| `--pixel_size` | — | `float` | `0.5` | `Target pixel size in microns` |
| `--output_path` | `-o` | `str` | `'AURORA_interim'` | `Output directory` |
| `--metadata_path` | `-m` | `str` | `'Sample_metadata.csv'` | `Sample metadata CSV file name` |
| `--processed_data_path` | `-d` | `str` | `'processed_data'` | `Processed data directory path` |
| `--plot` | — | `action='store_true'` | `False` | `Plot intermediate results for checking` |
| `--inference` | — | `action='store_true'` | `False` | `Inference mode flag` |
| `--image_path` | — | `str` | `'./wsis/'` | `Data directory path` |
| `--seg_path` | — | `str` | `'./wsis_segmentation/'` | `Save directory path` |

`--image_path` / `--seg_path` are used **only** when `--inference` is set. `target_pixel_size = args.pixel_size`.

### 1.2 Inputs READ (literal expressions)

| # | Purpose | Expression | Mode |
|---|---------|-----------|------|
| 1 | Sample sheet | `f"{main_dir}/{args.metadata_path}"` → `pd.read_csv(..., index_col=0, header=0)` | train |
| 1' | Sample sheet | `f"{main_dir}/{args.metadata_path}"` → `pd.read_csv(..., header=0)` then `meta.set_index('Image_name', inplace=True)`; needs `Image_name`, `Sample_name` (+ `Pixel_size_um`, `Image_folder` for non-`.svs`) | inference |
| 2 | Raw pixel size + spot diameter | `os.path.join(main_dir, args.processed_data_path, 'metadata', f"{image_name}.json")` → keys `pixel_size_um_estimated`, `spot_diameter` | train |
| 3 | Full-res H&E | `os.path.join(main_dir, args.processed_data_path, 'wsis', f"{image_name}.tif")` → `tifffile.TiffFile(...).pages[0].asarray()` | train |
| 4 | Tissue contours | `os.path.join(main_dir, args.processed_data_path, 'tissue_seg', f'{image_name}_contours.geojson')` → `os.path.exists` gate + `gpd.read_file` | train |
| 5 | ST data | `os.path.join(main_dir, args.processed_data_path, 'st', f"{image_name}.h5ad")` → `sc.read_h5ad`; uses `obsm['spatial']`, `obs_names` | train |
| 6 | WSI (per-folder) | `os.path.join(main_dir, f'{args.image_path}/{meta.loc[image_file,"Image_folder"]}/', image_file)` | inference |
| 6' | WSI (flat) | `os.path.join(main_dir, f'{args.image_path}/', image_file)` | inference |
| 7 | Slide MPP | `image_os.properties['openslide.mpp-x']` (only if `'.svs' in image_file`), else `meta.loc[image_file,'Pixel_size_um']` | inference |
| 8 | Segmentation mask | `mask_file = '.'.join(image_file.split('.')[:-1]) + '.geojson'`; `os.path.join(main_dir, f'{args.seg_path}/masks', mask_file)` → `gpd.read_file` | inference |
| 9 | Skip sentinel | `os.path.exists(f"{save_dir}/wsis_rescaled/{image_name}_mask_rescaled.tif")` | both |

Image name derivation: `image_name = image_file.split('.')[0]` (train) / `meta.loc[image_file,'Sample_name']` (inference).

### 1.3 Outputs WRITTEN / created

| # | Expression | Condition |
|---|-----------|-----------|
| 1 | `os.makedirs(f"{save_dir}", exist_ok=True)` | always |
| 2 | `os.makedirs(f"{save_dir}/wsis_rescaled", exist_ok=True)` | always |
| 3 | `os.makedirs(f"{save_dir}/check_figure", exist_ok=True)` | `--plot` |
| 4 | `f"{save_dir}/check_figure/{image_name}_raw_check.jpg"` (`cv2.imwrite`, RGB→BGR) | `--plot` and train |
| 5 | `f"{save_dir}/check_figure/{image_name}_rescaled_check.jpg"` | `--plot` and train |
| 6 | `f"{save_dir}/wsis_rescaled/{image_name}_rescaled.tif"` (`tifffile.imwrite`, `compression='zlib'`, `bigtiff=False` unless `height>40000 and width>60000`) | always |
| 7 | `f"{save_dir}/wsis_rescaled/{image_name}_mask_rescaled.tif"` (same compression/bigtiff) | always |
| 8 | `f"{save_dir}/wsis_rescaled/{image_name}_loc_rescaled.npy"` (`np.save`) | train |
| 9 | `f"{save_dir}/wsis_rescaled/{image_name}_barcode.pickle"` (`utils.save_pickle(bar, ...)`) | train |
| 10 | `f"{save_dir}/wsis_rescaled/{image_name}_pixel_size.npy"` = `np.array([spot_radius_pixel])` | train |
| 11 | `f"{save_dir}/rescale_parameters_{timestamp_string}.csv"`, `timestamp_string = datetime.now().strftime("%Y%m%d-%H%M%S")`; `DataFrame(index=<sample>, columns=['row_min','row_max','col_min','col_max','scale_factor'])` | always (after the loop) |

Note on #10: the file is named `pixel_size` but stores the **already-rescaled spot radius** (`spot_radius_pixel *= pixel_size_raw/target_pixel_size`), i.e. spot diameter/2 in target-pixel units. `extract_features.py` reads it as `spot_radius_pixel`.

Note on #11: filename is timestamped → a new file per run; no fixed name, so consumers must glob `rescale_parameters_*.csv`.

### 1.4 Third-party imports

`pandas`, `numpy`, `tqdm`, `tifffile`, `geopandas`, `scanpy`, `rasterio` (`from rasterio.features import rasterize`), `cv2` (opencv-python), `openslide` (**lazy, only if `--inference`**).

Local modules: `utils`, `image_utils` (`cut_and_rescale_image`, `plot_image_with_spot`).
Stdlib: `os`, `json`, `argparse`, `datetime` (imported mid-file at L151).

---

## 2. `code/extract_features.py`

`argparse.ArgumentParser(description='Rescale H&E images to target pixel size and adjust ST spot locations accordingly.')` — copy-pasted from `rescale.py`; the description is wrong for this script.

### 2.1 argparse

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--project_path` | `-p` | `str` | `'./'` | `Main directory path` |
| `--batch_size` | `-b` | `int` | `128` | `Batch size for processing` |
| `--output_path` | `-o` | `str` | `'AURORA_interim'` | `Output directory` |
| `--patch_sizes` | — | `int`, `nargs='+'` | `[3584, 896, 224]` | `List of patch sizes` |
| `--metadata_path` | `-m` | `str` | `'Sample_metadata.csv'` | `Sample metadata CSV file name` |
| `--processed_data_path` | `-d` | `str` | `'processed_data'` | `Processed data directory path` |
| `--plot` | — | `action='store_true'` | `False` | `Plot intermediate results for checking` |
| `--token` | `-t` | `str` | `None` | `Hugging Face token` |
| `--num_workers` | — | `int` | `4` | `Number of workers for data loading` |
| `--inference` | — | `action='store_true'` | `False` | `Inference mode flag` |
| `--image_path` | — | `str` | `'./wsis/'` | `Data directory path` |
| `--seg_path` | — | `str` | `'./wsis_segmentation/'` | `Save directory path` |
| `--UNI_local_path` | — | `str` | `None` | `Path to the local UNI model folder` |

`--image_path` and `--seg_path` are **accepted but never referenced** anywhere in the file (dead arguments; passing them is harmless).

Model loading gate (L44–49): if `--token` is `None` and `--UNI_local_path` is `None` → `raise ValueError("Either --token or --UNI_local_path must be provided.")`. `--token` path calls `get_UNI_model(args.token, "UNI2-h")` (HF hub download, needs network); `--UNI_local_path` calls `get_UNI_model(args.UNI_local_path, "UNI2-h", from_local=True)` → `timm.create_model(f'local-dir:{path}', pretrained=True, ...)`, i.e. the directory is read as a **timm local model dir** (expects `model.safetensors`/`pytorch_model.bin` + config).

### 2.2 Inputs READ (literal expressions)

| # | Purpose | Expression |
|---|---------|-----------|
| 1 | Sample sheet | `f"{main_dir}/{args.metadata_path}"` (index_col=0 train; header=0 + `set_index('Image_name')` inference) |
| 2 | Skip sentinel | `os.path.exists(f"{save_dir}/emb_{min(patch_sizes)}/{image_name}.npz")` (⇒ `emb_224/`) |
| 3 | Rescaled H&E | `f"{save_dir}/wsis_rescaled/{image_name}_rescaled.tif"` |
| 4 | Rescaled mask | `f"{save_dir}/wsis_rescaled/{image_name}_mask_rescaled.tif"` |
| 5 | Rescaled spot coords | `f"{save_dir}/wsis_rescaled/{image_name}_loc_rescaled.npy"` (train) |
| 6 | Rescaled spot radius | `f"{save_dir}/wsis_rescaled/{image_name}_pixel_size.npy"` → `np.load(...)[0]` (train) |
| 7 | HEST patch stack | `os.path.join(main_dir, args.processed_data_path, 'patches', f"{image_name}.h5")` via `get_HEST1k_patch` (reads `f['barcode']`, `f['img']`, `f['coords']`, and `f['img'].attrs` keys `downsample`, `patch_size_src`, `patch_size_target`, `pixel_size`) — train only |
| 8 | UNI2-h weights | `args.UNI_local_path` (timm `local-dir:`) or HF hub `MahmoodLab/UNI2-h` |

`image_name = image_file.split('.')[0]` (train) / `meta.loc[image_file,'Sample_name']` (inference).

### 2.3 Outputs WRITTEN / created

| # | Expression | Condition |
|---|-----------|-----------|
| 1 | `os.makedirs(f"{save_dir}")` = `{main_dir}/{output_path}/UNI_multiscale_patches` | always |
| 2 | `os.makedirs(f"{save_dir}/check_figure")` | **always** (not gated on `--plot`) |
| 3 | `os.makedirs(f"{save_dir}/wsis_rescaled")` | always (idempotent; normally created by `rescale.py`) |
| 4 | `os.makedirs(f"{save_dir}/locs")` | **train only** |
| 5 | `os.makedirs(f"{save_dir}/emb_{patch_size}")` | once per patch size |
| 6 | `f"{save_dir}/check_figure/{image_name}_mask_check.jpg"` (`cv2.imwrite(dataset.plot_mask())`) | `--plot` |
| 7 | `f"{save_dir}/check_figure/{image_name}_{patch_size}patch_check.jpg"` | `--plot` |
| 8 | `f"{save_dir}/check_figure/{image_name}_emb_{patch_size}_kmeans.png"` (`plt.savefig`, KMeans n_clusters=5) | `--plot` **and** `patch_size == 224` (literal) |
| 9 | `f"{save_dir}/emb_{patch_size}/{image_name}.npz"` via `np.savez(emb=…, pos=…, flag=…, pad=(dataset.row_pad, dataset.col_pad))` | always |
| 10 | `f"{save_dir}/locs/{image_name}-bars.pickle"` (`utils.save_pickle(barcode, …)`) | train |
| 11 | `f"{save_dir}/locs/{image_name}-locs.pickle"` (`utils.save_pickle(coords, …)`) | train |
| 12 | `f"{save_dir}/locs/{image_name}-meta.pickle"` (`utils.save_pickle(img_meta, …)`) | train |

Patch geometry: `patch_size == min(patch_sizes)` → `stride=patch_size`; all other sizes → `stride=patch_size//2` (a `max(patch_sizes)`/`stride=patch_size//4` branch is commented out at L88–89).
Video-memory hygiene: `gc.collect()` + `torch.cuda.empty_cache()` per patch size (L144–145).

### 2.4 Third-party imports

`torch`, `pandas`, `numpy`, `tqdm`, `tifffile`, `matplotlib` (`matplotlib.pyplot`), `sklearn` (`sklearn.cluster.KMeans`), `cv2`.

Local modules: `utils`, `image_utils` (`get_UNI_model`, `get_HEST1k_patch`), `datasets` (`PatchDataset`).
Stdlib: `os`, `gc`, `argparse`.

---

## 3. `code/segment_image.py`

`argparse.ArgumentParser()` — **no description**.

### 3.1 argparse

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--project_path` | `-p` | `str` | `'./'` | `Main directory path` |
| `--data_path` | `-d` | `str` | `'./wsis/'` | `Data directory path` |
| `--save_path` | `-s` | `str` | `'./wsis_segmentation/'` | `Save directory path` |
| `--sample_sheet` | — | `str` | `None` | `Dataset directory name` |
| `--code_path` | — | `str` | `'./code/'` | `Code directory path` |
| `--preset` | — | `str` | `'tcga.csv'` | `Preset file for segmentation` |

### 3.2 Inputs READ

| # | Purpose | Expression |
|---|---------|-----------|
| 1 | Sample sheet | `f"{args.project_path}/{args.sample_sheet}"` → `pd.read_csv(...)` if it endswith `.csv`, `sep="\t"` if `.tsv`. Any other extension ⇒ `wsis_list` never assigned ⇒ `NameError` |
| 2 | Folder column | `wsis_list.loc[idx, "Image_folder"]` (the `"Sample ID"` and `"File Name"` lookups are commented out at L28/L30) |
| 3 | Preset | `{args.project_path}/{args.preset}` → forwarded to `create_patches_fp.py --preset` |
| 4 | Segmentation code | `f"{args.code_path}/create_patches_fp.py"` (executed via `os.system("python …")`) |
| 5 | WSI source dir | `{args.project_path}/{args.data_path}` (flat mode) or `{args.project_path}/{args.data_path}/{file_id}` (per-sample mode) — enumerated inside `create_patches_fp.py` with `os.listdir(source)` |

### 3.3 Outputs WRITTEN / created

| # | Expression | Who |
|---|-----------|-----|
| 1 | `os.makedirs(f"{args.project_path}/{args.save_path}", exist_ok=True)` | `segment_image.py` L12 |
| 2 | `{args.project_path}/{args.save_path}/patches/`, `/masks/`, `/stitches/` | `create_patches_fp.py` L254–256, L278 (`patch_save_dir`, `mask_save_dir`, `stitch_save_dir`) |
| 3 | `{args.project_path}/{args.save_path}/process_list_autogen.csv` | `create_patches_fp.py` L91, L224 |
| 4 | `{args.project_path}/{args.save_path}/masks/{slide_id}.geojson` (`WSI_object.saveSegmentation`) | `create_patches_fp.py` L190 (because `--seg` is passed) |
| 5 | `{args.project_path}/{args.save_path}/masks/{slide_id}.png` (`WSI_object.visWSI(..., custom_downsample=10)`) | `create_patches_fp.py` L192–195 — `save_mask=True` is **hardcoded** in the call, not a CLI flag |
| 6 | `patches/{slide_id}.h5`, `stitches/{slide_id}.jpg` | **not produced**, because `segment_image.py` never passes `--patch`/`--stitch` (both default `False`); the two directories are still created empty |

### 3.4 Third-party imports

`pandas`. Stdlib: `os` (incl. `os.system`). No local modules.

---

## 4. `code/utils.py`

No argparse, no `__main__`. Module side effect at import time: `Image.MAX_IMAGE_PIXELS = None` (**disables Pillow's decompression-bomb guard process-wide for every importer**: `rescale.py`, `extract_features.py`, `plot.py`, `run_istar.py`, `prepare_st.py`, …).

### 4.1 Every public function

| # | Signature | Behaviour |
|---|-----------|-----------|
| 1 | `mkdir(path)` | `os.makedirs(os.path.dirname(path), exist_ok=True)` — creates the **parent directory of `path`**, not `path`; no-op when `dirname == ''` |
| 2 | `load_image(filename, verbose=True)` | `np.array(Image.open(filename))`; drops alpha when `ndim==3 and shape[-1]==4`; prints `Image loaded from {filename}` |
| 3 | `load_mask(filename, verbose=True)` | `load_image(...) > 0`; if 3-D then `.any(2)` |
| 4 | `save_image(img, filename)` | `mkdir(filename)`; `Image.fromarray(img).save(filename)`; prints `filename` |
| 5 | `read_lines(filename)` | list of `line.rstrip()` |
| 6 | `read_string(filename)` | `read_lines(filename)[0]` (raises `IndexError` on empty file) |
| 7 | `write_lines(strings, filename)` | `mkdir(filename)`; writes `f'{s}\n'` per element; prints `filename` |
| 8 | `write_string(string, filename)` | `write_lines([string], filename)` |
| 9 | `save_pickle(x, filename)` | see §4.2 |
| 10 | `load_pickle(filename, verbose=True)` | see §4.2 |
| 11 | `load_tsv(filename, index=True)` | `pd.read_csv(filename, sep='\t', header=0, index_col=0 if index else None)`; prints source path |
| 12 | `save_tsv(x, filename, **kwargs)` | `mkdir(filename)`; injects `kwargs['sep']='\t'` **only if absent**; `x.to_csv(filename, **kwargs)`; prints `filename` |
| 13 | `load_yaml(filename, verbose=False)` | `yaml.safe_load(open(filename,'r'))` |
| 14 | `save_yaml(filename, content)` | `yaml.dump(content, open(filename,'w'))`; **does not call `mkdir`** and prints the *file object* (`print(file)`), not the path |
| 15 | `join(x)` | `list(itertools.chain.from_iterable(x))` — one-level flatten |
| 16 | `get_most_frequent(x)` | `uniqs[counts.argmax()]` from `np.unique(x, return_counts=True)` |
| 17 | `sort_labels(labels, descending=True)` | returns `(labels, labels_uniq)` where labels are re-ranked by count (negatives untouched) |
| 18 | `normalize_column(col)` | min-max; returns `col` unchanged when `max_val <= min_val` |
| 19 | `save_pkl(filename, save_object)` | CLAM-borrowed; `pickle.dump` to `open(filename,'wb')`; **no `mkdir`**, no compression |
| 20 | `load_pkl(filename)` | CLAM-borrowed; `pickle.load(open(filename,'rb'))` |
| 21 | `save_hdf5(output_path, asset_dict, attr_dict=None, mode='a', chunk_size=32)` | h5py; creates datasets with `chunks=(chunk_size,)+shape[1:]`, `maxshape=(None,)+shape[1:]`, `dtype=val.dtype`; appends by `resize`+assign when the key exists; writes per-key attrs from `attr_dict[key]`; returns `output_path` |

### 4.2 `save_pickle` / `load_pickle` — exact behaviour

```python
def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    # print(filensame)          # L71 — commented-out typo, no effect

def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x
```

* **Arguments**: `save_pickle(x, filename)` — positional only, no kwargs, no protocol/compression parameters, no return value. `load_pickle(filename, verbose=True)` — one positional path plus the `verbose` print flag; returns the unpickled object.
* **Compression**: **none**. Plain `pickle` stream, no `gzip`/`bz2`/`lzma`, no `zipfile`, no `zlib`. (Compression in this pipeline happens only on TIFFs: `tifffile.imwrite(..., compression='zlib')`.)
* **Protocol**: `pickle.dump(x, file)` ⇒ `protocol=None` ⇒ `pickle.DEFAULT_PROTOCOL`. Measured on the bundled CPython 3.12.14: `DEFAULT_PROTOCOL == 4` (`HIGHEST_PROTOCOL == 5`). So files are protocol-4 and readable by any Python ≥ 3.4; they are **not** forward-pinned and the protocol is not recorded anywhere in the pipeline.
* **Extension**: **none enforced and none appended**. `filename` is used verbatim; the extension is purely a caller convention — `.pickle` in the low-res pipeline (`<name>_barcode.pickle`, `locs/<name>-bars.pickle`, `pred/<sample>-pred.pickle`, `istar_pred/<sample>/cnts-super/<gene>.pickle`), and `.pickle` also in `istar/` (`cnts-super/<name>.pickle`). `load_pickle` never inspects the suffix; a `.pickle` file is bytes-compatible with `pickle.load` regardless of name.
* **Directory creation**: `save_pickle` creates only the *parent directory* of `filename`, via `mkdir`; it cannot create a missing intermediate chain beyond one level (it is `makedirs`, so actually the whole parent chain is created). `load_pickle` creates nothing.
* There is deliberate duplication with the CLAM-derived `save_pkl`/`load_pkl` (§4.1 #19–20): the only differences are the missing `mkdir` and the absent verbose print.

### 4.3 Third-party imports

`PIL` (Pillow), `h5py`, `numpy`, `pandas`, `yaml` (PyYAML). Stdlib: `itertools`, `pickle`, `os`. No local imports.

---

## 5. `code/plot.py`

`argparse.ArgumentParser(description='AURORA Prediction Script')`

### 5.1 argparse

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--project_path` | `-p` | `str` | `'./'` | `Main directory path` |
| `--interim_path` | `-i` | `str` | `'AURORA_interim'` | `Interim data directory` |
| `--pred_path` | `-t` | `str` | `'AURORA_pred'` | `Directory to store predicted data` |
| `--resolution` | `-r` | `int` | `224` | `For iStar-enhanced prediction, use -1` |
| `--samples` | `-s` | `str`, `nargs='+'` | `None` | `List of sample names to plot` |
| `--genes` | `-g` | `str`, `nargs='+'` | `None` | `List of gene names to plot or a CSV file containing gene names to plot in the first column` |
| `--celltypes` | `-c` | `str`, `nargs='+'` | `None` | `List of cell types to plot or a CSV file containing cell types to plot in the first column` |
| `--plot_dir` | `-d` | `str` | `'AURORA_plots'` | `Directory to save plots` |
| `--HistoSweep_mask` | — | `action='store_true'` | `False` | `Use HistoSweep mask to the plots` |

Commented-out (present as dead code, L28–32): `--parameter_json/-j`, `--model_pth`, `--sample_sheet`, `--bulk`, `--batch_size`.

There is **no `--inference` flag**; mode is selected implicitly by `--resolution` (`> 0` ⇒ low-res prediction pickle; `-1`/other ≤ 0 ⇒ iStar branch).

### 5.2 Inputs READ (literal expressions)

| # | Purpose | Expression |
|---|---------|-----------|
| 1 | Rescaled H&E | `image_path = f"{main_dir}/{dataset}_rescaled.tif"` with `main_dir = f"{interim_dir}/UNI_multiscale_patches/wsis_rescaled/"` (i.e. `<p>/<i>/UNI_multiscale_patches/wsis_rescaled/<sample>_rescaled.tif`) |
| 2 | Rescaled mask | `mask_path = f"{main_dir}/{dataset}_mask_rescaled.tif"` (same dir prefix) |
| 3 | Gene list CSV | `pd.read_csv(args.genes)` — path taken verbatim from the CLI, resolved against **CWD**, not `project_path` |
| 4 | Cell-type list CSV | `pd.read_csv(args.celltypes)` — same caveat |
| 5 | Low-res predictions | `os.path.join(pred_dir, 'pred', f"{sample}-pred.pickle")` → `utils.load_pickle`; keys used: `patches_loc_before_pad`, `resolutions`, `exp_pred`, `prop_pred`, `genes`, `cell_types` |
| 6 | HistoSweep mask (optional) | `mask_path = os.path.join(main_dir, args.pred_path, 'istar_pred', sample, 'HistoSweep_Output', 'mask.png')` |
| 7 | iStar mask | `f"{pred_dir}/istar_pred/{sample}/mask-small.png"` (`cv2.imread(..., IMREAD_GRAYSCALE)`) |
| 8 | iStar gene maps | `f"{pred_dir}/istar_pred/{sample}/cnts-super/{gene}.pickle"` (raw `pickle.load`) |
| 9 | iStar cell-type maps | `f"{pred_dir}/istar_pred/{sample}/cnts-super/{cell_type}.pickle"` where `cell_type = cell_type.replace(" ","_").replace("-","_")` |
| 10 | Cell-type names for the iStar branch | `results['cell_types']` from `os.path.join(pred_dir, 'pred', f"{sample}-pred.pickle")` |

### 5.3 Outputs WRITTEN / created

| # | Expression | Condition |
|---|-----------|-----------|
| 1 | `os.makedirs(plot_dir, exist_ok=True)` (`plot_dir = os.path.join(main_dir, args.plot_dir)`) | always |
| 2 | `os.makedirs(os.path.join(plot_dir, sample), exist_ok=True)` | per sample |
| 3 | `f"{plot_dir}/{sample}/{sample}-{gene}-{args.resolution}pixels.png"` (`cv2.imwrite`) | `--resolution > 0` |
| 4 | `f"{plot_dir}/{sample}/{sample}-{celltype}_proportion-{args.resolution}pixels.png"` | `--resolution > 0` |
| 5 | `f"{plot_dir}/{sample}/{sample}-{gene}-iStar-enhanced.png"` | `--resolution <= 0` |
| 6 | `f"{plot_dir}/{sample}/{sample}-{celltype}_proportion-iStar-enhanced.png"` | `--resolution <= 0` |

### 5.4 Third-party imports

`numpy`, `pandas`, `matplotlib` (`matplotlib.pyplot`, `plt.get_cmap('turbo')`), `cv2`; `tifffile` (**lazy**, inside `load_image`).

Local modules: `utils` (`utils.load_pickle`).
Stdlib: `os`, `argparse`, `pickle` (lazy at L133 for the iStar branch, plus a second unused `import pickle` at L162).

---

## 6. `code/run_istar.py`

Sets `os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))` at import (L2) — process-wide cv2 limit raise.
`argparse.ArgumentParser(description='Enhance prediction by iStar.')`

### 6.1 argparse

| Flag | Short | Type | Default | Help |
|------|-------|------|---------|------|
| `--project_path` | `-p` | `str` | `'./'` | `Main directory path` |
| `--interim_path` | `-i` | `str` | `'AURORA_interim'` | `Interim data directory` |
| `--pred_path` | `-t` | `str` | `'AURORA_pred'` | `Directory to store predicted data` |
| `--sample_sheet` | — | `str` | `None` | `Dataset directory name` (de-facto **required**: `None` ⇒ `pd.read_csv("./None")`) |
| `--gene_list_file` | — | `str` | `'AURORA_interim/Gene_to_predict.csv'` | `Gene list file path` |
| `--celltype_file` | — | `str` | `'AURORA_interim/deconvolution/cell_types.csv'` | `Cell type list file path` |
| `--patch_size` | — | `int` | `224` | `Patch size for prediction` |
| `--epoch_istar` | — | `int` | `50` | `Epoch number for iStar` |

Commented out: `--dataset` (L35).

### 6.2 Inputs READ (literal expressions)

| # | Purpose | Expression |
|---|---------|-----------|
| 1 | Sample sheet | `f"{main_dir}/{args.sample_sheet}"` (`header=0`), then `meta['Sample_name'].values` |
| 2 | Gene list | `pd.read_csv(f"{main_dir}/{args.gene_list_file}")['Genes'].values` |
| 3 | Cell types | `pd.read_csv(f"{main_dir}/{args.celltype_file}", header=None).values.squeeze()` (no header; `" "` and `"-"` → `"_"`) |
| 4 | Rescaled H&E | `f"{main_dir}/{interim_dir}/UNI_multiscale_patches/wsis_rescaled/{dataset}_rescaled.tif"` (via module-level `load_image`) |
| 5 | Rescaled mask | `f"{main_dir}/{interim_dir}/UNI_multiscale_patches/wsis_rescaled/{dataset}_mask_rescaled.tif"` |
| 6 | Low-res predictions | `f"{main_dir}/{args.pred_path}/pred/{dataset}-pred.pickle"` → `utils.load_pickle`; keys `exp_pred`, `prop_pred`, `patches_loc_before_pad` |
| 7 | iStar core script | `bash $CODE_PATH/run_istar_core.sh` — read from the **`CODE_PATH` environment variable**, not from `--project_path` (repo copy at `code/run_istar_core.sh`) |
| 8 | (inside the shell script) | `$ISTAR_PATH/extract_features.py`, `$ISTAR_PATH/impute.py`, `$ISTAR_PATH/cluster.py` — `ISTAR_PATH` env var |

### 6.3 Outputs WRITTEN / created

| # | Expression | Who |
|---|-----------|-----|
| 1 | `os.makedirs(f"{main_dir}/{args.pred_path}/istar_pred", exist_ok=True)` | L47 |
| 2 | `os.makedirs(f"{main_dir}/{args.pred_path}/istar_pred/{dataset}", exist_ok=True)` | L54 |
| 3 | `{…}/istar_pred/{dataset}/he.tif` (`tifffile.imwrite`, `compression='zlib'`, `adjust_margins(pad=256, pad_value=255)`) | L56 |
| 4 | `{…}/istar_pred/{dataset}/mask.tif` (`compression='zlib'`, `pad_value=mask.min()`) | L58 |
| 5 | `{…}/istar_pred/{dataset}/mask-small.png` (`istar.utils.save_image(reduce_mask(mask, factor=16))`) | L60 |
| 6 | `{…}/istar_pred/{dataset}/cnts.tsv` (`sep="\t"`, `index=True`) | L75 |
| 7 | `{…}/istar_pred/{dataset}/gene-names.txt` (one column name per line) | L78–80 |
| 8 | `{…}/istar_pred/{dataset}/locs.tsv` (`sep="\t"`, `index=True`, columns `x`,`y`, `patch_loc_before_pad + 112`) | L87 |
| 9 | `run_istar_core.sh <prefix> <epochs>` writes into the same prefix: `pixel-size.txt`, `pixel-size-raw.txt`, `radius.txt`, `embeddings-gene.pickle`, `cnts-super/<name>.pickle`, `clusters-gene/`, `states/`, `training-data-plots/` | `code/run_istar_core.sh` + `code/istar/impute.py` |

### 6.4 Third-party imports

`tifffile`, `numpy`, `pandas`.

Local packages/modules: `utils`; `istar.preprocess` (`adjust_margins`, `reduce_mask`); `istar.utils` (`save_image`).
Stdlib: `os`, `argparse`.

---

## 7. `code/run_HistoSweep.py`

No module-level argparse; parser built in `build_parser()`; `main(argv=None) -> int`; `raise SystemExit(main())`.

Module constant: `DEFAULT_HISTOSWEEP_DIR = Path(__file__).resolve().parent / "HistoSweep"` → `<repo>/code/HistoSweep`.

`argparse.ArgumentParser(description="Run HistoSweep on AURORA rescaled WSIs and write per-sample AURORA-compatible tissue masks.")`

### 7.1 argparse

| Flag | Short | Type / action | Default | Help |
|------|-------|---------------|---------|------|
| `--project_path` | `-p` | `str` | `"./"` | `Main directory path (same as predict_highres).` |
| `--interim_path` | `-i` | `str` | `"AURORA_interim"` | `Interim data directory (same as predict_highres).` |
| `--patches_dir_name` | — | `str` | `"UNI_multiscale_patches"` | `Name of the subdirectory under --interim_path that holds the rescaled WSIs and HistoSweep masks (default: UNI_multiscale_patches).` |
| `--pred_path` | `-t` | `str` | `"AURORA_pred"` | `Directory for HistoSweep native outputs (same as predict_highres).` |
| `--sample_sheet` | — | `str` | `None` | `CSV with a 'Sample_name' column (same one passed to predict_highres).` |
| `--samples` | — | `str`, `nargs="+"` | `None` | `Optional explicit sample names; defaults to all rows in --sample_sheet.` |
| `--histosweep_path` | — | `str` | `str(DEFAULT_HISTOSWEEP_DIR)` = `code/HistoSweep` | `Path to the HistoSweep package directory.` |
| `--pixel_size` | — | `float` | `0.5` | `Target pixel size in microns (record only; the input is already rescaled).` |
| `--density_thresh` | — | `float` | `100` | `Density threshold; increase to remove more artifacts (e.g. fiducials).` |
| `--patch_size` | — | `int` | `16` | `Size of one square super-pixel used throughout processing.` |
| `--min_size` | — | `int` | `10` | `Minimum super-pixel cluster kept by background cleaning.` |
| `--clean_background` | — | `action="store_false"`, `dest="clean_background"` | `True` | `Disable cleaning of isolated debris and small specs outside tissue.` |
| `--force` | — | `action="store_true"` | `False` | `Recompute even if <sample>_HistoSweep_mask_rescaled.tif already exists.` |
| `--mask-suffix` | — | `str` (note the **hyphen**; dest `mask_suffix`) | `"HistoSweep"` | `Suffix of the generated mask file name (before '_mask_rescaled.tif').` |
| `--keep-native-outputs` | — | `action="store_true"` | `False` | `Keep HistoSweep's native outputs (mask.png, mask-small.png, plots) under <pred_path>/HistoSweep/. By default they are removed after all samples finish.` |

`parser.error(...)` validations: `--sample_sheet` or `--samples` required; `--patch_size > 0`; `--density_thresh > 0`; `--min_size > 0`.

### 7.2 Inputs READ (literal expressions)

| # | Purpose | Expression |
|---|---------|-----------|
| 1 | Sample list | `os.path.join(args.project_path, args.sample_sheet)` → `pd.read_csv(..., header=0)`; requires column `Sample_name` (else `ValueError`) |
| 2 | Rescaled H&E | `wsis_dir = Path(args.project_path)/args.interim_path/args.patches_dir_name/"wsis_rescaled"`; per sample `image_path = wsis_dir / f"{sample}_rescaled.tif"` |
| 3 | Skip sentinel | `mask_path = wsis_dir / f"{sample}_{args.mask_suffix}_mask_rescaled.tif"` (`.exists()` and `not args.force`) |
| 4 | HistoSweep grid mask | `Image.open(out_dir / "mask-small.png")` where `out_dir = native_out_root / sample`, `native_out_root = Path(args.project_path)/args.pred_path/"HistoSweep"` |
| 5 | HistoSweep package | `Path(args.histosweep_path).resolve()` must be a directory (else `FileNotFoundError`); inserted at `sys.path[0]`; then `from computeMetrics import compute_metrics_fast`, `from densityFiltering import compute_low_density_mask`, `from textureAnalysis import run_texture_analysis_optimized`, `from ratioFiltering import run_ratio_filtering`, `from generateMask import generate_final_mask`, `from saveParameters import saveParams` |
| 6 | Round-trip verification | `tifffile.imread(str(mask_path))` re-read after writing |

### 7.3 Outputs WRITTEN / created

| # | Expression | Condition |
|---|-----------|-----------|
| 1 | `out_dir.mkdir(parents=True, exist_ok=True)` = `{project_path}/{pred_path}/HistoSweep/{sample}` | per processed sample |
| 2 | `out_dir / "mask.png"`, `out_dir / "mask-small.png"` by `generate_final_mask` (`HistoSweep/generateMask.py` L30, L36) | per processed sample |
| 3 | `out_dir / "HistoSweep_parameters.txt"` by `saveParams(sample, str(out_dir), False, True, pixel_size, density_thresh, clean_background, min_size, patch_size, pixel_size)` (`HistoSweep/saveParameters.py` L10) | per processed sample |
| 4 | `out_dir / "AdditionalPlots" / "textureAnalysis_plots" / …` by `run_texture_analysis_optimized(prefix=sample, image=..., tissue_mask=..., output_dir=str(out_dir), patch_size=..., glcm_levels=64)` (`HistoSweep/textureAnalysis.py` L125–126) | per processed sample |
| 5 | `wsis_dir / f"{sample}_{args.mask_suffix}_mask_rescaled.tif"` → `tifffile.imwrite(str(mask_path), aurora_mask, compression="zlib", bigtiff=<orig_h>40000 or <orig_w>60000)`; dtype `uint8`, values `0/1` | the deliverable |
| 6 | **Deletion**: `shutil.rmtree(native_out_root, ignore_errors=True)` removes `{project_path}/{pred_path}/HistoSweep/` entirely | unless `--keep-native-outputs` |

Mask construction: `mask_small > 0` → `np.kron(tissue_grid, np.ones((patch_size, patch_size)))` → crop `[:orig_h, :orig_w]`; `orig_h, orig_w` are the shape of the *unpadded* rescaled image; the image itself is right/bottom padded to a multiple of `patch_size` in memory only (`_pad_right_bottom`, fill 255) and is never written.

### 7.4 Third-party imports

`numpy` (top level); `pandas` (**lazy**, inside `main`); `tifffile` and `PIL` (`from PIL import Image`) (**lazy**, inside `run_histosweep_for_sample`); optional `pyvips` is probed and, if missing, replaced by a stub module (`_ensure_pyvips_stub`) so that HistoSweep's `utils.py` imports cleanly.

Local: `HistoSweep` package modules (resolved via `--histosweep_path`, imported lazily).
Stdlib: `argparse`, `os`, `shutil`, `sys`, `types`, `pathlib`.

---

## 8. Directory inventory

`<P>` = `args.project_path` (default `./`); `<O>` = `--output_path` (default `AURORA_interim`).
"External" = produced outside the seven analysed scripts; noted because the layout depends on it.

### 8.1 Everything the low-res pipeline produces

| Path segment (full form) | Created by | Consumed by |
|---|---|---|
| `<P>/wsis_segmentation/` | `segment_image.py` L12 (`os.makedirs(f"{args.project_path}/{args.save_path}")`); also `create_patches_fp.py` L278 | — |
| `<P>/wsis_segmentation/masks/` | `create_patches_fp.py` L255 + L278 | `rescale.py` inference (L86, `.geojson` only) |
| `<P>/wsis_segmentation/masks/{slide_id}.geojson` | `create_patches_fp.py` L190 | `rescale.py` inference → `gpd.read_file` → rasterize → `_mask_rescaled.tif` |
| `<P>/wsis_segmentation/masks/{slide_id}.png` | `create_patches_fp.py` L192–195 (`save_mask=True` hardcoded) | **nothing** in the pipeline (QC artifact) |
| `<P>/wsis_segmentation/patches/` | `create_patches_fp.py` L254 + L278 | created empty by `segment_image.py` (no `--patch`); **not** the `patches/` that `extract_features.py` reads |
| `<P>/wsis_segmentation/stitches/` | `create_patches_fp.py` L256 + L278 | nothing (created empty) |
| `<P>/wsis_segmentation/process_list_autogen.csv` | `create_patches_fp.py` L91, L224 | nothing downstream |
| `<P>/wsis/` (`--data_path`, `--image_path`) | External (user) | `segment_image.py` → `create_patches_fp.py --source`; `rescale.py`/`extract_features.py` `--image_path` (inference) |
| `<P>/processed_data/` (train-side root, `--processed_data_path`) | External: `HEST_preprocess.py` L18 | read-only for all seven scripts |
| `<P>/processed_data/metadata/{sample}.json` | `HEST_preprocess.py` L54 (`.../metadata/{Sample}.json`) | `rescale.py` L63 (`pixel_size_um_estimated`, `spot_diameter`) |
| `<P>/processed_data/wsis/{sample}.tif` | `HEST_preprocess.py` L52 | `rescale.py` L69 |
| `<P>/processed_data/tissue_seg/{sample}_contours.geojson` | `HEST_preprocess.py` L55 | `rescale.py` L70 |
| `<P>/processed_data/st/{sample}.h5ad` | `HEST_preprocess.py` L51 | `rescale.py` L112 (`sc.read_h5ad`, `obsm['spatial']`) |
| `<P>/processed_data/patches/{sample}.h5` | `HEST_preprocess.py` L49 | `extract_features.py` L148 (`get_HEST1k_patch`) |
| `<P>/processed_data/{patches_vis,pixel_size_vis,spatial_plots,thumbnails,tmp}/` | `HEST_preprocess.py` L21–28 | not used by these seven |
| `<P>/<O>/` | External producers (`prepare_st.py`, `predict.py`, …) | — |
| `<P>/<O>/UNI_multiscale_patches/` | `rescale.py` L35; `extract_features.py` L52 | — |
| `<P>/<O>/UNI_multiscale_patches/wsis_rescaled/` | `rescale.py` L36; re-asserted by `extract_features.py` L54 | `rescale.py` (skip check), `extract_features.py` L73–77, `plot.py` L60, `run_istar.py` L13–14, `run_HistoSweep.py` L333–338 |
| `<P>/<O>/UNI_multiscale_patches/wsis_rescaled/{s}_rescaled.tif` | `rescale.py` L139 | `extract_features.py`, `plot.py`, `run_istar.py`, `run_HistoSweep.py` |
| `<P>/<O>/UNI_multiscale_patches/wsis_rescaled/{s}_mask_rescaled.tif` | `rescale.py` L141 | `extract_features.py`, `plot.py`, `run_istar.py` |
| `<P>/<O>/UNI_multiscale_patches/wsis_rescaled/{s}_loc_rescaled.npy` | `rescale.py` L144 | `extract_features.py` L76 |
| `<P>/<O>/UNI_multiscale_patches/wsis_rescaled/{s}_barcode.pickle` | `rescale.py` L146 (`utils.save_pickle`) | `prepare_st.py` L22 (out of scope) |
| `<P>/<O>/UNI_multiscale_patches/wsis_rescaled/{s}_pixel_size.npy` | `rescale.py` L147 | `extract_features.py` L77 |
| `<P>/<O>/UNI_multiscale_patches/wsis_rescaled/{s}_HistoSweep_mask_rescaled.tif` | `run_HistoSweep.py` L272 (`--mask-suffix` default `HistoSweep`) | `predict_highres*.py` / `run_chunked_tcga.py` (`--histosweep_mask_suffix`); **not** read by `plot.py` |
| `<P>/<O>/UNI_multiscale_patches/check_figure/` | `rescale.py` L38 (`--plot` only); `extract_features.py` L53 (always) | human QC |
| `<P>/<O>/UNI_multiscale_patches/check_figure/{s}_raw_check.jpg` | `rescale.py` L122 | QC |
| `<P>/<O>/UNI_multiscale_patches/check_figure/{s}_rescaled_check.jpg` | `rescale.py` L133 | QC |
| `<P>/<O>/UNI_multiscale_patches/check_figure/{s}_mask_check.jpg` | `extract_features.py` L94 | QC |
| `<P>/<O>/UNI_multiscale_patches/check_figure/{s}_{ps}patch_check.jpg` | `extract_features.py` L96 | QC |
| `<P>/<O>/UNI_multiscale_patches/check_figure/{s}_emb_{ps}_kmeans.png` | `extract_features.py` L136 (`ps == 224` only) | QC |
| `<P>/<O>/UNI_multiscale_patches/locs/` | `extract_features.py` L56 (train only) | `prepare_st.py` L22, `train_utils.py` L236 |
| `<P>/<O>/UNI_multiscale_patches/locs/{s}-bars.pickle` | `extract_features.py` L149 | `prepare_st.py`, `train_utils.py` |
| `<P>/<O>/UNI_multiscale_patches/locs/{s}-locs.pickle` | `extract_features.py` L150 | `prepare_st.py` |
| `<P>/<O>/UNI_multiscale_patches/locs/{s}-meta.pickle` | `extract_features.py` L151 | `prepare_st.py` |
| `<P>/<O>/UNI_multiscale_patches/emb_224/`, `emb_896/`, `emb_3584/` (`emb_{patch_size}/`) | `extract_features.py` L100 | `train_utils.py` L458/L731/L1076, `predict*.py`, `test*.py` |
| `<P>/<O>/UNI_multiscale_patches/emb_{ps}/{s}.npz` | `extract_features.py` L139 (`emb`,`pos`,`flag`,`pad`) | same as above |
| `<P>/<O>/UNI_multiscale_patches/rescale_parameters_{YYYYMMDD-HHMMSS}.csv` | `rescale.py` L157 | nothing in-repo (QC / provenance) |
| `<P>/<O>/Gene_to_predict.csv`, `<P>/<O>/Gene_normalize_factor.csv` | External: `prepare_st.py` L100–101 | `run_istar.py` L41 (`['Genes']`) |
| `<P>/<O>/deconvolution/cell_types.csv` | **External, no in-repo producer** (bulk deconvolution) | `run_istar.py` L42 (`header=None`) |
| `<P>/<O>/cnts/`, `<P>/<O>/bulk/` | External: `prepare_st.py` L21/L23 (via `save_tsv`) | `train_utils.py` (out of scope) |
| `<P>/AURORA_pred/` (`--pred_path`) | External: `predict.py` / `predict_highres.py` | — |
| `<P>/AURORA_pred/pred/` | External producer | `plot.py` L69/L164, `run_istar.py` L62 |
| `<P>/AURORA_pred/pred/{s}-pred.pickle` | External (`predict.py`) | `plot.py` (low-res branch), `run_istar.py` |
| `<P>/AURORA_pred/istar_pred/` | `run_istar.py` L47 | — |
| `<P>/AURORA_pred/istar_pred/{s}/` | `run_istar.py` L54 | `plot.py` (iStar branch) |
| `<P>/AURORA_pred/istar_pred/{s}/he.tif`, `mask.tif`, `mask-small.png` | `run_istar.py` L56/L58/L60 | `plot.py` L134 (`mask-small.png`); istar pipeline |
| `<P>/AURORA_pred/istar_pred/{s}/cnts.tsv`, `gene-names.txt`, `locs.tsv` | `run_istar.py` L75/L78/L87 | `code/istar/impute.py` (reads `gene-names.txt`), `code/istar/utils.py` (`get_locs`) |
| `<P>/AURORA_pred/istar_pred/{s}/pixel-size.txt`, `pixel-size-raw.txt`, `radius.txt` | `code/run_istar_core.sh` L12–14 | `code/istar/*` |
| `<P>/AURORA_pred/istar_pred/{s}/embeddings-gene.pickle` | `code/istar/impute.py` L308–310 | `code/istar/cluster.py`, `marker_score.py` |
| `<P>/AURORA_pred/istar_pred/{s}/cnts-super/` and `cnts-super/{name}.pickle` | `code/istar/impute.py` L326 | `plot.py` L144/L168 |
| `<P>/AURORA_pred/istar_pred/{s}/clusters-gene/`, `states/`, `training-data-plots/` | `code/istar/cluster.py`, `impute.py` | istar QC |
| `<P>/AURORA_pred/istar_pred/{s}/HistoSweep_Output/mask.png` | **Legacy** `code/HistoSweep/run_HistoSweep_single.py` (L8: `output_dir = f"{HE_prefix}/HistoSweep_Output"`) / `run_HistoSweep_multi.py` | `plot.py` L62 and L136 (`--HistoSweep_mask`) — **not** produced by `run_HistoSweep.py` |
| `<P>/AURORA_pred/HistoSweep/` | `run_HistoSweep.py` L344 (`native_out_root`) | `run_HistoSweep.py` only; **deleted** at L390 unless `--keep-native-outputs` |
| `<P>/AURORA_pred/HistoSweep/{s}/mask.png`, `mask-small.png`, `HistoSweep_parameters.txt`, `AdditionalPlots/textureAnalysis_plots/...` | `run_HistoSweep.py` L202–251 (via HistoSweep modules) | intermediate only |
| `<P>/<plot_dir>/` (`AURORA_plots/`) | `plot.py` L45 | final user output |
| `<P>/<plot_dir>/{sample}/` | `plot.py` L59 | final user output |
| `<P>/<plot_dir>/{sample}/{sample}-{gene}-{res}-pixels.png`, `{sample}-{celltype}_proportion-{res}pixels.png`, `{sample}-{gene}-iStar-enhanced.png`, `{sample}-{celltype}_proportion-iStar-enhanced.png` | `plot.py` L103/L131/L160/L188 | final user output |

### 8.2 Distinct top-level names the low-res pipeline touches

`wsis/`, `wsis_segmentation/` (`masks/`, `patches/`, `stitches/`, `process_list_autogen.csv`), `processed_data/` (`metadata/`, `wsis/`, `tissue_seg/`, `st/`, `patches/`), `AURORA_interim/` → `UNI_multiscale_patches/` (`wsis_rescaled/`, `check_figure/`, `locs/`, `emb_224/`, `emb_896/`, `emb_3584/`), `AURORA_interim/deconvolution/`, `AURORA_interim/cnts/`, `AURORA_interim/bulk/`, `AURORA_pred/` → `pred/`, `istar_pred/<sample>/` (+`cnts-super/`, `clusters-gene/`, `states/`, `training-data-plots/`, `HistoSweep_Output/`), `AURORA_pred/HistoSweep/<sample>/` (+`AdditionalPlots/textureAnalysis_plots/`), `AURORA_plots/<sample>/`.

---

## 9. Cross-script mismatches and hazards found while reading

1. **`plot.py --HistoSweep_mask` reads a path nothing in the low-res pipeline writes.** `plot.py` L62/L136 expect `<pred>/istar_pred/<sample>/HistoSweep_Output/mask.png` (legacy `HistoSweep_single.py` layout). `run_HistoSweep.py` writes `<pred>/HistoSweep/<sample>/mask.png` and deletes it at the end of a successful run. Either `--keep-native-outputs` plus a manual copy, or `plot.py` needs updating, is required for the flag to have an effect; today it silently falls back to the default mask with only a printed warning.
2. **`wsis_segmentation/patches/` is created but never populated, and is a different `patches/` from the one consumed.** `extract_features.py` L148 reads `<P>/<processed_data_path>/patches/{name}.h5` (HEST output), whereas `segment_image.py`'s subprocess would write `.h5` under `<P>/wsis_segmentation/patches/` if `--patch` were passed (it never is).
3. **`segment_image.py --preset tcga.csv` default does not resolve in a fresh checkout.** The preset is read as `{project_path}/{preset}` (i.e. `<P>/tcga.csv`), but presets ship in `examples/` (`examples/tcga.csv`, `examples/lung.csv`, and that file also needs `white_thresh`/`black_thresh`, which the built-in `seg_params` dict does not contain). Missing preset ⇒ `pd.read_csv` `FileNotFoundError` inside `create_patches_fp.py` before any work.
4. **`segment_image.py --sample_sheet` with an extension other than `.csv`/`.tsv`** leaves `wsis_list` unbound → `NameError` at L27. Also the sample sheet must have an `Image_folder` column (the `Sample ID`/`File Name` alternatives are commented out).
5. **`rescale.py` writes a timestamped CSV and no stable name**, so any consumer must glob `rescale_parameters_*.csv`; repeated runs accumulate files.
6. **`extract_features.py` accepts `--image_path` and `--seg_path` but never uses them** (dead CLI surface in both train and inference modes); its argparse description is also copy-pasted from `rescale.py`.
7. **`extract_features.py` hardcodes `224` for the KMeans QC plot** (`if patch_size == 224`, L127) instead of `min(patch_sizes)`, so with `--patch_sizes 56 14` (as used by `run_chunked_tcga.py`) no embedding plot is produced.
8. **`plot.py` argument handling is broken in two ways:** L49 `if '.csv' in args.genes and args.genes is not None` — (a) `args.genes` is a *list* (`nargs='+'`), so `'.csv' in [...]` is list membership and never matches `genes.csv`, making the documented "CSV of gene names" feature dead; (b) when only `--celltypes` is passed, `'.csv' in None` raises `TypeError` **before** the `is not None` guard, even though L46 explicitly allows "at least one of --genes or --celltypes". CSV paths are also resolved against CWD, not `--project_path`.
9. **`plot.py` validates `--resolution` too late:** L70 indexes `results['patches_loc_before_pad'][args.resolution]` before the membership check at L72–73, so a bad resolution raises `KeyError` rather than the intended `ValueError` message.
10. **`plot.py` `-r > 0` reads the *low-res* `pred/{sample}-pred.pickle`**, not the `-highres-pred.pickle` written by `predict_highres.py`; the two are different files with different keys.
11. **`run_istar.py` hardcodes `224`/`112`/index `2`** where `--patch_size` is offered: `result['exp_pred'][:,2,:]` and `result['prop_pred'][:,2,:]` assume the third patch resolution is the fine one, and the keep-mask uses literal `+ 224` (L66–67) while `--patch_size` is only used as a dict key (L64) and to name the resolution. `--patch_size != 224` produces inconsistent geometry.
12. **`run_istar.py --sample_sheet` is de-facto mandatory** (default `None` ⇒ `pd.read_csv("./None")`), and the iStar core is launched as `bash $CODE_PATH/run_istar_core.sh …`, i.e. it depends on the `CODE_PATH` **environment variable**, not on `--project_path`; the shell script in turn needs `ISTAR_PATH`.
13. **`run_HistoSweep.py` creates and then deletes its own QC outputs** (`shutil.rmtree(native_out_root)` at L390). Combined with finding 1, `--HistoSweep_mask` cannot work after a default (cleaned-up) run.
14. **BigTIFF predicates differ between scripts:** `rescale.py` L135 uses `height > 40000 and width > 60000` (both), `run_HistoSweep.py` L271 uses `orig_h > 40000 or orig_w > 60000` (either). A 50 000 × 50 000 slide yields a BigTIFF mask from `run_HistoSweep.py` and a classic TIFF mask from `rescale.py`.
15. **`utils.py` disables Pillow's decompression-bomb guard globally** (`Image.MAX_IMAGE_PIXELS = None` at import). Every script that does `import utils` inherits it. `run_istar.py` additionally raises `OPENCV_IO_MAX_IMAGE_PIXELS` to 2^40 at import.
16. **`save_pickle`/`load_pickle` name the file only by caller convention** — no compression, no extension enforcement, protocol 4 (CPython ≤3.12 default). The parallel CLAM pair `save_pkl`/`load_pkl` exists with no `mkdir`, so `save_pickle` should be preferred wherever a directory may not exist.
17. The literal `UNI_multiscale_patches` in `rescale.py`/`extract_features.py`/`plot.py`/`run_istar.py` cannot be overridden, while `run_HistoSweep.py --patches_dir_name` and `run_chunked_tcga.py --patches_dir_name` can — a renaming would silently break the low-res half of the pipeline.
