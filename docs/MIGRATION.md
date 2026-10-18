# Migration guide

Nothing was deleted or made incompatible. This page maps the old commands onto
the new ones, explains where the outputs moved, and lists the small import-hygiene
edits that were made inside `code/`.

---

## 1. Old command → new command

`command.sh` still works unchanged with the legacy separate checkpoints. The
table below shows the equivalent in the new code system.

### Inference (apply an existing model to new samples)

| old | new |
|---|---|
| `python -u $CODE_PATH/segment_image.py -p $P -d wsis -s wsis_segmentation --code_path $CODE_PATH --preset lung.csv` | `python -m aurora run --run R --stage segment --preset examples/lung.csv` |
| `python -u $CODE_PATH/rescale.py --inference -p $P -m Inference_samples.csv -o AURORA_interim --image_path wsis --seg_path wsis_segmentation --plot` | `python -m aurora run --run R --stage rescale --plot` |
| `python -u $CODE_PATH/extract_features.py --inference -p $P -m Inference_samples.csv -o AURORA_interim --image_path wsis --seg_path wsis_segmentation -b 128 --plot --UNI_local_path ../uni2h` | `python -m aurora run --run R --stage features_lowres --batch-size 128 --uni-local-path ../uni2h --plot` |
| `python -u $CODE_PATH/extract_features_highres.py -p $P -c $CODE_PATH -m Inference_samples.csv --sample_name_col 1 -o AURORA_interim --plot --UNI_local_path ../uni2h` | `python -m aurora run --run R --stage features_highres --uni-local-path ../uni2h --plot` |
| `python -u $CODE_PATH/predict.py -p $P -i AURORA_interim -t AURORA_pred -j model_parameter_LUAD.json --model_pth AURORA_LUAD.pth --sample_sheet Inference_samples.csv --bulk TCGA_LUAD_expression.csv --save_embedding` | `python -m aurora run --run R --stage predict_lowres --bulk TCGA_LUAD_expression.csv` |
| `python -u $CODE_PATH/predict_highres.py -p $P -i AURORA_interim -t AURORA_pred -j model_parameter_LUAD_highres.json --model_pth AURORA_highres_model_weights.pth --sample_sheet Inference_samples.csv --batch_size 100` | `python -m aurora run --run R --stage predict_highres --prediction-batch-size 100` |
| *(all of the above, in order)* | `python -m aurora run --run R` |
| `python -u $CODE_PATH/run_chunked_tcga.py …` | `python -m aurora run --run R` (chunking is automatic; `code/run_chunked_tcga.py` is still there for very large cohorts) |

Every command takes `--project`, `--run`, `--samples` and `--model` once at the
top level instead of repeating `-p/-i/-t/-j/--model_pth`.

### Training

| old | new |
|---|---|
| `python -u $CODE_PATH/rescale.py -p $P -d hest_data -m Training_samples.csv -o debug --plot` | `python -m aurora run --mode training --run R --stage rescale --processed-data hest_data --plot` |
| `python -u $CODE_PATH/extract_features.py -p $P -d hest_data -m Training_samples.csv -o debug -b 128 --UNI_local_path ./uni2h` | `python -m aurora run --mode training --run R --stage features_lowres --uni-local-path ./uni2h` |
| `python -u $CODE_PATH/prepare_st.py -p $P -d hest_data -m Training_samples.csv -o debug` | `… --stage prepare_st` |
| `python -u $CODE_PATH/smooth_cnt.py -p $P -d hest_data -m Breast/all_samples.csv -o Breast` | `… --stage smooth_cnts` |
| `python -u $CODE_PATH/train.py -p $P -d hest_data -i Breast -t Breast -j Breast/model_parameter_BRCA.json --train_samples Breast/train_samples.csv --train_sample_col 0 --log --smooth_cnts` | `… --stage train_lowres --smooth-counts` |
| `python -u $CODE_PATH/extract_features_highres.py -p $P -c $CODE_PATH -m Training_samples.csv -o debug --UNI_local_path ./uni2h` | `… --stage features_highres --uni-local-path ./uni2h` |
| `python -u $CODE_PATH/prepare_highres_training.py -p $P -t debug -d hest_data -j model_parameter_BRCA.json --model_pth AURORA_BRCA.pth --sample_sheet debug/Training_samples.csv` | `… --stage prepare_highres_training` |
| `python -u $CODE_PATH/train_highres.py -p $P -t debug -d hest_data -i debug -j model_parameter_BRCA_highres.json --train_samples Training_samples.csv --log` | `… --stage train_highres` |
| *(no equivalent)* | `… --stage combine_checkpoints` — merges the two trained models into one combined `.pth` |
| `python -u $CODE_PATH/test.py …` | unchanged: the developer evaluation scripts still work |
| `python -u $CODE_PATH/finetune.py …` | `python -m aurora run --mode training --run R --stage finetune --finetune-gene-list …` |

Add `--auto-prerequisites` to any of these to let the CLI run whatever is
missing first.

---

## 2. Where the outputs moved

### Before

Everything landed in the project root under ad-hoc names:

```
<project>/
├── wsis_segmentation/{masks,contours,patches,process_list_autogen.csv}
├── AURORA_interim/UNI_multiscale_patches/
│   ├── wsis_rescaled/  check_figure/  locs/
│   ├── emb_224/ emb_896/ emb_3584/ emb_56/ emb_14/     <- low- and high-res together
│   └── rescale_parameters_<timestamp>.csv              <- a new file every run
├── AURORA_pred/
│   ├── pred/                                           <- both stages, same folder
│   ├── processed_pkl/  processed_pkl_test_highres/
│   └── istar_pred/  HistoSweep/
├── AURORA_plots/
├── AURORA_train/                                       <- or "Breast", or "debug", ...
└── test_smooth_pred/, debug/, Breast/, ...             <- per-experiment folders
```

### After

Everything for one run lives under one directory:

```
<project>/aurora_runs/<run>/
├── run.json
├── config/
├── logs/
├── 01_segmentation/{masks,contours,patches,viz}
├── 02_features/UNI_multiscale_patches/{wsis_rescaled,check_figure,locs,emb_*}
├── 03_predictions/{lowres,highres}/{pred,processed_pkl*}
├── 04_histosweep/  05_plots/
├── checkpoints/       (training)
└── tmp/
```

The two names the original scripts hard-code — `UNI_multiscale_patches` and
`processed_pkl` / `processed_pkl_test_highres` — are kept as *inner* directory
names, so the original scripts keep working when they are pointed at a stage
directory. Only the top level changed.

Copy an old tree into the new layout (copies only, never deletes):

```bash
python -m aurora migrate-layout --project . --run migrated --dry-run
python -m aurora migrate-layout --project . --run migrated
```

---

## 3. Naming: everything is keyed by `Sample_name`

Every per-sample artefact the pipeline writes — rescaled images, embeddings,
predictions, plots, and the tissue masks — is named after `Sample_name`.
`Image_name` is only used to find the slide on disk.

This matters whenever the two differ, which is the normal TCGA case because the
GDC appends a file UUID:

```csv
Image_name,Sample_name
TCGA-44-5645-01B-04-BS4.76ED5128-FB3D-44C7-A679-848860CAA3C5.svs,TCGA-44-5645-01B-04-BS4
```

Two inconsistencies in the original code follow from this, both handled:

0. **Optional columns are only written when they carry information.** Several
   legacy scripts branch on column *presence*, not on the values:
   `code/rescale.py` does `if 'Image_folder' in meta.columns` and
   `code/predict.py` does the same for `Bulk_sample_name`. Because
   `pandas.read_csv` turns a blank field into `NaN`, a sheet with a present but
   empty `Image_folder` column used to make `rescale.py` build
   `…/wsis/nan/<file>.svs` and fail with `FileNotFoundError`. The writer now
   drops a column when every row is empty, and completes the blanks when only
   some rows are empty (`Image_folder` → `.`, a no-op path element;
   `Bulk_sample_name` → `Sample_name`). `Image_name` and `Sample_name` are always
   written first, in that order, because `extract_features_highres.py` is given
   `--sample_name_col 1`.
1. `code/create_patches_fp.py` writes `masks/<slide_file_stem>.geojson`, not
   `<Sample_name>.geojson`, because it names the mask after the file it found in
   `--source`. `aurora/naming.py` links the two spellings together after
   segmentation (hard link — one inode, no extra storage, nothing overwritten),
   so both `<Sample_name>.geojson` and `<image_stem>.geojson` exist. That also
   satisfies `code/rescale.py`, which derives the mask name from the `Image_name`
   column of the sheet it is given.
2. `code/rescale.py` builds the *slide* path from the same column, so an
   `Image_name` that does not match a file on disk fails. The pipeline resolves
   each row against the filesystem — declared `Image_name` first, then
   `<Sample_name>` with the same extension — and hands the stages
   `config/samples_resolved.csv`, a copy of the sheet with the resolved file
   name. The substitution is reported as a note, and a completely unresolvable
   slide is reported before anything runs, listing every path that was tried.

`tests/test_naming.py` checks all of this; it is also part of
`tests/selftest.py`.

---

## 4. iStar was removed from the pipeline

iStar is no longer used, so it has been taken out of the new code system
entirely:

| removed | where it used to be |
|---|---|
| the `istar` stage | `aurora/spec.py` (`STAGE_ORDER`, `STAGE_PACKAGES`) |
| the `--istar-epochs` option | `aurora/cli.py`, `aurora/config.py` |
| the `istar` dependency group (scikit-image, einops, pytorch-lightning, umap-learn, seaborn, tomli) | `aurora/deps.py` |
| `requirements/optional-istar.txt` | now [`docs/legacy/requirements_optional-istar.txt`](legacy/requirements_optional-istar.txt) |
| the `04_istar/` run directory | the layout is now `04_histosweep/`, `05_plots/` |

**The entry script was kept, the library was not.** `code/run_istar.py` and
`code/run_istar_core.sh` are still in the repository. `code/istar/` itself was
removed from the working tree by the user while this change was being made (it is
in the Recycle Bin — see
[removed_files_notice.md](removed_files_notice.md#6-codeistar-removed-by-the-user)).
Either way, nothing in `aurora/`, `tools/`, `run.sh` or the dependency policy
references iStar any more, so it is inert: an inference install does not need
`einops`, `pytorch-lightning`, `umap-learn`, `seaborn` or `tomli`.

The `histosweep` stage was **not** an iStar consumer, contrary to what an earlier
revision of this document said. `code/run_HistoSweep.py` sweeps the *rescaled H&E
image* and writes `<sample>_HistoSweep_mask_rescaled.tif` beside it; that mask is
what the high-resolution prediction applies. It is now a mandatory stage, it
depends on `rescale` rather than `predict_lowres`, and `predict_highres` declares
it as a prerequisite. See §6a.

`code/plot.py` is unaffected: the `plot` stage feeds it a projection of the
predictions into the layout it expects (see §7), so it works with the default
`--output-format modern` too.

---

## 5. The combined checkpoint

The two model files

```
trained_model/AURORA_LUAD_V2.pth                    (low resolution)
trained_model/AURORA_highres_model_weights.pth      (high resolution)
```

become one:

```bash
python tools/merge_aurora_checkpoints.py \
    --lowres trained_model/AURORA_LUAD_V2.pth \
    --highres trained_model/AURORA_highres_model_weights.pth \
    --lowres-params trained_model/model_parameter_LUAD.json \
    --highres-params trained_model/model_parameter_LUAD_highres.json \
    --gene-list trained_model/LUAD_gene.csv \
    --cell-types trained_model/LUAD_cell_types.csv \
    --norm-factors trained_model/LUAD_gene_normalize_factor.csv \
    --model-name AURORA_LUAD --cancer-type LUAD \
    --out trained_model/AURORA_LUAD_combined.pth
```

* the two source files are only read;
* every byte is copied verbatim and its SHA-256 is recorded — **the archive
  layout is never rewritten**, because `torch.save` nests every record under a
  root directory named after the output file and `torch.load` rejects an archive
  without one (`RuntimeError: file in archive is not in a subdirectory:
  data.pkl`). The root is recorded in the manifest as `archive_root` and
  `aurora validate` re-checks it;
* the manifest records the patch sizes, the embedding dimensions, the model
  parameters and the `highres → lowres` requirement.

The **new** code system only accepts the combined file. Handing it a legacy
single-stage `.pth` raises an error that contains the merge command above. The
**old** scripts under `code/` still read the old separate files, unchanged.

> **Rebuild any checkpoint built before this guide's current revision.** The first
> version of the merge tool "normalised" the archive layout; that was wrong and
> produced files that `torch.load` rejects with
> `file in archive is not in a subdirectory: data.pkl`. Payloads are now copied
> verbatim. `aurora validate`, the weight loader and `aurora run` all detect the
> old format before PyTorch is involved and print the rebuild command, so a stale
> checkpoint cannot fail silently.

---

## 6. Edits made inside `code/`

Every original script is still present, with the same command-line interface and
the same numerics. The only changes are import-hygiene fixes so that an
inference-only installation does not need training-only packages. In each case
the original statement is kept as a comment next to the replacement.

| file | change | why |
|---|---|---|
| `code/train_utils.py` | `from sklearn.cluster import MiniBatchKMeans` → resolved on first attribute access (`__getattr__`) | `train_utils` is imported by `predict.py`, `predict_highres.py` and `datasets.py`, so scikit-learn used to be mandatory for inference. `train_utils.MiniBatchKMeans` and `from train_utils import MiniBatchKMeans` still work when it is installed. |
| `code/model.py` | `from torch_geometric.nn import GraphConv, GCNConv` → resolved inside `GCN.__init__` / `GraphLSTM.__init__` | only the graph model variants need `torch_geometric`; the AURORA model does not |
| `code/rescale.py` | `import scanpy as sc` → inside `read_h5ad()` | only the training branch reads `.h5ad` files |
| `code/rescale.py` | `geopandas.read_file` → `read_geojson_geometries()` (standard library) | the GeoJSON geometries passed to `rasterio.features.rasterize` are identical, so the mask is unchanged, but geopandas is no longer needed |
| `code/rescale.py` | `gpd.read_file(...)` + `.is_empty` → `geometry_is_empty()` | same behaviour, no geopandas |
| `code/extract_features.py` | matplotlib + `sklearn.cluster.KMeans` → inside `plot_embedding_kmeans()` | only the `--plot` check figure needs them; if scikit-learn is absent the figure is skipped with a note |
| `code/extract_features_highres.py` | matplotlib + KMeans / MiniBatchKMeans → inside `plot_catch_kmeans()` / `plot_grid_kmeans()` | same reasoning |
| `code/extract_features_highres.py` | `import pyvips` → inside `import_pyvips()` | only the optional CATCH backend needs pyvips |
| `code/datasets.py` | `keep_flag = np.sum(...) > ...` → `keep_flag = bool(np.sum(...) > ...)` | **this one is a bug fix, not hygiene.** `np.sum(...) > ...` returns a NumPy scalar (`np.bool_`) in current NumPy, and `torch.utils.data.default_collate` sends a batch of NumPy scalars to `torch.as_tensor(list)`, which raises `TypeError: 'numpy.bool' object cannot be interpreted as an integer` because `np.bool_` has no `__index__`. Wrapping in `bool()` fixes the low-resolution feature stage. Note `np.bool_.__name__` is `'bool'`, so the type looks correct in a traceback or a `type(x).__name__` print — check `type(x).__module__` instead. `tests/test_dataloader_collate.py` guards it. |
| `code/plot.py` | the twelve hardcoded `//10` conversions → `//downscale`, with a new `--downscale` argument | **additive, and the default stays 10**, so existing callers get identical figures. The `plot` stage passes 7, because a figure must be at least 1/7 of the rescaled slide: at 1/10 a `56` px patch covered 5 pixels and a `14` px patch only 1, so the high-resolution levels could not be read. |
Nothing else was touched, and no file was removed. Verify with:

```bash
python tools/audit_inference_deps.py     # what the inference path really imports
python tests/selftest.py                 # includes a check that every original script still parses
```

---

## 6a. HistoSweep is mandatory

`code/run_HistoSweep.py` sweeps the **rescaled H&E image** (density + texture +
ratio filtering) and writes a 0/1 tissue mask beside it:

```
<run>/02_features/UNI_multiscale_patches/wsis_rescaled/<sample>_HistoSweep_mask_rescaled.tif
```

It does not consume iStar output, and HistoSweep itself is vendored at
`code/HistoSweep`, so nothing has to be cloned.

Changes made:

| what | where | why |
|---|---|---|
| `histosweep` is no longer `optional` and is in `INFERENCE_PIPELINE` | `aurora/spec.py` | it must always run |
| its prerequisite is `rescale`, not `predict_lowres` | `aurora/spec.py` | it only needs the rescaled slide, so it need not wait for a prediction |
| its declared output is the **mask**, not HistoSweep's native files | `aurora/spec.py` (`_histosweep_outputs`) | `run_HistoSweep.py` deletes its native outputs on success, so verifying those failed every time |
| `predict_highres` requires `histosweep` | `aurora/spec.py` | enforced by the fail-closed gate |
| `predict_highres` **applies** the mask | `aurora/inference/predict_highres.py`, `aurora/masks.py` | ported from `code/predict_highres_TCGA.py` (`load_histosweep_mask`, `histosweep_keep_flags`), the only variant that used it |
| a missing mask is an error | `aurora/masks.py::require_mask` | silently predicting without it would quietly change which patches are kept |
| `--histosweep-patch-size`, `--rescaled-dir` | `aurora/cli.py`, `aurora/inference/common.py` | the alignment super-pixel size (default 14) and the mask directory were hardcoded |

**Axis order.** The mask is `(row = y, col = x)`. The prediction points are passed
in the same `(y, x)` order that the file stores under `pos`, *not* the `(x, y)`
spelling used by `patches_loc_before_pad`, or the mask would filter the
transpose. `tests/test_histosweep_mask.py` pins this with an asymmetric mask that
gives a different answer under the two orders.

**This changes results.** High-resolution predictions made before this change
were computed over every patch; they must be re-run
(`--stage predict_highres --force`) to have the mask applied.

---

## 6b. Default output formats

The defaults now mirror the original `command.sh`, so the compact files are what
you get unless you ask for something else.

| stage | default | opt-out |
|---|---|---|
| `predict_lowres` | `--uint8-output` (quantised) | `--no-uint8-output` |
| `predict_lowres` | `--only-224` (finest level only, matching `predict.py --only_224`) | `--no-only-224` |
| `predict_highres` | `--uint8-output`, which writes **HDF5** `<sample>-highres-pred.h5` | `--no-uint8-output` (writes the float pickle) |
| `predict_highres` | `--h5-layout spatial_gene_delta` | `--h5-layout patch_major` |
| `predict_highres` | `--h5-compression zstd --h5-compression-level 9` | `--h5-compression gzip`/`none` |

Rationale: `command.sh` lines 36-52 run `code/run_chunked_tcga.py` with
`--uint8_output --h5_layout spatial_gene_delta --h5_compression zstd
--h5_compression_level 9`, and the stepwise command on line 27 uses
`predict.py`. Those were the intended outputs; they are now the defaults rather
than flags to remember.

The format itself is unchanged. `aurora/h5out.py` and `aurora/h5in.py` are ports
of the writer and reader that already existed:

| new module | ported from | what it does |
|---|---|---|
| `aurora/h5out.py` | `code/predict_highres_TCGA_optimized.py` (`write_highres_uint8_h5` and helpers) | writes the quantised matrices, spatially sorted and delta-encoded along the row axis, with the same HDF5 attributes (`encoding = spatial_gene_delta_v1`, `logical_shape`, `delta_block_rows`, `row_order_<ps>`, …) |
| `aurora/h5in.py` | `code/highres_h5_loader.py` | reads them back, optionally dequantising with the stored per-channel maxima and restoring the original row order |

`zstd` needs `hdf5plugin`; without it the stage says so and suggests `gzip`,
rather than writing a file nothing can open.

**Plotting still works.** Because `--uint8-output` is now the default, the
high-resolution output is HDF5, which `code/plot.py` cannot open — so the `plot`
stage reads it through `aurora/h5in.py` and converts it into the legacy view like
any other source. The values are dequantised with the stored maxima, and
`patch_loc_<ps>` is flipped from `(y, x)` to the `(x, y)` that `code/plot.py`
expects. Only `--compress-output` (`.npz.zst`) remains unplottable, and the error
says so.

---

## 7. Things that were already inconsistent

These are pre-existing behaviours. Some are preserved rather than silently
changed; the ones the new pipeline actively works around are marked.

| observation | where | what the new code does |
|---|---|---|
| **per-sample artefacts are keyed by `Sample_name`, but the tissue masks are keyed by the slide file** | `code/create_patches_fp.py` names masks after the file it finds in `--source`, while `rescale.py`, `extract_features*.py` and `predict*.py` all key off `Sample_name`; `rescale.py` then asks for the mask under the *image* name | **worked around**: after segmentation the pipeline hard-links `<Sample_name>.geojson` and `<image_stem>.geojson` together (`aurora/naming.py`), and the gate accepts either spelling; if `Image_name` names no file on disk, the slide is also looked up as `<Sample_name>` + the same extension |
| `code/predict.py` writes a dict-of-patch-sizes layout, but `code/plot.py` and `code/run_istar.py` read the older 3-D array layout | `predict.py`, `plot.py`, `run_istar.py` | **worked around**: the `plot` stage projects the resolution being plotted into `tmp/legacy_pred_view/` and points `code/plot.py` at that (`aurora/plotting.py`), so no re-run is needed; `tools/convert_prediction_format.py` does the same by hand. `run_istar.py` is no longer wired in (see §4) |
| `code/plot.py` always opens `<pred>/pred/<sample>-pred.pickle`, but `predict_highres` writes `<sample>-highres-pred.pickle` | `plot.py:69`, `predict_highres.py` | **worked around**: `--plot-predictions highres` builds the view under `tmp/legacy_pred_view_highres/`, renaming to the filename `plot.py` opens, and writes figures to `05_plots/highres/`. `code/plot.py` needs no change |
| `code/plot.py` requires `cell_types` to be a NumPy array (it calls `.tolist()`) while `genes` must be a list (`.index()`) | `plot.py:82,111` | **fixed** in `predict_lowres`/`predict_highres`/`to_legacy`, matching `code/predict.py`, which builds cell types with `.values.squeeze()`. A list made the cell-type branch fail with `AttributeError: 'list' object has no attribute 'tolist'` |
| the modern layout stores `patches_loc_before_pad` in `(y, x)`, but `code/plot.py` flips it again and uses column 0 as `y`, so it must be `(x, y)` | `predict_lowres` | **fixed**: `pos` stays `(y, x)` to match `code/predict.py`, `patches_loc_before_pad` is now `(x, y)`. A `(y, x)` value would have produced transposed figures; `tests/test_prediction_layout.py` replays plot.py's indexing to catch it |
| `code/run_istar.py` and `code/run_HistoSweep.py` are POSIX-only (`bash`, `mv`, `rm`) | those scripts | only the `histosweep` stage remains, and it is optional; run it on Linux/macOS |
| `code/plot.py` tested `'.csv' in args.genes` / `args.celltypes` **before** the `is not None` guard, so either argument left out crashed | `plot.py:58,62` | **fixed**: the guard now comes first, so `--genes` alone and `--celltypes` alone both work, which is what the script's own "at least one of" error always claimed. `and` evaluates left to right, which is why the original only worked when *both* were passed |
| `code/train.py` / `code/train_highres.py` reference `writer` even without `--log` | those scripts | the `train_lowres` / `train_highres` stages always pass `--log` |
| the shipped low-resolution checkpoint is a re-zip with a top-level directory | `AURORA_LUAD_V2.pth` | the merge tool strips it so `torch.load` sees the canonical layout, and reports that it did |
| `LUAD_cell_types.csv` has no header, `LUAD_gene.csv` does | `trained_model/` | the asset readers detect this and are validated at merge time |
| the normalisation-factor table has two rows with an empty value (`DAZ2`, `HNRNPCL3`), which `pandas` turns into `NaN` | `LUAD_gene_normalize_factor.csv` | the reader only parses the genes actually in the gene list, and refuses a non-numeric value for a gene that *is* used |
| `np.sum(bool/float array) > threshold` yields a NumPy scalar (`np.bool_`), whose `__name__` is `'bool'` | `code/datasets.py` `PatchDataset.__getitem__` | **fixed** by wrapping in `bool()`; see §6 |
| both shipped checkpoints were saved from CUDA | `trained_model/` | the loader always passes `map_location`, so CPU-only machines work |
| `segment_image.py --preset tcga.csv` resolves against the project root, not `examples/` | `segment_image.py` | `aurora` searches the project, the current directory and `examples/` |
| `--train_samples` resolves against the project root in `train_highres.py` but against the run root in `train.py` | those scripts | the pipeline materialises `config/train_samples.csv` where each expects it |

---

## 8. If you want the old behaviour back

Nothing needs undoing. Use `command.sh` and `code/` exactly as before with the
legacy separate checkpoints; the new package is additive. The only files that
changed are the import-hygiene edits in section 4, and each of those preserves
the original import statement in a comment.
