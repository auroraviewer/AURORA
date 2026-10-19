# Notice: files that disappeared during the reorganisation session

This note records, with evidence, files that were **not** present at the end of
the reorganisation even though they existed at the start. The reorganisation's
own file operations only *create* files (plus the import-hygiene edits listed in
[MIGRATION.md](MIGRATION.md)); nothing in `aurora/`, `tools/`, `tests/` or
`run.sh` deletes anything. This record exists so the change is visible rather
than silent.

## 1. `code_dev/` — missing, not recoverable from the Recycle Bin

`code_dev/` (641 files, a duplicate development copy of `code/`) was listed in
the workspace root at the start of the session and is absent now:

```
> cmd /c "dir /a /b"
...
code
command.sh
docs
examples
...
```

`C:\$Recycle.Bin\<user SID>` contains **no entry whose original path mentions
`code_dev`**, so it was removed with a permanent delete rather than a recycle.
No copy exists anywhere under the user's home directory.

## 2. Sixteen items from `code/` — in the Recycle Bin and recoverable

Recycle-bin metadata (`$I*` records: original path, size, deletion time) shows a
bulk removal from `code/`:

| deleted (local time) | size | original path |
|---|---|---|
| 13:17:55 | 17,982 | `code/train_GCN.py` |
| 13:17:55 | 6,836 | `code/test_GAT.py` |
| 13:17:55 | 6,818 | `code/test_GCN.py` |
| 13:17:55 | 18,016 | `code/train_GAT.py` |
| 13:17:57 | 20,307 | `code/train2.py` |
| 13:17:57 | 20,039 | `code/train_highres2.py` |
| 13:17:58 | 20,014 | `code/train_random.py` |
| 13:18:00 | 12,378 | `code/test_random.py` |
| 13:18:01 | 2,563 | `code/test_pcc_R2C7.py` |
| 13:18:02 | 2,994 | `code/test_falsepositive_R2C7.py` |
| 13:18:05 | 2,748 | `code/run_HistoSweep_old.py` |
| 13:18:09 | 70,620,816 | `code/CATCH_main/` (directory) |
| 13:18:12 | 2,924 | `code/CATCH_config_templet.yaml` |
| 13:18:19 | 8,710 | `code/rescale_BRCA_chemo.py` |
| 13:18:24 | 17,733 | `code/predict_BRCA_immunotherapy.py` |
| 13:18:47 | 15,253 | `code/run_CATCH.py` |

All sixteen are still in the Recycle Bin under
`C:\$Recycle.Bin\S-1-5-21-846469885-2542487897-900814942-1001` (pairs
`$I<id>.<ext>` = metadata, `$R<id>.[ext]` = the data) and can be restored by
opening the Recycle Bin in Explorer, or with:

```powershell
# Inspect what is there, with original paths
python tools\list_recycled_files.py
```

## 3. What this does and does not affect

* **Not affected:** every script the pipeline uses. `code/create_patches_fp.py`,
  `rescale.py`, `extract_features.py`, `extract_features_highres.py`,
  `predict.py`, `predict_highres.py`, `train.py`, `train_highres.py`,
  `prepare_st.py`, `prepare_highres_training.py`, `smooth_cnt.py`, `finetune.py`,
  `test.py`, `test_highres.py`, `plot.py`, `model.py`, `train_utils.py`,
  `datasets.py`, `image_utils.py`, `utils.py`, `run_istar.py`, `run_HistoSweep.py`,
  `run_chunked_tcga.py`, `wsi_core/`, `istar/`, plus `command.sh`, `README.md`,
  `requirements.txt` and `trained_model/` are all present.
  `tests/selftest.py` verifies this on every run.
* **Affected:** the optional CATCH high-resolution feature backend
  (`extract_features_highres.py --use-CATCH`) can no longer run, because
  `code/run_CATCH.py`, `code/CATCH_main/` and `code/CATCH_config_templet.yaml`
  are gone. That backend was already documented as non-runnable from `code/`
  alone (it shells out to `run_CATCH.py`, which the analysis report at
  [docs/_recon/highres_pipeline.md](_recon/highres_pipeline.md) notes was missing
  from `code/`). The default UNI2-h high-resolution path is unaffected.
* **Affected:** the experimental model variants (`train_GCN`, `train_GAT`,
  `train_random`, `train2`, `train_highres2`, the `test_*` probes and the
  BRCA-specific scripts) are gone from `code/`. The new pipeline never used
  them; `aurora/spec.py` has no stage that references them.

## 4. Reproducing this record

`tools/list_recycled_files.py` prints the same table straight from the Recycle
Bin, so this document can be re-verified at any time. It only reads metadata; it
never restores or deletes anything.

## 5. Later removals (after the reorganisation was delivered)

| deleted (local time) | size | original path |
|---|---|---|
| 21:59:44 | 0 | `.aurora_selftest__1p7gb89` (empty scratch directory) |
| 21:59:44 | 0 | `.aurora_selftest_u5qxwrf3` (empty scratch directory) |
| 21:59:44 | 0 | `.probe_t700_h8qjozn3` (empty scratch directory) |
| 21:59:44 | 0 | `.probe_t777_9dsxlf3e` (empty scratch directory) |
| 21:59:44 | 0 | `.probe_w6o1h9sg` (empty scratch directory) |

The five scratch directories listed in section 5 of the previous revision of
this note have now been removed and the workspace root is clean.

### 6. `code/istar/` — removed by the user

| deleted (local time) | size | original path |
|---|---|---|
| 22:13:13 | 227,766 | `<repo>/code/istar` |

The iStar library was removed from `code/` at the same time the iStar stage was
being taken out of the new code system. It is **recoverable** from the Recycle
Bin if needed. `code/run_istar.py` and `code/run_istar_core.sh` are still in
place; only the library directory went.

Consequences:

* The new pipeline has no `istar` stage, no `istar` dependency group and no
  `requirements/optional-istar.txt` (preserved at
  [`docs/legacy/requirements_optional-istar.txt`](legacy/requirements_optional-istar.txt)).
* `code/run_istar.py` can no longer be executed as-is, because it imports
  `istar.preprocess` and `istar.utils` from the removed directory. Restore
  `code/istar/` from the Recycle Bin if you ever need to run it again.
* The `histosweep` stage is unaffected by the iStar removal: it sweeps the
  rescaled H&E image and writes a tissue mask, and `predict_highres` applies it.
  It never consumed iStar output. It is now a **mandatory** stage — see
  [MIGRATION.md §6a](MIGRATION.md).
* `code/plot.py` is unaffected.

