#!/usr/bin/env python3
"""Run the TCGA high-resolution inference workflow in chunks of samples.

Splits the samples in ``--sample_sheet`` into ``--chunk_size`` groups and, for
each chunk, runs the three-step pipeline in order:

    1. extract_features_highres_optimized.py   (high-res UNI embeddings)
    2. run_HistoSweep.py                       (HistoSweep tissue-quality masks)
    3. predict_highres_TCGA_optimized.py       (AURORA predictions -> optimized uint8 HDF5)

After a chunk finishes successfully, its per-sample high-resolution embeddings
(``<interim>/UNI_multiscale_patches/emb_<ps>/<sample>.npz``) and processed test
pickles (``<pred>/processed_pkl_test_highres/<sample>.pkl``) are deleted so
disk usage stays bounded.  The shared rescaled WSIs/masks are kept; HistoSweep
masks are kept unless ``--delete-histosweep-masks`` is given.  The folder name
under ``<interim>`` holding these artifacts (``UNI_multiscale_patches`` by
default) is configurable with ``--patches_dir_name`` and is forwarded to every
step.

Each step is invoked on a small per-chunk CSV (a subset of the sample sheet), so
all three tools only see the current chunk.  Processing is idempotent
(extraction skips samples whose embeddings exist, prediction reuses cached
pickles), so an interrupted run can be resumed with ``--chunk-start``.

predict_highres_TCGA_optimized.py is self-contained: it loads its own high-res
data via ``load_data_pred_TCGA_highres`` (defined inside the script) and no
longer depends on ``train_utils.load_data2_pred``.  The HistoSweep mask
suffix/patch size set on this script are forwarded to both run_HistoSweep and
the predict step so the mask is written, read, and cleaned up under the same
name.  The optimized HDF5 writer is configured through the forwarded
``--h5_layout``, ``--h5_compression``, ``--h5_compression_level``,
``--h5_chunk_rows``, ``--h5_chunk_channels`` and ``--h5_store_row_order``
options (see debug.sh).

Example (mirrors the "TCGA workflow" block in debug.sh, 5 samples per chunk)::

    python -u $CODE_PATH/run_chunked_tcga.py -p $PROJECT_PATH -c $CODE_PATH \\
        -m Inference_samples.csv --chunk_size 5
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the TCGA high-res inference workflow (extract -> HistoSweep -> "
            "predict) in chunks of samples, deleting per-chunk embeddings/pickles."
        )
    )
    # Paths
    parser.add_argument("--project_path", "-p", type=str, default="./",
                        help="Main directory path (same as debug.sh PROJECT_PATH).")
    parser.add_argument("--code_path", "-c", type=str, default="./AURORA/code/",
                        help="Directory containing the AURORA code scripts.")
    parser.add_argument("--sample_sheet", "-m", type=str, default="Inference_samples.csv",
                        help="CSV with a 'Sample_name' column (subset CSV is passed to each step).")
    parser.add_argument("--sample_name_col", "--sample-name-col", type=int, default=1,
                        help="Column index used by extract_features (--sample_name_col).")
    parser.add_argument("--interim_path", "-i", type=str, default="AURORA_interim")
    parser.add_argument("--patches_dir_name", type=str, default="UNI_multiscale_patches",
                        help="Name of the subdirectory under <interim> that holds rescaled "
                             "WSIs, HistoSweep masks, and multiscale embeddings "
                             "(default: UNI_multiscale_patches). Forwarded to extract, "
                             "HistoSweep, and predict so all three agree on the name.")
    parser.add_argument("--pred_path", "-t", type=str, default="AURORA_pred",
                        help="Prediction output directory (contains processed_pkl_test_highres/).")
    parser.add_argument("--histosweep_tmp", type=str, default="HistoSweep_tmp",
                        help="'pred_path' passed to run_HistoSweep.py (-t).")
    parser.add_argument("--histosweep_path", type=str, default="../HistoSweep/HistoSweep",
                        help="Path to the HistoSweep package directory.")
    parser.add_argument("--histosweep_patch_size", "--histosweep-patch-size", type=int, default=14)
    parser.add_argument("--histosweep_mask_suffix", type=str, default="HistoSweep",
                        help="Suffix of the HistoSweep mask file (before '_mask_rescaled.tif').")

    # Model / prediction
    parser.add_argument("--parameter_json", "-j", type=str,
                        default="model_parameter_LUAD_highres.json")
    parser.add_argument("--model_pth", type=str, default="AURORA_highres_model_weights.pth")
    parser.add_argument("--uni_local_path", "--UNI_local_path", type=str, default="../uni2h")
    parser.add_argument("--predict_batch_size", "--predict-batch-size", type=int, default=100)
    parser.add_argument(
        "--float16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use float16 prediction buffers to reduce RAM usage (default: on).",
    )

    # Prediction output (predict_highres_TCGA_optimized.py / optimized HDF5)
    parser.add_argument("--uint8_output", action=argparse.BooleanOptionalAction, default=True,
                        help="Pass --uint8_output to predict (uint8 HDF5 path); disable with "
                             "--no-uint8_output for the sparse-pickle path.")
    parser.add_argument("--compress_output", action="store_true",
                        help="Also pass --compress_output to predict (write .npz.zst instead of HDF5).")
    parser.add_argument("--h5_layout", choices=["spatial_gene_delta", "patch_major"],
                        default="spatial_gene_delta",
                        help="HDF5 prediction layout (passed to predict as --h5_layout).")
    parser.add_argument("--h5_compression", choices=["zstd", "gzip", "none"], default="zstd",
                        help="HDF5 filter (zstd requires hdf5plugin; gzip is the portable fallback).")
    parser.add_argument("--h5_compression_level", type=int, default=9,
                        help="Zstd -131072..22 or gzip 0..9 compression level.")
    parser.add_argument("--h5_chunk_rows", type=int, default=16384,
                        help="Rows per independently decodable spatial-delta block.")
    parser.add_argument("--h5_chunk_channels", type=int, default=256,
                        help="Channels per HDF5 chunk for transformed prediction matrices.")
    parser.add_argument("--h5_store_row_order", action=argparse.BooleanOptionalAction, default=True,
                        help="Store the sorted-row-to-original-row permutation in the HDF5.")

    # Chunking
    parser.add_argument("--chunk_size", type=int, default=5,
                        help="Number of samples processed per chunk.")
    parser.add_argument("--chunk-start", type=int, default=0,
                        help="0-based index of the first chunk to process (resume).")
    parser.add_argument("--chunk-end", type=int, default=None,
                        help="Exclusive 0-based index of the last chunk to process "
                             "(default: process through the last chunk).")

    # Feature extraction parameters (defaults mirror the debug.sh TCGA workflow;
    # both --flag_name and --flag-name spellings are accepted)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=128)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--compile_mode", "--compile-mode", type=str, default="off")
    parser.add_argument("--aggregation_mode", "--aggregation-mode", type=str, default="auto")
    parser.add_argument("--strip_temp_dir", "--strip-temp-dir", type=str, default=None,
                        help="Scratch dir for strip accumulation; defaults to "
                             "<interim>/<patches_dir_name>/strip_tmp.")
    parser.add_argument("--smoothing_workers", "--smoothing-workers", type=int, default=4)
    parser.add_argument("--smoothing_row_chunk", "--smoothing-row-chunk", type=int, default=128)
    parser.add_argument(
        "--smoothing_backend",
        "--smoothing-backend",
        choices=["auto", "cpu", "gpu"],
        default="gpu",
        help="Smoothing backend used by extract_features_highres_optimized.py.",
    )
    parser.add_argument(
        "--smoothing_channel_chunk",
        "--smoothing-channel-chunk",
        type=int,
        default=1536,
    )
    parser.add_argument(
        "--smoothing_gpu_algorithm",
        "--smoothing-gpu-algorithm",
        choices=["auto", "prefix", "separable", "pool"],
        default="auto",
        help="CUDA box-filter algorithm used by the extractor.",
    )
    parser.add_argument(
        "--smoothing_column_tile",
        "--smoothing-column-tile",
        type=int,
        default=4096,
        help="Maximum occupied-column span per CUDA smoothing tile.",
    )
    parser.add_argument(
        "--smoothing_diagnostics",
        "--smoothing-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print CUDA smoothing timing and tiling diagnostics (default: on).",
    )
    parser.add_argument("--filter_prop", "--filter-prop", type=float, default=0.5)
    parser.add_argument("--patch_sizes", "--patch-sizes", nargs="+", type=int, default=[56, 14],
                        help="Output patch sizes; must match what extraction produces "
                             "and what the per-chunk cleanup removes.")
    parser.add_argument("--inference", action=argparse.BooleanOptionalAction, default=True,
                        help="Pass --inference to extract_features (default: on); disable "
                             "with --no-inference.")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False,
                        help="Pass --plot to extract_features (default: off); enable with "
                            "--plot.")
    parser.add_argument("--new_format_sample_sheet", action=argparse.BooleanOptionalAction, default=False,
                        help="If the sample sheet is in the new format (with 'File Name' column), convert it to the old format (with 'Sample_name' column) before chunking.")

    # Cleanup / control
    parser.add_argument("--no-delete-embeddings", action="store_true",
                        help="Keep per-chunk high-res embeddings after each chunk.")
    parser.add_argument("--no-delete-pkls", action="store_true",
                        help="Keep per-chunk processed test pickles after each chunk.")
    parser.add_argument("--delete-histosweep-masks", action="store_true",
                        help="Also delete each chunk's HistoSweep mask after it finishes.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="On a step failure, move on to the next chunk instead of aborting.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands that would run without executing them.")
    parser.add_argument("--tmp_dir", type=str, default=None,
                        help="Directory for the per-chunk CSVs (auto-created and cleaned if unset).")
    parser.add_argument("--chemo_format", action="store_true",
                        help="If the sample sheet is in the chemo format (with 'File Name' column), convert it to the old format (with 'Sample_name' column) before chunking.")
    return parser


def _run_step(name: str, cmd, dry_run: bool) -> bool:
    print(f"\n===== [{name}] =====", flush=True)
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return True
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        print(f"[error] '{name}' failed with exit code {result.returncode}", flush=True)
        return False
    return True


def _extract_cmd(args, chunk_csv: Path) -> list:
    project = str(args.project_path)
    strip_tmp = args.strip_temp_dir or os.path.join(
        project, args.interim_path, args.patches_dir_name, "strip_tmp"
    )
    cmd = [
        sys.executable, "-u",
        os.path.join(args.code_path, "extract_features_highres_optimized.py"),
        "-p", project,
        "-c", args.code_path,
        "-m", str(chunk_csv),
        "--sample_name_col", str(args.sample_name_col),
    ]
    if args.inference:
        cmd.append("--inference")
    cmd += [
        "-o", args.interim_path,
    ]
    if args.plot:
        cmd.append("--plot")
    cmd += [
        "--UNI_local_path", args.uni_local_path,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--stride", str(args.stride),
        "--compile-mode", args.compile_mode,
        "--aggregation-mode", args.aggregation_mode,
        "--strip-temp-dir", strip_tmp,
        "--smoothing-workers", str(args.smoothing_workers),
        "--smoothing-backend", args.smoothing_backend,
        "--smoothing-gpu-algorithm", args.smoothing_gpu_algorithm,
        "--smoothing-row-chunk", str(args.smoothing_row_chunk),
        "--smoothing-channel-chunk", str(args.smoothing_channel_chunk),
        "--smoothing-column-tile", str(args.smoothing_column_tile),
        "--filter-prop", str(args.filter_prop),
        "--patch_sizes", *[str(ps) for ps in args.patch_sizes],
        "--patches_dir_name", args.patches_dir_name,
    ]
    cmd.append(
        "--smoothing-diagnostics"
        if args.smoothing_diagnostics
        else "--no-smoothing-diagnostics"
    )
    return cmd


def _histosweep_cmd(args, chunk_csv: Path) -> list:
    return [
        sys.executable, "-u",
        os.path.join(args.code_path, "run_HistoSweep.py"),
        "-p", args.project_path,
        "-i", args.interim_path,
        "-t", args.histosweep_tmp,
        "--sample_sheet", str(chunk_csv),
        "--histosweep_path", args.histosweep_path,
        "--patch_size", str(args.histosweep_patch_size),
        "--mask-suffix", args.histosweep_mask_suffix,
        "--patches_dir_name", args.patches_dir_name,
    ]


def _predict_cmd(args, chunk_csv: Path) -> list:
    cmd = [
        sys.executable, "-u",
        os.path.join(args.code_path, "predict_highres_TCGA_optimized.py"),
        "-p", args.project_path,
        "-i", args.interim_path,
        "-t", args.pred_path,
        "-j", args.parameter_json,
        "--model_pth", args.model_pth,
        "--sample_sheet", str(chunk_csv),
        "--batch_size", str(args.predict_batch_size),
        "--prediction_dtype", "float16" if args.float16 else "float32",
        "--histosweep_mask_suffix", args.histosweep_mask_suffix,
        "--histosweep_patch_size", str(args.histosweep_patch_size),
        "--patches_dir_name", args.patches_dir_name,
    ]
    if args.uint8_output:
        cmd.append("--uint8_output")
    cmd += [
        "--h5_layout", args.h5_layout,
        "--h5_compression", args.h5_compression,
        "--h5_compression_level", str(args.h5_compression_level),
        "--h5_chunk_rows", str(args.h5_chunk_rows),
        "--h5_chunk_channels", str(args.h5_chunk_channels),
        "--h5_store_row_order" if args.h5_store_row_order else "--no-h5_store_row_order",
    ]
    if args.compress_output:
        cmd.append("--compress_output")
    return cmd


def delete_chunk_artifacts(args, chunk) -> None:
    """Delete per-chunk high-res embeddings and processed test pickles."""
    project = str(args.project_path)
    deleted = 0

    if not args.no_delete_embeddings:
        for ps in args.patch_sizes:
            emb_dir = os.path.join(
                project, args.interim_path, args.patches_dir_name, f"emb_{ps}"
            )
            for sample in chunk:
                for fname in (f"{sample}.npz", f"{sample}_raw.npz"):
                    path = os.path.join(emb_dir, fname)
                    if os.path.exists(path):
                        os.remove(path)
                        deleted += 1
                        print(f"    [cleanup] removed {path}")

    if not args.no_delete_pkls:
        pkl_dir = os.path.join(project, args.pred_path, "processed_pkl_test_highres")
        for sample in chunk:
            path = os.path.join(pkl_dir, f"{sample}.pkl")
            if os.path.exists(path):
                os.remove(path)
                deleted += 1
                print(f"    [cleanup] removed {path}")

    if args.delete_histosweep_masks:
        mask_dir = os.path.join(
            project, args.interim_path, args.patches_dir_name, "wsis_rescaled"
        )
        for sample in chunk:
            path = os.path.join(
                mask_dir, f"{sample}_{args.histosweep_mask_suffix}_mask_rescaled.tif"
            )
            if os.path.exists(path):
                os.remove(path)
                deleted += 1
                print(f"    [cleanup] removed {path}")

    if deleted == 0:
        print("    [cleanup] nothing to delete")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Resolve the sample sheet relative to --project_path, matching how the
    # downstream extract / HistoSweep / predict scripts resolve their inputs
    # (each does Path(project_path) / sample_sheet).
    sheet = Path(args.project_path) / args.sample_sheet
    if not sheet.is_file():
        raise FileNotFoundError(f"sample sheet not found: {sheet}")
    meta = pd.read_csv(sheet, header=0)
    if not args.new_format_sample_sheet:
        ## Convert wsis_sample_sheet_filtered.csv to Inference_samples.csv
        if not args.chemo_format:
            dataset_test = list(meta['File Name'].values)
        else:
            dataset_test = list(meta['Filename'].values)
        dataset_test = [x.split('.')[0] for x in dataset_test]
        meta_new = pd.DataFrame({'Sample_name': dataset_test})
        meta_new.to_csv(f"{args.project_path}/{args.interim_path}/Inference_samples_tmp.csv", index=False)
        meta = meta_new
        args.sample_name_col=0
        args.sample_sheet=f"{args.interim_path}/Inference_samples_tmp.csv"
    if "Sample_name" not in meta.columns:
        raise ValueError(
            f"{sheet} must contain a 'Sample_name' column; got {list(meta.columns)}"
        )
    if not (0 <= args.sample_name_col < meta.shape[1]):
        raise ValueError(
            f"--sample_name_col {args.sample_name_col} is outside the sheet columns"
        )
    if meta.columns[args.sample_name_col] != "Sample_name":
        print(
            f"[warn] column {args.sample_name_col} is "
            f"'{meta.columns[args.sample_name_col]}' (not 'Sample_name'); make sure "
            "extract_features uses the right --sample_name_col.",
            flush=True,
        )

    samples = list(meta["Sample_name"].astype(str))
    chunk_size = max(1, args.chunk_size)
    chunks = [samples[i : i + chunk_size] for i in range(0, len(samples), chunk_size)]
    start = max(0, args.chunk_start)
    end = len(chunks) if args.chunk_end is None or args.chunk_end < 0 else min(args.chunk_end, len(chunks))
    if start >= len(chunks):
        print("[warn] --chunk-start is beyond the number of chunks; nothing to do.")
        return 0

    if args.tmp_dir:
        tmp_dir = Path(args.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        auto_clean = False
    else:
        # Keep per-chunk scratch under --project_path so all paths stay within
        # the project (mirrors how the downstream steps resolve their inputs).
        tmp_dir = Path(
            tempfile.mkdtemp(prefix="chunk_sheets_", dir=str(Path(args.project_path)))
        )
        auto_clean = True

    print(f"Sample sheet      : {sheet}")
    print(f"Samples           : {len(samples)}")
    print(f"Chunk size        : {chunk_size} -> {len(chunks)} chunk(s)")
    print(f"Processing chunks : {start}-{end - 1} (0-based)")
    if args.dry_run:
        print("DRY RUN: commands are printed but not executed.")

    failed = []
    try:
        for ci in range(start, end):
            chunk = chunks[ci]
            chunk_csv = tmp_dir / f"chunk_{ci:03d}.csv"
            meta[meta["Sample_name"].astype(str).isin(chunk)].to_csv(chunk_csv, index=False)
            # Pass the chunk sheet to each step as a path relative to
            # --project_path, matching how the steps resolve --sample_sheet
            # (they join it under -p; predict_highres_TCGA_optimized uses plain
            # f"{main_dir}/{sheet}", so an absolute path would be mangled).
            chunk_sheet_arg = os.path.relpath(chunk_csv, start=args.project_path)
            print(f"\n########## Chunk {ci + 1}/{len(chunks)}  ({len(chunk)} samples) ##########")
            for s in chunk:
                print(f"    {s}")

            ok = (
                _run_step("extract_features_highres_optimized", _extract_cmd(args, chunk_sheet_arg), args.dry_run)
                and _run_step("run_HistoSweep", _histosweep_cmd(args, chunk_sheet_arg), args.dry_run)
                and _run_step("predict_highres_TCGA_optimized", _predict_cmd(args, chunk_sheet_arg), args.dry_run)
            )

            if ok:
                print("\n[cleanup] deleting this chunk's artifacts", flush=True)
                delete_chunk_artifacts(args, chunk)
            else:
                print("[error] chunk did not complete; its artifacts were left in place", flush=True)
                failed.append((ci + 1, chunk))
                if not args.continue_on_error:
                    print("[abort] stopping (use --continue-on-error to keep going).", flush=True)
                    break
    finally:
        if auto_clean:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n===== Summary =====")
    if failed:
        print(f"  {len(failed)} chunk(s) failed:")
        for ci, chunk in failed:
            print(f"    chunk {ci}: {chunk}")
        return 1
    print("  all chunks completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
