#!/usr/bin/env python3
"""Accelerated high-resolution feature extraction with UNI2-h.

The default CUDA path prefilters empty source patches before transforms,
overlap-adds UNI global/local features directly on the regular 14-pixel GPU
lattice, streams completed sparse rows to float16 scratch storage, and smooths
bounded row/channel windows directly into float16 output.  A compact-transfer
CPU accumulator remains available for non-aligned scales and explicit fallback.
The saved ``emb``, ``pos``, ``flag``, and ``pad`` contract is unchanged.
"""

from __future__ import annotations

import argparse
import gc
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from highres_inference_pipeline import (
    ActivePatchBatcher,
    CpuAccumulatorPipeline,
    stage_valid_feature_batch,
)
from highres_gpu_strip_pipeline import (
    RegularGridStripAccumulator,
    save_sparse_npz,
    smoothing_work_units,
    stream_smooth_sparse_store,
)
from image_utils_optimized import (
    HighResFeatureExtractor,
    _SplitPatchAccumulator,
    get_UNI_model,
    smooth_embedding_halves,
)

UNI_TOKEN_SIZE = 14
UNI_EMBED_DIM = 1536


def _choose_extractor(
    active_batch_size: int,
    compiled_batch_size: int,
    eager_extractor,
    compiled_extractor,
):
    """Use the fixed-shape compiled graph only for a complete active batch."""
    if (
        active_batch_size == compiled_batch_size
        and compiled_extractor is not eager_extractor
    ):
        return compiled_extractor
    return eager_extractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract multiscale high-resolution UNI features from rescaled H&E images."
        )
    )
    parser.add_argument("--project_path", "-p", type=str, default="./")
    parser.add_argument("--code_path", "-c", type=str, default="./AURORA/code/")
    parser.add_argument("--catch_path", type=str, default="./CATCH/")
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    parser.add_argument("--output_path", "-o", type=str, default="AURORA_interim")
    parser.add_argument(
        "--patches_dir_name",
        type=str,
        default="UNI_multiscale_patches",
        help=(
            "Name of the subdirectory under --output_path that stores the "
            "multiscale embeddings (emb_<ps>/), rescaled WSIs (wsis_rescaled/), "
            "and HistoSweep masks (default: UNI_multiscale_patches)."
        ),
    )
    parser.add_argument(
        "--patch_sizes",
        nargs="+",
        type=int,
        default=[56, 14],
        help="Output grid sizes in pixels.",
    )
    parser.add_argument(
        "--metadata_path",
        "-m",
        type=str,
        default="Sample_metadata.csv",
    )
    parser.add_argument("--sample_name_col", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument(
        "--filter-prop",
        type=float,
        default=0.2,
        help=(
            "Minimum tissue fraction inside a 14px token (and inside a source "
            "patch) for it to be kept. Passed as filter_prop to "
            "PrefilteredHighResPatchDataset/PatchDataset to prefilter empty "
            "tokens and patches."
        ),
    )
    parser.add_argument("--UNI_local_path", type=str, default=None)
    parser.add_argument(
        "--CATCH_config_templet_path",
        type=str,
        default="CATCH_config_templet.yaml",
    )
    parser.add_argument("--use-CATCH", dest="use_CATCH", action="store_true")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        choices=[
            "max-autotune",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
            "default",
            "off",
        ],
        help=(
            "torch.compile mode for full, fixed-size active batches. Only the "
            "final active tail runs eagerly because of shape."
        ),
    )

    parser.add_argument(
        "--aggregation-mode",
        choices=["auto", "gpu-strip", "cpu"],
        default="auto",
        help=(
            "Aggregation backend. 'auto' uses regular-grid GPU strip accumulation "
            "for 14-pixel-aligned outputs on CUDA and otherwise uses the CPU path."
        ),
    )
    parser.add_argument(
        "--strip-writer-queue-depth",
        "--streaming-queue-depth",
        dest="strip_writer_queue_depth",
        type=int,
        default=2,
        help="Maximum completed strip payloads waiting for the sparse row writer.",
    )
    parser.add_argument(
        "--strip-contribution-chunk",
        "--gpu-accumulation-token-chunk",
        dest="strip_contribution_chunk",
        type=int,
        default=8192,
        help="Valid token contributions processed per GPU index_add_ chunk.",
    )
    parser.add_argument(
        "--strip-max-row-span-tokens",
        type=int,
        default=0,
        help=(
            "Optional maximum fine-grid source-row span inside one model batch. "
            "The default 0 keeps active batches full; positive values can trade "
            "more short batches for tighter memory bounds on unusually sparse images."
        ),
    )
    parser.add_argument(
        "--strip-rows",
        "--gpu-strip-rows",
        dest="strip_rows",
        type=int,
        default=None,
        help=(
            "Fine-grid rows per GPU accumulation strip. The default is an aligned "
            "value derived from the requested output scales."
        ),
    )
    parser.add_argument(
        "--strip-temp-dir",
        type=str,
        default=None,
        help=(
            "Scratch directory for sparse row stores and float16 smoothing memmaps. "
            "Defaults to a temporary directory under the output tree."
        ),
    )
    parser.add_argument(
        "--smoothing-row-chunk",
        "--streaming-row-chunk",
        dest="smoothing_row_chunk",
        type=int,
        default=128,
        help=(
            "Output grid rows smoothed at once, excluding the required halo. "
            "The 32 GiB CUDA path defaults to 128 to reduce repeated halos."
        ),
    )
    parser.add_argument(
        "--smoothing-workers",
        type=int,
        default=1,
        help=(
            "Independent smoothing channel chunks processed concurrently. "
            "Values of 2-4 can accelerate CPU smoothing, but each worker owns "
            "one dense float32 buffer; the default 1 preserves minimum memory."
        ),
    )
    parser.add_argument(
        "--smoothing-backend",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help=(
            "Smoothing backend for GPU-strip aggregation. 'auto' uses CUDA when "
            "available; 'cpu' preserves the current OpenCV implementation."
        ),
    )

    parser.add_argument(
        "--smoothing-column-tile",
        type=int,
        default=4096,
        help=(
            "Maximum occupied-column span per CUDA smoothing tile. Tiles also "
            "split at large empty gaps; 1024-4096 is the useful range."
        ),
    )
    parser.add_argument(
        "--smoothing-gpu-algorithm",
        choices=["auto", "prefix", "separable", "pool"],
        default="auto",
        help=(
            "CUDA box-filter primitive. 'auto' uses prefix sums for 16x16 and "
            "native pooling for small kernels."
        ),
    )
    parser.add_argument(
        "--smoothing-diagnostics",
        action="store_true",
        help=(
            "Print the loaded smoothing module, grid density, tile reduction, "
            "per-half time, and final NPZ write time."
        ),
    )

    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save <image>_raw.npz. Disabled by default to reduce I/O.",
    )
    parser.add_argument(
        "--empty-cache-between-images",
        "--clear-cache-per-image",
        dest="empty_cache_between_images",
        action="store_true",
        help=(
            "Call gc.collect() and torch.cuda.empty_cache() after each image. "
            "Normally slower; enable only for demonstrated fragmentation."
        ),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="DataLoader batches prefetched per worker.",
    )
    parser.add_argument(
        "--cpu-merge-queue-depth",
        "--cpu-pipeline-depth",
        "--pipeline-depth",
        dest="cpu_merge_queue_depth",
        type=int,
        default=2,
        help=(
            "Maximum compact device-to-host payloads waiting for the ordered "
            "CPU accumulation worker."
        ),
    )
    parser.add_argument(
        "--smoothing-channel-chunk",
        type=int,
        default=1536,
        help=(
            "Number of embedding channels smoothed at once. The 32 GiB CUDA "
            "path defaults to one complete embedding half (1536 channels)."
        ),
    )
    parser.add_argument(
        "--accumulator-chunk-budget",
        type=int,
        default=100_000,
        help="Approximate number of unique buffered grid entries before compaction.",
    )
    parser.add_argument(
        "--accumulator-channel-chunk",
        type=int,
        default=64,
        help="Number of channels reduced at once during CPU accumulation.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.UNI_local_path is None:
        raise ValueError("--UNI_local_path must be provided.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative.")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive.")
    if args.cpu_merge_queue_depth <= 0:
        raise ValueError("--cpu-merge-queue-depth must be positive.")
    if args.smoothing_channel_chunk <= 0:
        raise ValueError("--smoothing-channel-chunk must be positive.")
    if args.accumulator_chunk_budget <= 0:
        raise ValueError("--accumulator-chunk-budget must be positive.")
    if args.accumulator_channel_chunk <= 0:
        raise ValueError("--accumulator-channel-chunk must be positive.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if not (0.0 < args.filter_prop <= 1.0):
        raise ValueError("--filter-prop must be in (0, 1].")
    if args.strip_writer_queue_depth <= 0:
        raise ValueError("--strip-writer-queue-depth must be positive.")
    if args.strip_contribution_chunk <= 0:
        raise ValueError("--strip-contribution-chunk must be positive.")
    if args.strip_max_row_span_tokens < 0:
        raise ValueError("--strip-max-row-span-tokens cannot be negative.")
    if args.strip_rows is not None and args.strip_rows <= 0:
        raise ValueError("--strip-rows must be positive when provided.")
    if args.smoothing_row_chunk <= 0:
        raise ValueError("--smoothing-row-chunk must be positive.")
    if args.smoothing_workers <= 0:
        raise ValueError("--smoothing-workers must be positive.")
    if args.smoothing_column_tile <= 0:
        raise ValueError("--smoothing-column-tile must be positive.")
    if not args.patch_sizes or any(size <= 0 for size in args.patch_sizes):
        raise ValueError("--patch_sizes must contain positive integers.")


def _resolve_aggregation_mode(
    args: argparse.Namespace,
    device: torch.device,
    patch_sizes: Sequence[int],
) -> str:
    """Resolve the regular-grid strip backend without silently changing semantics."""

    mode = str(getattr(args, "aggregation_mode", "auto"))
    if mode not in {"auto", "gpu-strip", "cpu"}:
        raise ValueError(f"Unknown aggregation mode: {mode}")

    aligned_scales = bool(patch_sizes) and all(
        int(size) > 0 and int(size) % UNI_TOKEN_SIZE == 0
        for size in patch_sizes
    )
    aligned_stride = int(args.stride) % UNI_TOKEN_SIZE == 0

    if mode == "gpu-strip":
        if not aligned_scales:
            raise ValueError(
                "GPU strip aggregation requires output patch sizes that are "
                "multiples of 14."
            )
        if not aligned_stride:
            raise ValueError(
                "GPU strip aggregation requires --stride to be a multiple of 14."
            )
        return "gpu-strip"
    if mode == "cpu":
        return "cpu"
    if device.type == "cuda" and aligned_scales and aligned_stride:
        return "gpu-strip"
    return "cpu"


def filter_empty_patches(
    patches: torch.Tensor,
    positions: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    keep_flag: torch.Tensor,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    int,
]:
    """Compatibility helper for callers that still filter one batch at a time."""
    if keep_flag.ndim < 2:
        raise ValueError(f"keep_flag must be batched, got shape {keep_flag.shape}")
    if patches.shape[0] != keep_flag.shape[0]:
        raise ValueError("patches and keep_flag must have the same batch size")
    if len(positions) != 2:
        raise ValueError("positions must contain row and column tensors")

    active = keep_flag.bool().reshape(keep_flag.shape[0], -1).any(dim=1)
    removed = int((~active).sum().item())
    if removed == 0:
        return patches, (positions[0], positions[1]), keep_flag, 0
    return (
        patches[active],
        (positions[0][active], positions[1][active]),
        keep_flag[active],
        removed,
    )


def build_valid_contributions(
    positions: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    keep_flag: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    token_size: int = UNI_TOKEN_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map valid token flags to patch indices, token indices, and pixel positions."""
    if keep_flag.ndim != 3:
        raise ValueError(
            f"Expected keep_flag with shape (B, H, W), got {tuple(keep_flag.shape)}"
        )
    if keep_flag.shape[1:] != (grid_h, grid_w):
        raise ValueError(
            f"keep_flag spatial shape {tuple(keep_flag.shape[1:])} does not match "
            f"the model grid {(grid_h, grid_w)}"
        )
    if token_size <= 0:
        raise ValueError("token_size must be positive")
    if len(positions) != 2:
        raise ValueError("positions must contain row and column tensors")

    valid_mask = keep_flag.bool().reshape(keep_flag.shape[0], -1).cpu().numpy()
    batch_indices, token_indices = np.nonzero(valid_mask)
    batch_indices = batch_indices.astype(np.int64, copy=False)
    token_indices = token_indices.astype(np.int64, copy=False)
    if batch_indices.size == 0:
        return (
            batch_indices,
            token_indices,
            np.empty((0, 2), dtype=np.int32),
        )

    token_rows = token_indices // grid_w
    token_cols = token_indices % grid_w
    patch_rows = positions[0].cpu().numpy().astype(np.int64, copy=False)
    patch_cols = positions[1].cpu().numpy().astype(np.int64, copy=False)

    block_positions = np.empty((batch_indices.size, 2), dtype=np.int32)
    block_positions[:, 0] = (
        patch_rows[batch_indices] + token_size * token_rows
    ).astype(np.int32)
    block_positions[:, 1] = (
        patch_cols[batch_indices] + token_size * token_cols
    ).astype(np.int32)
    return batch_indices, token_indices, block_positions


def _valid_token_metadata(
    *,
    keep_flag: torch.Tensor,
    row_positions: torch.Tensor,
    col_positions: torch.Tensor,
    grid_h: int,
    grid_w: int,
    token_size: int = UNI_TOKEN_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid indices and fine-grid row/column positions."""
    batch_indices, token_indices, block_positions = build_valid_contributions(
        (row_positions, col_positions),
        keep_flag,
        grid_h=grid_h,
        grid_w=grid_w,
        token_size=token_size,
    )
    return batch_indices, token_indices, block_positions // token_size


def _image_stem(image_file: object) -> str:
    return Path(str(image_file)).stem


def _outputs_exist(save_dir: Path, patch_sizes: Sequence[int], image_name: str) -> bool:
    return all((save_dir / f"emb_{size}" / f"{image_name}.npz").exists() for size in patch_sizes)


def _load_image_files(args: argparse.Namespace):
    import pandas as pd

    metadata_path = Path(args.project_path) / args.metadata_path
    if not args.inference:
        metadata = pd.read_csv(
            metadata_path, index_col=args.sample_name_col, header=0
        )
        return list(metadata.index)

    metadata = pd.read_csv(metadata_path, header=0)
    if args.sample_name_col < 0 or args.sample_name_col >= metadata.shape[1]:
        raise IndexError(
            f"--sample_name_col {args.sample_name_col} is outside metadata with "
            f"{metadata.shape[1]} columns"
        )
    return metadata.iloc[:, args.sample_name_col].values


def _plot_embedding_map(
    embeddings: np.ndarray,
    positions: np.ndarray,
    flags: np.ndarray,
    *,
    image_shape: tuple[int, int],
    patch_size: int,
    output_path: Path,
    n_clusters: int = 20,
) -> None:
    import cv2
    import matplotlib.pyplot as plt
    from sklearn.cluster import MiniBatchKMeans

    valid = np.asarray(flags).astype(bool, copy=False)
    valid_count = int(valid.sum())
    if valid_count == 0:
        print(f"    Plot skipped: no valid embeddings for {output_path.name}")
        return

    cluster_count = min(n_clusters, valid_count)
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_count,
        batch_size=min(4096, max(cluster_count, valid_count)),
        n_init="auto",
        random_state=0,
    )
    valid_embeddings = embeddings[valid].astype(np.float32, copy=False)
    valid_labels = kmeans.fit_predict(valid_embeddings)
    valid_positions = positions[valid].astype(np.int64, copy=False)

    image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    palette = (
        plt.get_cmap("tab20")(np.arange(max(20, cluster_count)))[:, :3] * 255
    ).astype(np.uint8)

    for (row, col), label in zip(valid_positions, valid_labels, strict=True):
        row_start = max(0, int(row))
        row_end = min(image_shape[0], int(row) + patch_size)
        col_start = max(0, int(col))
        col_end = min(image_shape[1], int(col) + patch_size)
        image[row_start:row_end, col_start:col_end] = palette[label % len(palette)][::-1]

    image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
    cv2.imwrite(str(output_path), image)


def _run_catch(
    args: argparse.Namespace,
    image_files,
    save_dir: Path,
    patch_sizes: list[int],
) -> None:
    import matplotlib.pyplot as plt
    import pyvips
    import tifffile
    import yaml
    from sklearn.cluster import KMeans

    print("Start extracting features using CATCH")
    for image_file in image_files:
        image_name = _image_stem(image_file)
        if _outputs_exist(save_dir, patch_sizes, image_name):
            print(f"{image_name} already processed")
            continue

        image_rescaled = tifffile.imread(
            save_dir / "wsis_rescaled" / f"{image_name}_rescaled.tif"
        )
        mask_rescaled = tifffile.imread(
            save_dir / "wsis_rescaled" / f"{image_name}_mask_rescaled.tif"
        )
        print(f"Processing {image_name}")

        for patch_size in patch_sizes:
            print(f"    Patch size: {patch_size}")
            output_dir = save_dir / f"emb_{patch_size}"
            temp_dir = output_dir / "CATCH_tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            scale_factor = UNI_TOKEN_SIZE / patch_size
            image_use = pyvips.Image.new_from_array(image_rescaled).resize(scale_factor)
            mask_use = pyvips.Image.new_from_array(
                mask_rescaled.astype("float32")
            ).resize(scale_factor)
            image_use.write_to_file(str(temp_dir / f"{image_name}_rescaled.tif"))
            mask_use.write_to_file(str(temp_dir / f"{image_name}_mask_rescaled.tif"))

            with open(args.CATCH_config_templet_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            config["global"]["dataset"] = "CATCH_tmp"
            config["global"]["sample_list"] = [image_name]
            config["global"]["read_base_dir"] = str(output_dir)
            config["global"]["result_base_dir"] = str(output_dir)
            config["global"]["weight_dir"] = args.UNI_local_path
            config_path = temp_dir / "config.yaml"
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle)

            command = [
                "python",
                "-u",
                str(Path(args.code_path) / "run_CATCH.py"),
                "--config",
                str(config_path),
                "--catch_path",
                args.catch_path,
            ]
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                print(
                    f"Error in running CATCH for {image_name} at patch size {patch_size}"
                )
                continue

            source = temp_dir / f"{image_name}_emb14.npz"
            destination = output_dir / f"{image_name}.npz"
            shutil.move(str(source), str(destination))
            shutil.rmtree(temp_dir, ignore_errors=True)

            if args.plot:
                with np.load(destination) as catch_result:
                    embeddings = catch_result["emb"]
                    positions = catch_result["pos"]
                    flags = catch_result["flag"].astype(bool, copy=False)
                if flags.any():
                    clusters = min(5, int(flags.sum()))
                    labels = KMeans(n_clusters=clusters, n_init="auto").fit_predict(
                        embeddings[flags].astype(np.float32, copy=False)
                    )
                    plt.scatter(
                        x=positions[flags, 1],
                        y=-positions[flags, 0],
                        c=labels,
                        marker=",",
                        cmap="tab20",
                        s=0.1,
                    )
                    plt.savefig(
                        save_dir
                        / "check_figure"
                        / f"{image_name}_emb_{patch_size}_kmeans.png"
                    )
                    plt.close()

            if patch_size != UNI_TOKEN_SIZE:
                with np.load(destination) as catch_result:
                    pad = catch_result["pad"]
                    positions = catch_result["pos"]
                    corrected = (
                        positions - [pad[0] // 2, pad[1] // 2]
                    ) * (patch_size / UNI_TOKEN_SIZE)
                    corrected_result = {
                        "emb": catch_result["emb"],
                        "pos": corrected.astype(int),
                        "flag": catch_result["flag"],
                        "pad": [0, 0],
                        "raw_pos": positions,
                        "raw_pad": pad,
                    }
                np.savez(destination, **corrected_result)

        print(f"    Finished extracting features for {image_name}")


def _make_accumulator(args: argparse.Namespace) -> _SplitPatchAccumulator:
    return _SplitPatchAccumulator(
        global_dim=UNI_EMBED_DIM,
        spatial_dim=UNI_EMBED_DIM,
        chunk_budget=args.accumulator_chunk_budget,
        channel_chunk=args.accumulator_channel_chunk,
    )


def _validate_model_batch_outputs(
    *,
    global_features: torch.Tensor,
    spatial_features: torch.Tensor,
    batch_size: int,
    grid_h: int,
    grid_w: int,
) -> None:
    """Validate the extractor contract shared by both aggregation backends."""

    if global_features.ndim != 2 or global_features.shape[0] != batch_size:
        raise RuntimeError(
            "Extractor must return global features with shape (B, C); got "
            f"{tuple(global_features.shape)}"
        )
    if spatial_features.ndim != 3:
        raise RuntimeError(
            "Optimized HighResFeatureExtractor must return [B, N, C] spatial "
            f"tokens, got {tuple(spatial_features.shape)}"
        )
    if spatial_features.shape[0] != batch_size:
        raise RuntimeError(
            "Spatial feature batch size does not match the input batch: "
            f"{spatial_features.shape[0]} versus {batch_size}"
        )
    expected_tokens = int(grid_h * grid_w)
    if spatial_features.shape[1] != expected_tokens:
        raise RuntimeError(
            "Spatial token count does not match the extractor grid: "
            f"{spatial_features.shape[1]} versus {expected_tokens}"
        )


def _run_uni_gpu_strip_image(
    args: argparse.Namespace,
    *,
    dataset,
    extractor,
    compiled_extractor,
    device: torch.device,
    amp_dtype: torch.dtype,
    image_rescaled: np.ndarray,
    save_dir: Path,
    image_name: str,
    patch_sizes: Sequence[int],
) -> None:
    """Extract one image with regular-grid overlap-add and streamed output."""

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    max_row_span = int(getattr(args, "strip_max_row_span_tokens", 0))
    loader_kwargs = {
        "dataset": dataset,
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
    }
    if max_row_span > 0:
        from datasets import RowSpanBatchSampler

        loader_kwargs["batch_sampler"] = RowSpanBatchSampler(
            dataset,
            batch_size=int(args.batch_size),
            max_row_span_tokens=max_row_span,
        )
    else:
        loader_kwargs.update(
            shuffle=False,
            batch_size=int(args.batch_size),
        )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    dataloader = DataLoader(**loader_kwargs)

    scratch_parent_value = getattr(args, "strip_temp_dir", None)
    scratch_parent = (
        Path(scratch_parent_value)
        if scratch_parent_value
        else save_dir / ".strip_scratch"
    )
    scratch_parent.mkdir(parents=True, exist_ok=True)

    pad = np.array([dataset.row_pad, dataset.col_pad], dtype=np.int32)
    copy_stream = (
        torch.cuda.Stream(device=device) if device.type == "cuda" else None
    )
    compiled_full_batches = 0
    eager_full_batches = 0
    eager_short_batches = 0

    with tempfile.TemporaryDirectory(
        prefix=f"{image_name}_",
        dir=str(scratch_parent),
    ) as scratch_dir:
        strip_accumulator = RegularGridStripAccumulator(
            output_scales=patch_sizes,
            fine_grid_rows=int(dataset.fine_grid_shape[0]),
            fine_grid_cols=int(dataset.fine_grid_shape[1]),
            fine_contribution_counts=dataset.fine_contribution_counts,
            global_dim=UNI_EMBED_DIM,
            spatial_dim=UNI_EMBED_DIM,
            token_size=UNI_TOKEN_SIZE,
            token_grid_h=int(extractor.grid_h),
            token_grid_w=int(extractor.grid_w),
            device=device,
            temp_dir=scratch_dir,
            writer_queue_depth=int(
                getattr(args, "strip_writer_queue_depth", 2)
            ),
            strip_rows=getattr(args, "strip_rows", None),
            contribution_chunk=int(
                getattr(args, "strip_contribution_chunk", 8192)
            ),
            copy_stream=copy_stream,
        )

        progress_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        )
        progress = tqdm(
            total=len(dataset),
            desc="    Active patches",
            unit="patch",
            bar_format=progress_format,
        )
        try:
            with strip_accumulator:
                with torch.inference_mode():
                    for patches, positions, keep_flag in dataloader:
                        active_batch_size = int(patches.shape[0])
                        active_extractor = _choose_extractor(
                            active_batch_size,
                            int(args.batch_size),
                            extractor,
                            compiled_extractor,
                        )
                        used_compiled = (
                            active_extractor is compiled_extractor
                            and compiled_extractor is not extractor
                        )
                        if used_compiled:
                            compiled_full_batches += 1
                        elif active_batch_size == int(args.batch_size):
                            eager_full_batches += 1
                        else:
                            eager_short_batches += 1

                        patches_device = patches.to(
                            device,
                            non_blocking=device.type == "cuda",
                        )
                        with torch.autocast(
                            device_type=device.type,
                            dtype=amp_dtype,
                            enabled=device.type == "cuda",
                        ):
                            global_features, spatial_features = active_extractor(
                                patches_device
                            )

                        _validate_model_batch_outputs(
                            global_features=global_features,
                            spatial_features=spatial_features,
                            batch_size=active_batch_size,
                            grid_h=extractor.grid_h,
                            grid_w=extractor.grid_w,
                        )
                        strip_accumulator.add_batch(
                            global_features=global_features,
                            spatial_features=spatial_features,
                            positions=positions,
                            keep_flag=keep_flag,
                        )
                        progress.update(active_batch_size)
                        del patches_device, global_features, spatial_features
                stores = strip_accumulator.stores
        finally:
            progress.close()

        source_total = int(
            getattr(dataset, "source_total_patches", len(dataset))
        )
        active_total = int(
            getattr(dataset, "active_patch_count", len(dataset))
        )
        skipped_total = int(
            getattr(dataset, "skipped_patch_count", source_total - active_total)
        )
        print(
            "    Inference summary: "
            f"{source_total} source patches indexed, "
            f"{strip_accumulator.processed_patches} active patches inferred, "
            f"{skipped_total} empty patches prefiltered before transforms, "
            f"{compiled_full_batches} compiled full batches, "
            f"{eager_full_batches} eager full batches, "
            f"{eager_short_batches} eager short batches, "
            f"{strip_accumulator.processed_contributions} valid token contributions, "
            f"{strip_accumulator.writer.processed_chunks} streamed row chunks."
        )

        for patch_size in patch_sizes:
            patch_size = int(patch_size)
            print(f"    Streaming results for patch size: {patch_size}")
            output_dir = save_dir / f"emb_{patch_size}"
            output_dir.mkdir(parents=True, exist_ok=True)
            store = stores[patch_size]
            raw_embeddings = store.open_features()
            print(f"    Unique occupied positions: {store.num_features}")

            plot_positions = None
            plot_flags = None
            if args.plot:
                plot_positions = store.output_positions()
                plot_flags = np.ones((store.num_features,), dtype=bool)
                _plot_embedding_map(
                    raw_embeddings,
                    plot_positions,
                    plot_flags,
                    image_shape=image_rescaled.shape[:2],
                    patch_size=patch_size,
                    output_path=(
                        save_dir
                        / "check_figure"
                        / f"{image_name}_emb_{patch_size}_kmeans.png"
                    ),
                )

            if args.save_raw:
                save_sparse_npz(
                    store,
                    output_dir / f"{image_name}_raw.npz",
                    embeddings=raw_embeddings,
                    pad=pad,
                )

            if patch_size == UNI_TOKEN_SIZE:
                first_size, second_size = 16, 4
            else:
                first_size, second_size = 4, 2

            smoothing_path = (
                Path(scratch_dir) / f"{image_name}_{patch_size}_smoothed.f16"
            )
            work_total = smoothing_work_units(
                store,
                global_dim=UNI_EMBED_DIM,
            )
            smoothing_started = time.perf_counter()
            with tqdm(
                total=work_total,
                desc=f"    Smoothing {patch_size}px",
                unit="value",
                unit_scale=True,
                bar_format=progress_format,
            ) as smoothing_progress:
                smoothed_embeddings = stream_smooth_sparse_store(
                    store,
                    output_path=smoothing_path,
                    global_dim=UNI_EMBED_DIM,
                    first_size=first_size,
                    second_size=second_size,
                    channel_chunk=int(args.smoothing_channel_chunk),
                    row_chunk=int(getattr(args, "smoothing_row_chunk", 32)),
                    workers=int(getattr(args, "smoothing_workers", 1)),
                    backend=(
                        "gpu"
                        if args.smoothing_backend == "gpu"
                        or (
                            args.smoothing_backend == "auto"
                            and device.type == "cuda"
                        )
                        else "cpu"
                    ),
                    device=device,
                    column_tile=int(args.smoothing_column_tile),
                    gpu_algorithm=str(args.smoothing_gpu_algorithm),
                    diagnostics=bool(args.smoothing_diagnostics),
                    progress_callback=smoothing_progress.update,
                )
            smoothing_elapsed = time.perf_counter() - smoothing_started
            print(f"    Smoothing compute time: {smoothing_elapsed:.1f}s")

            save_started = time.perf_counter()
            save_sparse_npz(
                store,
                output_dir / f"{image_name}.npz",
                embeddings=smoothed_embeddings,
                pad=pad,
            )
            save_elapsed = time.perf_counter() - save_started
            print(f"    Final NPZ write time: {save_elapsed:.1f}s")

            if args.plot:
                _plot_embedding_map(
                    smoothed_embeddings,
                    plot_positions,
                    plot_flags,
                    image_shape=image_rescaled.shape[:2],
                    patch_size=patch_size,
                    output_path=(
                        save_dir
                        / "check_figure"
                        / f"{image_name}_smoothed_emb_{patch_size}_kmeans.png"
                    ),
                )
            del raw_embeddings, smoothed_embeddings
            if plot_positions is not None:
                del plot_positions, plot_flags

def _run_uni(
    args: argparse.Namespace,
    image_files,
    save_dir: Path,
    patch_sizes: list[int],
) -> None:
    import cv2
    import tifffile
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # Keep project-local dataset imports lazy so utility imports remain light.
    from datasets import PatchDataset

    print("Start extracting features using UNI")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if getattr(args, "smoothing_diagnostics", False):
        import highres_gpu_strip_pipeline as smoothing_module

        print(f"Smoothing module: {Path(smoothing_module.__file__).resolve()}")

    model, transform = get_UNI_model(args.UNI_local_path, "UNI2-h", from_local=True)
    model.to(device).eval()
    model.requires_grad_(False)
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(False)

    model_dim = int(getattr(model, "num_features", UNI_EMBED_DIM))
    if model_dim != UNI_EMBED_DIM:
        raise ValueError(
            f"This high-resolution smoothing/output path expects {UNI_EMBED_DIM} "
            f"channels, but the model reports {model_dim}."
        )

    extractor = HighResFeatureExtractor(model).eval()
    if extractor.grid_h * extractor.grid_w != 256:
        raise ValueError(
            "UNI2-h at 224x224 is expected to produce a 16x16 token grid; got "
            f"{extractor.grid_h}x{extractor.grid_w}."
        )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        amp_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        amp_dtype = torch.bfloat16

    compiled_extractor = extractor
    if args.compile_mode != "off" and device.type == "cuda":
        try:
            compiled_extractor = torch.compile(
                extractor,
                mode=args.compile_mode,
                dynamic=False,
            )
            print(
                f"torch.compile enabled (mode={args.compile_mode}, dynamic=False). "
                "Repacked full active batches use the compiled graph."
            )
        except Exception as exc:  # pragma: no cover - hardware/compiler dependent
            print(f"torch.compile failed ({exc!r}); continuing eagerly.")
            compiled_extractor = extractor
    elif device.type != "cuda":
        print("torch.compile skipped because CUDA is unavailable.")

    aggregation_mode = _resolve_aggregation_mode(args, device, patch_sizes)
    print(f"Aggregation mode: {aggregation_mode}")

    aligned_patch_sizes = [size for size in patch_sizes if size % UNI_TOKEN_SIZE == 0]
    irregular_patch_sizes = [size for size in patch_sizes if size % UNI_TOKEN_SIZE != 0]
    if irregular_patch_sizes:
        print(
            "Non-14-aligned scales will use separate direct accumulators: "
            f"{irregular_patch_sizes}"
        )

    for image_file in tqdm(image_files, desc="Images"):
        image_name = _image_stem(image_file)
        if _outputs_exist(save_dir, patch_sizes, image_name):
            print(f"{image_name} already processed")
            continue

        print(f"Processing {image_name}")
        image_rescaled = tifffile.imread(
            save_dir / "wsis_rescaled" / f"{image_name}_rescaled.tif"
        )
        mask_rescaled = tifffile.imread(
            save_dir / "wsis_rescaled" / f"{image_name}_mask_rescaled.tif"
        )

        # loc_rescaled and spot_radius_pixel are only used to draw ST spot
        # circles on the --plot check figure; neither is needed for feature
        # extraction, so skip loading them.
        loc_rescaled = None
        spot_radius_pixel = None

        if aggregation_mode == "gpu-strip":
            try:
                from datasets import PrefilteredHighResPatchDataset
            except ImportError:  # compatibility with the alternate public name
                from datasets import (
                    RegularGridPatchDataset as PrefilteredHighResPatchDataset,
                )
            dataset_class = PrefilteredHighResPatchDataset
        else:
            dataset_class = PatchDataset

        dataset = dataset_class(
            image_rescaled,
            mask_rescaled,
            transform,
            loc_rescaled,
            spot_radius_pixel,
            patch_size=224,
            stride=args.stride,
            filter_prop=args.filter_prop,
            high_res=True,
        )

        if args.plot:
            image_check = dataset.plot_mask()
            cv2.imwrite(
                str(save_dir / "check_figure" / f"{image_name}_mask_check.jpg"),
                image_check,
            )
            image_check = dataset.plot_patch_loc()
            cv2.imwrite(
                str(
                    save_dir
                    / "check_figure"
                    / f"{image_name}_highres_patch_check.jpg"
                ),
                image_check,
            )

        if aggregation_mode == "gpu-strip":
            _run_uni_gpu_strip_image(
                args,
                dataset=dataset,
                extractor=extractor,
                compiled_extractor=compiled_extractor,
                device=device,
                amp_dtype=amp_dtype,
                image_rescaled=image_rescaled,
                save_dir=save_dir,
                image_name=image_name,
                patch_sizes=patch_sizes,
            )
            if args.empty_cache_between_images:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            print(f"    Finished extracting features for {image_name}")
            continue

        loader_kwargs = {
            "dataset": dataset,
            "shuffle": False,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        }
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
        dataloader = DataLoader(**loader_kwargs)

        fine_state = _make_accumulator(args) if aligned_patch_sizes else None
        direct_states = {
            patch_size: _make_accumulator(args)
            for patch_size in irregular_patch_sizes
        }

        progress_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        patch_progress = tqdm(
            total=len(dataset),
            desc=f"    {image_name} patches",
            unit="patch",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
            smoothing=0.1,
            bar_format=progress_format,
        )
        progress_lock = threading.Lock()

        def update_patch_progress(amount: int) -> None:
            if amount <= 0:
                return
            with progress_lock:
                patch_progress.update(int(amount))

        active_batches = ActivePatchBatcher(
            dataloader,
            batch_size=args.batch_size,
            pin_memory=device.type == "cuda",
            on_skipped_patches=update_patch_progress,
        )
        copy_stream = (
            torch.cuda.Stream(device=device)
            if device.type == "cuda"
            else None
        )
        queue_depth = int(
            getattr(
                args,
                "cpu_merge_queue_depth",
                getattr(args, "cpu_pipeline_depth", 2),
            )
        )

        inferred_patches = 0
        contribution_count = 0
        compiled_full_batches = 0
        eager_tail_batches = 0

        try:
            with CpuAccumulatorPipeline(
                fine_state=fine_state,
                direct_states=direct_states,
                token_size=UNI_TOKEN_SIZE,
                max_pending=queue_depth,
                on_processed_patches=update_patch_progress,
            ) as merge_pipeline:
                with torch.inference_mode():
                    for active_batch in active_batches:
                        active_batch_size = active_batch.batch_size
                        batch_indices, token_indices, block_positions = (
                            build_valid_contributions(
                                active_batch.positions,
                                active_batch.keep_flag,
                                grid_h=extractor.grid_h,
                                grid_w=extractor.grid_w,
                                token_size=UNI_TOKEN_SIZE,
                            )
                        )
                        if batch_indices.size == 0:
                            continue

                        active_extractor = _choose_extractor(
                            active_batch_size,
                            args.batch_size,
                            extractor,
                            compiled_extractor,
                        )
                        used_compiled = (
                            active_extractor is compiled_extractor
                            and compiled_extractor is not extractor
                        )

                        patches_device = active_batch.patches.to(
                            device,
                            non_blocking=device.type == "cuda",
                        )
                        with torch.autocast(
                            device_type=device.type,
                            dtype=amp_dtype,
                            enabled=device.type == "cuda",
                        ):
                            global_features, spatial_features = active_extractor(
                                patches_device
                            )

                        if (
                            global_features.ndim != 2
                            or global_features.shape[0] != active_batch_size
                        ):
                            raise RuntimeError(
                                "Extractor must return global features with shape "
                                f"(B, C); got {tuple(global_features.shape)}"
                            )
                        if spatial_features.ndim != 3:
                            raise RuntimeError(
                                "Optimized HighResFeatureExtractor must return "
                                "[B, N, C] spatial tokens, got "
                                f"{tuple(spatial_features.shape)}"
                            )
                        if spatial_features.shape[0] != active_batch_size:
                            raise RuntimeError(
                                "Spatial feature batch size does not match the input "
                                f"batch: {spatial_features.shape[0]} versus "
                                f"{active_batch_size}"
                            )
                        expected_tokens = extractor.grid_h * extractor.grid_w
                        if spatial_features.shape[1] != expected_tokens:
                            raise RuntimeError(
                                "Spatial token count does not match the extractor grid: "
                                f"{spatial_features.shape[1]} versus {expected_tokens}"
                            )

                        payload = stage_valid_feature_batch(
                            global_features=global_features,
                            spatial_features=spatial_features,
                            batch_indices=batch_indices,
                            token_indices=token_indices,
                            pixel_positions=block_positions,
                            copy_stream=copy_stream,
                        )
                        merge_pipeline.submit(payload)

                        inferred_patches += active_batch_size
                        contribution_count += int(batch_indices.size)
                        compiled_full_batches += int(used_compiled)
                        eager_tail_batches += int(
                            active_batch_size < args.batch_size and not used_compiled
                        )

                        del patches_device, global_features, spatial_features

        finally:
            with progress_lock:
                patch_progress.close()

        skipped_empty_patches = active_batches.stats.skipped_patches
        pad = np.array([dataset.row_pad, dataset.col_pad], dtype=np.int32)
        stats = active_batches.stats
        eager_full_batches = stats.full_batches - compiled_full_batches
        print(
            "    Inference summary: "
            f"{stats.source_patches} source patches scanned, "
            f"{inferred_patches} active patches inferred, "
            f"{skipped_empty_patches} empty patches skipped, "
            f"{compiled_full_batches} compiled full batches, "
            f"{eager_full_batches} eager full batches, "
            f"{eager_tail_batches} eager tail batches, "
            f"{contribution_count} valid token contributions, "
            f"{merge_pipeline.processed_payloads} CPU merge payloads."
        )

        for patch_size in patch_sizes:
            print(f"    Summarizing results for patch size: {patch_size}")
            output_dir = save_dir / f"emb_{patch_size}"
            output_dir.mkdir(parents=True, exist_ok=True)

            if patch_size % UNI_TOKEN_SIZE == 0:
                if fine_state is None:  # defensive; cannot occur for this branch
                    raise RuntimeError("Missing fine-grid accumulator")
                embeddings, positions, flags = fine_state.finalize(
                    patch_size=patch_size,
                    base_patch_size=UNI_TOKEN_SIZE,
                )
            else:
                embeddings, positions, flags = direct_states[patch_size].finalize(
                    patch_size=patch_size,
                    base_patch_size=patch_size,
                )

            print(f"    Deduplicated patches: {embeddings.shape[0]} unique positions")
            if args.plot:
                _plot_embedding_map(
                    embeddings,
                    positions,
                    flags,
                    image_shape=image_rescaled.shape[:2],
                    patch_size=patch_size,
                    output_path=(
                        save_dir
                        / "check_figure"
                        / f"{image_name}_emb_{patch_size}_kmeans.png"
                    ),
                )

            if args.save_raw:
                np.savez(
                    output_dir / f"{image_name}_raw.npz",
                    emb=embeddings,
                    pos=positions,
                    flag=flags,
                    pad=pad,
                )

            print("    Smoothing")
            if patch_size == UNI_TOKEN_SIZE:
                first_size, second_size = 16, 4
            else:
                first_size, second_size = 4, 2

            smoothed_embeddings = smooth_embedding_halves(
                embeddings,
                positions,
                flags,
                position_unit=patch_size,
                first_size=first_size,
                second_size=second_size,
                channel_chunk=args.smoothing_channel_chunk,
                check_unique=False,
            ).astype(np.float16, copy=False)

            np.savez(
                output_dir / f"{image_name}.npz",
                emb=smoothed_embeddings,
                pos=positions,
                flag=flags,
                pad=pad,
            )

            if args.plot:
                _plot_embedding_map(
                    smoothed_embeddings,
                    positions,
                    flags,
                    image_shape=image_rescaled.shape[:2],
                    patch_size=patch_size,
                    output_path=(
                        save_dir
                        / "check_figure"
                        / f"{image_name}_smoothed_emb_{patch_size}_kmeans.png"
                    ),
                )

        if args.empty_cache_between_images:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"    Finished extracting features for {image_name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)

    main_dir = Path(args.project_path)
    save_dir = main_dir / args.output_path / args.patches_dir_name
    patch_sizes = list(dict.fromkeys(args.patch_sizes))

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "check_figure").mkdir(parents=True, exist_ok=True)
    image_files = _load_image_files(args)

    if args.use_CATCH:
        _run_catch(args, image_files, save_dir, patch_sizes)
    else:
        _run_uni(args, image_files, save_dir, patch_sizes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
