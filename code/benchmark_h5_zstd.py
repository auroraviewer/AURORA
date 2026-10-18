#!/usr/bin/env python3
"""Benchmark lossless Zstandard strategies on large uint8 HDF5 prediction matrices.

The script samples several windows from each requested dataset, so it does not load
an entire multi-gigabyte matrix into RAM. It compares:

  patch_major
      Existing C-order layout: patches x genes.
  gene_major
      Exact transpose in storage: genes x patches.
  spatial_gene_major
      Coordinate-sort patches, then store genes x patches.
  spatial_gene_delta
      Coordinate-sort, transpose, then store modulo-256 differences between
      adjacent patches for each gene. This is exactly reversible.
  spatial_gene_xor
      Coordinate-sort, transpose, then XOR adjacent patches for each gene.
      This is exactly reversible.

This benchmarks an external/whole-stream Zstandard codec. HDF5-internal Zstandard
will usually be close only when its chunk size/order resembles the tested stream.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

try:
    import zstandard as zstd  # type: ignore
except ImportError:  # The zstd CLI is used as a fallback.
    zstd = None

MIB = 1024 ** 2
GIB = 1024 ** 3


@dataclass
class CandidateResult:
    dataset: str
    layout: str
    level: int
    sample_raw_bytes: int
    sample_compressed_bytes: int
    entropy_bits_per_byte: float
    seconds: float
    full_dataset_raw_bytes: int

    @property
    def ratio(self) -> float:
        return self.sample_compressed_bytes / self.sample_raw_bytes

    @property
    def reduction_pct(self) -> float:
        return 100.0 * (1.0 - self.ratio)

    @property
    def projected_full_bytes(self) -> float:
        return self.ratio * self.full_dataset_raw_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Zstandard layouts/transforms on HDF5 uint8 matrices."
    )
    parser.add_argument("h5_path", type=Path, help="Input HDF5 file")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["exp_pred_14", "exp_pred_56"],
        help="Datasets to benchmark (default: exp_pred_14 exp_pred_56)",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=16384,
        help="Total sampled rows per dataset (default: 16384)",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=4,
        help="Number of evenly spaced sample windows (default: 4)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[3, 9, 19],
        help="Zstandard levels to test (default: 3 9 19)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "Zstandard worker threads. 0 is single-threaded and usually gives "
            "the cleanest ratio comparison; -1 uses all logical CPUs."
        ),
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help="Skip coordinate-sorted and predictive transforms",
    )
    return parser.parse_args()


def human_bytes(value: float) -> str:
    if value >= GIB:
        return f"{value / GIB:.3f} GiB"
    return f"{value / MIB:.2f} MiB"


def fixed_dataset_raw_bytes(hf: h5py.File) -> int:
    """Approximate raw payload for fixed-width datasets (object strings are tiny here)."""
    total = 0

    def visitor(_name: str, obj: object) -> None:
        nonlocal total
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind != "O":
            total += int(obj.size) * int(obj.dtype.itemsize)

    hf.visititems(visitor)
    return total


def make_sample_slices(n_rows: int, sample_rows: int, windows: int) -> List[slice]:
    if n_rows <= 0:
        return []
    sample_rows = min(max(1, sample_rows), n_rows)
    windows = min(max(1, windows), sample_rows)

    # Use equal-size windows. Any remainder is added to the final window.
    base = max(1, sample_rows // windows)
    starts = np.linspace(0, max(0, n_rows - base), num=windows, dtype=np.int64)

    slices: List[slice] = []
    used = 0
    seen: set[Tuple[int, int]] = set()
    for idx, start_value in enumerate(starts):
        remaining = sample_rows - used
        if remaining <= 0:
            break
        length = base if idx < windows - 1 else remaining
        length = min(length, remaining, n_rows)
        start = int(min(start_value, n_rows - length))
        end = start + length
        key = (start, end)
        if key not in seen:
            slices.append(slice(start, end))
            seen.add(key)
            used += length

    # Duplicate/overlapping starts can occur for very small arrays. Fill any shortfall
    # with a deterministic window from the beginning.
    if used < sample_rows:
        missing = sample_rows - used
        slices.append(slice(0, missing))

    return slices


def read_sample(ds: h5py.Dataset, slices: Sequence[slice]) -> np.ndarray:
    blocks = [np.asarray(ds[sl]) for sl in slices]
    if not blocks:
        return np.empty((0,) + ds.shape[1:], dtype=ds.dtype)
    return np.ascontiguousarray(np.concatenate(blocks, axis=0))


def byte_entropy(array: np.ndarray) -> float:
    """Marginal Shannon entropy of byte values, in bits per byte."""
    contiguous = np.ascontiguousarray(array)
    byte_view = contiguous.view(np.uint8).reshape(-1)
    counts = np.bincount(byte_view, minlength=256).astype(np.float64)
    nonzero = counts[counts > 0]
    probabilities = nonzero / nonzero.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def coordinate_dataset_name(dataset_name: str) -> Optional[str]:
    if dataset_name.startswith("exp_pred_"):
        return "patch_loc_" + dataset_name.removeprefix("exp_pred_")
    if dataset_name.startswith("prop_pred_"):
        return "patch_loc_" + dataset_name.removeprefix("prop_pred_")
    return None


def transformed_arrays(
    sample: np.ndarray,
    locations: Optional[np.ndarray],
    include_spatial: bool,
) -> Iterable[Tuple[str, np.ndarray]]:
    if sample.dtype != np.uint8 or sample.ndim != 2:
        raise ValueError(
            f"Expected a 2-D uint8 matrix, got shape={sample.shape}, dtype={sample.dtype}"
        )

    yield "patch_major", np.ascontiguousarray(sample)
    yield "gene_major", np.ascontiguousarray(sample.T)

    if not include_spatial or locations is None:
        return
    if locations.ndim != 2 or locations.shape[0] != sample.shape[0]:
        raise ValueError(
            "Coordinate sample must have the same number of rows as the prediction sample"
        )

    # Sort by the first coordinate, then the second. Coordinates remain aligned with
    # predictions; storing the sorted coordinates makes this transformation lossless.
    order = np.lexsort((locations[:, 1], locations[:, 0]))
    gene_major = np.ascontiguousarray(sample[order].T)
    yield "spatial_gene_major", gene_major

    # Exact modulo-256 first differences. The first value of every gene is stored
    # literally. Reconstruct with cumulative sum modulo 256.
    delta = np.empty_like(gene_major)
    delta[:, 0] = gene_major[:, 0]
    np.subtract(gene_major[:, 1:], gene_major[:, :-1], out=delta[:, 1:])
    yield "spatial_gene_delta", delta

    xor = np.empty_like(gene_major)
    xor[:, 0] = gene_major[:, 0]
    np.bitwise_xor(gene_major[:, 1:], gene_major[:, :-1], out=xor[:, 1:])
    yield "spatial_gene_xor", xor


def compress_size(array: np.ndarray, level: int, threads: int) -> Tuple[int, float]:
    contiguous = np.ascontiguousarray(array)
    source = memoryview(contiguous).cast("B")
    started = time.perf_counter()

    if zstd is not None:
        compressor = zstd.ZstdCompressor(
            level=level,
            threads=threads,
            write_checksum=False,
            write_content_size=True,
        )
        compressed = compressor.compress(source)
        elapsed = time.perf_counter() - started
        return len(compressed), elapsed

    executable = shutil.which("zstd")
    if executable is None:
        raise RuntimeError(
            "Neither the Python 'zstandard' package nor the zstd CLI is available. "
            "Install one of them, e.g. 'python -m pip install zstandard'."
        )

    command = [executable, "-q", "-c", f"-{level}"]
    if level >= 20:
        command.append("--ultra")
    if threads == -1:
        command.append("-T0")
    elif threads > 0:
        command.append(f"-T{threads}")
    # With no -T option, the CLI uses its single-threaded path.
    completed = subprocess.run(
        command,
        input=source,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"zstd CLI failed with exit code {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace')}"
        )
    elapsed = time.perf_counter() - started
    return len(completed.stdout), elapsed


def print_dataset_results(results: Sequence[CandidateResult]) -> None:
    print(
        f"{'layout':26s} {'level':>5s} {'entropy':>9s} {'ratio':>9s} "
        f"{'saved':>9s} {'sample zstd':>13s} {'time':>9s}"
    )
    print("-" * 94)
    for result in results:
        print(
            f"{result.layout:26s} {result.level:5d} "
            f"{result.entropy_bits_per_byte:8.3f}b "
            f"{result.ratio:8.3f}x "
            f"{result.reduction_pct:8.2f}% "
            f"{human_bytes(result.sample_compressed_bytes):>13s} "
            f"{result.seconds:8.2f}s"
        )


def main() -> int:
    args = parse_args()
    if not args.h5_path.is_file():
        print(f"Input file not found: {args.h5_path}", file=sys.stderr)
        return 2
    if args.sample_rows <= 0 or args.windows <= 0:
        print("--sample-rows and --windows must be positive", file=sys.stderr)
        return 2

    all_results: List[CandidateResult] = []

    with h5py.File(args.h5_path, "r") as hf:
        actual_file_bytes = os.path.getsize(args.h5_path)
        total_fixed_raw = fixed_dataset_raw_bytes(hf)
        selected_full_raw = 0

        print(f"File: {args.h5_path}")
        print(f"Actual HDF5 size:       {human_bytes(actual_file_bytes)}")
        print(f"Fixed-width raw bytes:  {human_bytes(total_fixed_raw)}")
        print()

        for dataset_name in args.datasets:
            if dataset_name not in hf:
                print(f"[warning] Missing dataset: {dataset_name}", file=sys.stderr)
                continue

            ds = hf[dataset_name]
            if not isinstance(ds, h5py.Dataset):
                print(f"[warning] Not a dataset: {dataset_name}", file=sys.stderr)
                continue
            if ds.ndim != 2 or ds.dtype != np.dtype("uint8"):
                print(
                    f"[warning] Skipping {dataset_name}: expected 2-D uint8, "
                    f"got shape={ds.shape}, dtype={ds.dtype}",
                    file=sys.stderr,
                )
                continue

            slices = make_sample_slices(ds.shape[0], args.sample_rows, args.windows)
            sample = read_sample(ds, slices)
            full_raw_bytes = int(ds.size) * int(ds.dtype.itemsize)
            selected_full_raw += full_raw_bytes

            loc_sample: Optional[np.ndarray] = None
            loc_name = coordinate_dataset_name(dataset_name)
            if not args.no_spatial and loc_name and loc_name in hf:
                loc_sample = read_sample(hf[loc_name], slices)
            elif not args.no_spatial:
                print(
                    f"[warning] Coordinates not found for {dataset_name}; "
                    "spatial candidates skipped.",
                    file=sys.stderr,
                )

            print(f"Dataset: {dataset_name}")
            print(f"  full shape:       {ds.shape}")
            print(f"  full raw size:    {human_bytes(full_raw_bytes)}")
            print(f"  sampled rows:     {sample.shape[0]:,}")
            print(f"  sample raw size:  {human_bytes(sample.nbytes)}")

            dataset_results: List[CandidateResult] = []
            for layout, transformed in transformed_arrays(
                sample, loc_sample, include_spatial=not args.no_spatial
            ):
                entropy = byte_entropy(transformed)
                for level in args.levels:
                    compressed_bytes, elapsed = compress_size(
                        transformed, level=level, threads=args.threads
                    )
                    result = CandidateResult(
                        dataset=dataset_name,
                        layout=layout,
                        level=level,
                        sample_raw_bytes=transformed.nbytes,
                        sample_compressed_bytes=compressed_bytes,
                        entropy_bits_per_byte=entropy,
                        seconds=elapsed,
                        full_dataset_raw_bytes=full_raw_bytes,
                    )
                    dataset_results.append(result)
                    all_results.append(result)

            print_dataset_results(dataset_results)
            print()

        if not all_results:
            print("No eligible datasets were benchmarked.", file=sys.stderr)
            return 1

        other_raw = max(0, total_fixed_raw - selected_full_raw)
        grouped: Dict[Tuple[str, int], List[CandidateResult]] = {}
        for result in all_results:
            grouped.setdefault((result.layout, result.level), []).append(result)

        print("Projected full-file payload from sampled ratios")
        print("(Adds all non-selected fixed-width datasets at their raw size.)")
        print(
            f"{'layout':26s} {'level':>5s} {'projected':>14s} "
            f"{'vs raw':>10s} {'vs current H5':>15s}"
        )
        print("-" * 79)

        projections: List[Tuple[float, str, int]] = []
        for (layout, level), results in grouped.items():
            # Only compare candidates represented for every selected dataset.
            projected = other_raw + sum(r.projected_full_bytes for r in results)
            ratio_vs_raw = projected / total_fixed_raw if total_fixed_raw else math.nan
            reduction_vs_current = (
                100.0 * (1.0 - projected / actual_file_bytes)
                if actual_file_bytes
                else math.nan
            )
            projections.append((projected, layout, level))
            print(
                f"{layout:26s} {level:5d} {human_bytes(projected):>14s} "
                f"{ratio_vs_raw:9.3f}x {reduction_vs_current:14.2f}%"
            )

        best_projected, best_layout, best_level = min(projections)
        print()
        print(
            "Best sampled candidate: "
            f"{best_layout}, zstd level {best_level}, "
            f"projected payload {human_bytes(best_projected)}"
        )
        print(
            "Treat the projection as directional: full-file ratio depends on HDF5 "
            "chunking, sample representativeness, and archive/filter overhead."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
