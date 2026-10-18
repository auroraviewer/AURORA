#!/usr/bin/env python3
"""Read AURORA high-resolution HDF5 files, including spatial-gene-delta v1.

For Zstandard-compressed HDF5 files, install ``hdf5plugin``. Importing it
registers the HDF5 filter; this module attempts that import automatically.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

try:  # Registers optional HDF5 filters when available.
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None


OPTIMIZED_H5_ENCODING = "spatial_gene_delta_v1"


def _attr_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_strings(dataset: h5py.Dataset) -> np.ndarray:
    try:
        return np.asarray(dataset.asstr()[...])
    except (AttributeError, TypeError):
        values = np.asarray(dataset[...])
        if values.dtype.kind == "S":
            return np.char.decode(values, "utf-8")
        return values.astype(str)


def _logical_shape(dataset: h5py.Dataset) -> tuple[int, int]:
    encoding = _attr_text(dataset.attrs.get("encoding", "none"))
    if encoding == OPTIMIZED_H5_ENCODING:
        shape = tuple(int(v) for v in dataset.attrs["logical_shape"])
        if len(shape) != 2:
            raise ValueError(f"Invalid logical_shape on {dataset.name}: {shape}")
        return shape
    if dataset.ndim != 2:
        raise ValueError(f"Prediction dataset {dataset.name} is not 2-D")
    return tuple(int(v) for v in dataset.shape)


def iter_prediction_blocks(
    h5_file: h5py.File,
    dataset_name: str,
    *,
    block_rows: int | None = None,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield decoded ``(start, end, rows_by_channels)`` blocks.

    For the optimized layout, blocks are in spatial row order. Use
    ``read_prediction_matrix(..., restore_original_order=True)`` when the exact
    pre-sort row order is required.
    """
    dataset = h5_file[dataset_name]
    encoding = _attr_text(dataset.attrs.get("encoding", "none"))
    n_rows, _ = _logical_shape(dataset)

    if encoding == OPTIMIZED_H5_ENCODING:
        encoded_block_rows = int(dataset.attrs["delta_block_rows"])
        if block_rows is not None and int(block_rows) != encoded_block_rows:
            raise ValueError(
                f"{dataset_name} must be decoded in its stored block size "
                f"({encoded_block_rows} rows), not {block_rows}"
            )
        for start in range(0, n_rows, encoded_block_rows):
            end = min(start + encoded_block_rows, n_rows)
            try:
                encoded = dataset[:, start:end]
            except OSError as exc:
                codec = _attr_text(h5_file.attrs.get("compression_codec", "unknown"))
                if codec == "zstd" and hdf5plugin is None:
                    raise RuntimeError(
                        "This file uses the HDF5 Zstandard filter. Install "
                        "hdf5plugin (`python -m pip install hdf5plugin`) before reading."
                    ) from exc
                raise
            decoded = np.add.accumulate(encoded, axis=1, dtype=np.uint8).T
            yield start, end, np.ascontiguousarray(decoded)
        return

    if encoding not in {"none", "", "None"}:
        raise ValueError(f"Unsupported encoding {encoding!r} on {dataset.name}")

    rows_per_block = int(block_rows or (dataset.chunks[0] if dataset.chunks else 16384))
    rows_per_block = max(1, rows_per_block)
    for start in range(0, n_rows, rows_per_block):
        end = min(start + rows_per_block, n_rows)
        yield start, end, np.asarray(dataset[start:end, :])


def read_prediction_matrix(
    h5_file: h5py.File,
    dataset_name: str,
    *,
    restore_original_order: bool = False,
) -> np.ndarray:
    """Decode a prediction dataset to its logical rows-by-channels matrix."""
    dataset = h5_file[dataset_name]
    n_rows, n_channels = _logical_shape(dataset)
    result = np.empty((n_rows, n_channels), dtype=dataset.dtype)
    for start, end, block in iter_prediction_blocks(h5_file, dataset_name):
        result[start:end, :] = block

    if not restore_original_order:
        return result

    order_name = _attr_text(dataset.attrs.get("row_order_dataset", ""))
    if not order_name:
        encoding = _attr_text(dataset.attrs.get("encoding", "none"))
        if encoding == OPTIMIZED_H5_ENCODING:
            raise ValueError(
                f"{dataset_name} is spatially sorted, but its row permutation was "
                "not stored. Read with restore_original_order=False."
            )
        return result

    stored_to_original = np.asarray(h5_file[order_name][...], dtype=np.int64)
    if stored_to_original.shape != (n_rows,):
        raise ValueError(
            f"{order_name} has shape {stored_to_original.shape}; expected {(n_rows,)}"
        )
    restored = np.empty_like(result)
    restored[stored_to_original, :] = result
    return restored


def read_patch_locations(
    h5_file: h5py.File,
    patch_size: int,
    *,
    restore_original_order: bool = False,
) -> np.ndarray:
    """Read row-aligned patch coordinates for one patch size."""
    name = f"patch_loc_{patch_size}"
    locations = np.asarray(h5_file[name][...])
    if not restore_original_order:
        return locations

    order_name = _attr_text(h5_file[name].attrs.get("row_order_dataset", ""))
    if not order_name:
        return locations
    stored_to_original = np.asarray(h5_file[order_name][...], dtype=np.int64)
    restored = np.empty_like(locations)
    restored[stored_to_original, :] = locations
    return restored


def available_patch_sizes(h5_file: h5py.File) -> list[int]:
    if "patch_sizes" in h5_file:
        return [int(v) for v in np.asarray(h5_file["patch_sizes"][...]).reshape(-1)]
    sizes = []
    for name in h5_file.keys():
        match = re.fullmatch(r"exp_pred_(\d+)", name)
        if match:
            sizes.append(int(match.group(1)))
    return sorted(sizes, reverse=True)


def load_highres_h5(
    path: str | Path,
    *,
    restore_original_order: bool = False,
    dequantize: bool = False,
) -> dict:
    """Load all prediction levels and metadata.

    ``dequantize=False`` returns exact saved uint8 matrices. With
    ``dequantize=True``, matrices are reconstructed as float32 using their
    per-channel maxima.
    """
    result: dict = {
        "exp_pred": {},
        "prop_pred": {},
        "exp_max": {},
        "prop_max": {},
        "pos": {},
    }
    with h5py.File(path, "r") as h5_file:
        result["genes"] = _read_strings(h5_file["genes"])
        result["cell_types"] = _read_strings(h5_file["cell_types"])
        result["resolutions"] = np.asarray(h5_file["resolutions"][...])
        result["patch_sizes"] = available_patch_sizes(h5_file)
        result["layout"] = _attr_text(
            h5_file.attrs.get("prediction_layout", "patch_major")
        )

        for patch_size in result["patch_sizes"]:
            exp = read_prediction_matrix(
                h5_file,
                f"exp_pred_{patch_size}",
                restore_original_order=restore_original_order,
            )
            prop = read_prediction_matrix(
                h5_file,
                f"prop_pred_{patch_size}",
                restore_original_order=restore_original_order,
            )
            exp_max = np.asarray(
                h5_file[f"exp_max_{patch_size}"][...], dtype=np.float32
            )
            prop_max = np.asarray(
                h5_file[f"prop_max_{patch_size}"][...], dtype=np.float32
            )

            if dequantize:
                exp = exp.astype(np.float32) * (exp_max[np.newaxis, :] / 255.0)
                prop = prop.astype(np.float32) * (prop_max[np.newaxis, :] / 255.0)

            result["exp_pred"][patch_size] = exp
            result["prop_pred"][patch_size] = prop
            result["exp_max"][patch_size] = exp_max
            result["prop_max"][patch_size] = prop_max
            result["pos"][patch_size] = read_patch_locations(
                h5_file,
                patch_size,
                restore_original_order=restore_original_order,
            )
    return result


def print_summary(path: str | Path) -> None:
    with h5py.File(path, "r") as h5_file:
        layout = _attr_text(h5_file.attrs.get("prediction_layout", "patch_major"))
        codec = _attr_text(h5_file.attrs.get("compression_codec", "unknown"))
        level = h5_file.attrs.get("compression_level", "unknown")
        print(f"File: {path}")
        print(f"Layout: {layout}; compression: {codec} level {level}")
        for patch_size in available_patch_sizes(h5_file):
            for prefix in ("exp_pred", "prop_pred"):
                dataset = h5_file[f"{prefix}_{patch_size}"]
                logical = _logical_shape(dataset)
                encoding = _attr_text(dataset.attrs.get("encoding", "none"))
                print(
                    f"  {dataset.name}: logical={logical}, stored={dataset.shape}, "
                    f"dtype={dataset.dtype}, encoding={encoding}, chunks={dataset.chunks}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", type=Path)
    parser.add_argument(
        "--extract",
        metavar="DATASET",
        help="Decode one prediction dataset, such as exp_pred_14.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .npy path used with --extract.",
    )
    parser.add_argument(
        "--restore-original-order",
        action="store_true",
        help="Undo spatial row sorting when the row permutation is present.",
    )
    args = parser.parse_args()

    if not args.extract:
        print_summary(args.h5_path)
        return
    if args.output is None:
        parser.error("--output is required with --extract")

    with h5py.File(args.h5_path, "r") as h5_file:
        matrix = read_prediction_matrix(
            h5_file,
            args.extract,
            restore_original_order=args.restore_original_order,
        )
    np.save(args.output, matrix)
    print(f"Saved {args.extract} with shape {matrix.shape} to {args.output}")


if __name__ == "__main__":
    main()
