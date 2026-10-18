"""Read AURORA high-resolution quantised HDF5 predictions.

Port of ``code/highres_h5_loader.py``.  It decodes both layouts written by
:mod:`aurora.h5out` (and by the original
``code/predict_highres_TCGA_optimized.py``):

``spatial_gene_delta``
    The prediction datasets are stored transposed as ``(channels, rows)`` with
    the rows in a spatial raster order and each row delta-coded against the
    previous one within a block of ``delta_block_rows`` rows, modulo 256.  The
    permutation back to the original row order is in ``row_order_<ps>``.
``patch_major``
    Plain ``(rows, channels)`` matrices in the original row order.

Use :func:`load_highres_h5` for the common case: it returns every patch size with
the metadata, optionally de-quantised and optionally restored to the original row
order.

Row order
---------

By default the matrices and coordinates come back in the **stored** (spatially
sorted) order, which is what the original loader does.  Pass
``restore_original_order=True`` to undo the sort; the two outputs are then
row-aligned with each other and with the input that was written.

Reading zstd
------------

Files written with the ``zstd`` codec need ``hdf5plugin``, which registers the
HDF5 filter with h5py.  It is imported on first use; when it is missing and the
file records ``compression_codec = 'zstd'``, reading raises a
:class:`~aurora.errors.ConfigError` naming the fix rather than a bare h5py error.

Dependencies
------------

``import aurora.h5in`` needs nothing installed: ``h5py``, ``numpy`` and
``hdf5plugin`` are all imported inside the functions that use them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

from .errors import ConfigError
if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import h5py

#: Value of the ``encoding`` attribute on a delta-coded prediction dataset.
OPTIMIZED_H5_ENCODING = "spatial_gene_delta_v1"

#: Fallback row block size for a dataset with no chunks of its own.
DEFAULT_ROWS_PER_BLOCK = 16384

_hdf5plugin_module: Optional[Any] = None
_hdf5plugin_checked = False


def _numpy() -> Any:
    from .deps import require

    return require("numpy", purpose="reading AURORA HDF5 predictions")


def _h5py() -> Any:
    from .deps import require

    return require("h5py", purpose="reading AURORA HDF5 predictions")


def _load_hdf5plugin() -> Optional[Any]:
    """Import ``hdf5plugin`` once, registering its HDF5 filters.

    Returns the module, or ``None`` when it is not installed.  Importing it is
    what makes h5py able to read a zstd-compressed dataset; the original loader
    did this at module import time.
    """
    global _hdf5plugin_module, _hdf5plugin_checked
    if not _hdf5plugin_checked:
        try:
            import hdf5plugin

            _hdf5plugin_module = hdf5plugin
        except ImportError:
            _hdf5plugin_module = None
        _hdf5plugin_checked = True
    return _hdf5plugin_module


def _zstd_unavailable_error(codec: str, detail: str = "") -> ConfigError:
    detail = f" ({detail})" if detail else ""
    return ConfigError(
        f"this file uses the HDF5 Zstandard filter (compression_codec={codec!r}) "
        f"but hdf5plugin is not installed{detail}",
        hint=(
            "Install it with:\n"
            "  python -m pip install hdf5plugin\n"
            "and import hdf5plugin before reading the file. Files written with "
            "'--h5-compression gzip' or '--h5-compression none' need no extra package."
        ),
    )


def _require_filter_available(h5_file: "h5py.File") -> None:
    """Fail early, and clearly, when a zstd file cannot be read.

    h5py only reports the missing filter when a dataset is actually read, which
    makes it look like corruption.  ``compression_codec`` in the root attributes
    says up front which filter is needed.
    """
    codec = _attr_text(h5_file.attrs.get("compression_codec", "unknown"))
    if codec == "zstd" and _load_hdf5plugin() is None:
        raise _zstd_unavailable_error(codec)


def open_h5(path: str | Path) -> Any:
    """Open an HDF5 file for reading, explaining a file that is not one.

    h5py reports an empty, truncated or non-HDF5 file as a bare
    ``OSError: Unable to synchronously open file (file signature not found)``,
    which reads like a bug rather than a bad input.  A prediction written as
    ``.npz.zst`` or an interrupted download both land here.
    """
    h5py = _h5py()
    # Register any available HDF5 filters (zstd) before opening.
    _load_hdf5plugin()
    try:
        return h5py.File(path, "r")
    except OSError as exc:
        raise ConfigError(
            f"{path} is not a readable HDF5 file: {exc}",
            hint=(
                "Expected the quantised high-resolution output written by "
                "aurora.h5out.write_highres_uint8_h5. Check that the file is not "
                "empty or truncated, and that it is not a compressed '.npz.zst' "
                "prediction (plotted by re-running predict_highres without "
                "--compress-output)."
            ),
        ) from exc


def _require_dataset(h5_file: "h5py.File", name: str) -> Any:
    """A dataset that must exist for the file to be an AURORA prediction file."""
    try:
        return h5_file[name]
    except KeyError as exc:
        present = ", ".join(sorted(h5_file.keys())[:8])
        raise ConfigError(
            f"{h5_file.filename} has no '{name}' dataset, so it is not an AURORA "
            "high-resolution prediction file",
            hint=(
                f"Expected a dataset named '{name}'. The file contains: "
                f"{present or '(nothing)'}."
            ),
        ) from exc


def _attr_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_strings(dataset: "h5py.Dataset") -> Any:
    np = _numpy()
    try:
        return np.asarray(dataset.asstr()[...])
    except (AttributeError, TypeError):
        values = np.asarray(dataset[...])
        if values.dtype.kind == "S":
            return np.char.decode(values, "utf-8")
        return values.astype(str)


def _logical_shape(dataset: "h5py.Dataset") -> Tuple[int, int]:
    """``(rows, channels)`` of a prediction dataset as stored logically.

    The dataset on disk is transposed for the delta encoding, so its real shape
    comes from the ``logical_shape`` attribute in that case.
    """
    encoding = _attr_text(dataset.attrs.get("encoding", "none"))
    if encoding == OPTIMIZED_H5_ENCODING:
        shape = tuple(int(v) for v in dataset.attrs["logical_shape"])
        if len(shape) != 2:
            raise ConfigError(
                f"invalid logical_shape on {dataset.name}: {shape}",
                hint=(
                    "The file's encoding metadata is inconsistent. Rewrite it with "
                    "aurora.h5out.write_highres_uint8_h5, or re-run the "
                    "high-resolution prediction stage."
                ),
            )
        return shape
    if dataset.ndim != 2:
        raise ConfigError(
            f"prediction dataset {dataset.name} is not 2-D (ndim={dataset.ndim})",
            hint=(
                "Expected a rows-by-channels matrix. This does not look like an "
                "AURORA high-resolution uint8 HDF5 file."
            ),
        )
    return tuple(int(v) for v in dataset.shape)


def iter_prediction_blocks(
    h5_file: "h5py.File",
    dataset_name: str,
    *,
    block_rows: Optional[int] = None,
) -> Iterator[Tuple[int, int, Any]]:
    """Yield decoded ``(start, end, rows_by_channels)`` blocks.

    For the optimized layout, blocks are in spatial row order and **must** be
    requested in the stored ``delta_block_rows`` size: each block starts with
    absolute values, so a different split cannot be decoded.  Use
    ``read_prediction_matrix(..., restore_original_order=True)`` when the exact
    pre-sort row order is required.
    """
    np = _numpy()
    dataset = h5_file[dataset_name]
    encoding = _attr_text(dataset.attrs.get("encoding", "none"))
    n_rows, _ = _logical_shape(dataset)

    if encoding == OPTIMIZED_H5_ENCODING:
        encoded_block_rows = int(dataset.attrs["delta_block_rows"])
        if block_rows is not None and int(block_rows) != encoded_block_rows:
            raise ConfigError(
                f"{dataset_name} must be decoded in its stored block size "
                f"({encoded_block_rows} rows), not {block_rows}",
                hint=(
                    "Each delta block begins with absolute values, so it cannot be "
                    "split. Omit block_rows to use the stored block size."
                ),
            )
        for start in range(0, n_rows, encoded_block_rows):
            end = min(start + encoded_block_rows, n_rows)
            try:
                encoded = dataset[:, start:end]
            except OSError as exc:
                codec = _attr_text(h5_file.attrs.get("compression_codec", "unknown"))
                if codec == "zstd" and _load_hdf5plugin() is None:
                    raise _zstd_unavailable_error(codec, str(exc)) from exc
                raise
            decoded = np.add.accumulate(encoded, axis=1, dtype=np.uint8).T
            yield start, end, np.ascontiguousarray(decoded)
        return

    if encoding not in {"none", "", "None"}:
        raise ConfigError(
            f"unsupported encoding {encoding!r} on {dataset.name}",
            hint=(
                f"Supported encodings are {OPTIMIZED_H5_ENCODING!r} (spatial gene "
                "delta) and 'none' (plain patch-major)."
            ),
        )

    rows_per_block = int(
        block_rows or (dataset.chunks[0] if dataset.chunks else DEFAULT_ROWS_PER_BLOCK)
    )
    rows_per_block = max(1, rows_per_block)
    for start in range(0, n_rows, rows_per_block):
        end = min(start + rows_per_block, n_rows)
        yield start, end, np.asarray(dataset[start:end, :])


def read_prediction_matrix(
    h5_file: "h5py.File",
    dataset_name: str,
    *,
    restore_original_order: bool = False,
) -> Any:
    """Decode a prediction dataset to its logical rows-by-channels matrix."""
    np = _numpy()
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
            raise ConfigError(
                f"{dataset_name} is spatially sorted, but its row permutation was "
                "not stored",
                hint=(
                    "Read it with restore_original_order=False and use the "
                    "spatially sorted coordinates in 'patch_loc_<ps>' instead."
                ),
            )
        return result

    stored_to_original = np.asarray(h5_file[order_name][...], dtype=np.int64)
    if stored_to_original.shape != (n_rows,):
        raise ConfigError(
            f"{order_name} has shape {stored_to_original.shape}; expected {(n_rows,)}",
            hint=(
                "The row permutation does not match the prediction rows, so the file "
                "is inconsistent; rewrite it with aurora.h5out."
            ),
        )
    restored = np.empty_like(result)
    restored[stored_to_original, :] = result
    return restored


def read_patch_locations(
    h5_file: "h5py.File",
    patch_size: int,
    *,
    restore_original_order: bool = False,
) -> Any:
    """Read row-aligned patch coordinates for one patch size.

    The stored coordinates follow the spatial row order; with
    ``restore_original_order=True`` the permutation is undone so they line up with
    the original row order again.
    """
    np = _numpy()
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


def available_patch_sizes(h5_file: "h5py.File") -> List[int]:
    """Patch sizes present in the file, from ``patch_sizes`` or the dataset names."""
    np = _numpy()
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
) -> Dict[str, Any]:
    """Load all prediction levels and metadata.

    ``dequantize=False`` returns exact saved uint8 matrices. With
    ``dequantize=True``, matrices are reconstructed as float32 using their
    per-channel maxima.

    The returned dict has ``genes``, ``cell_types``, ``resolutions``,
    ``patch_sizes``, ``layout`` and, per patch size, ``exp_pred``, ``prop_pred``,
    ``exp_max``, ``prop_max`` and ``pos``.
    """
    np = _numpy()
    # Register any available HDF5 filters (zstd) before opening.
    _load_hdf5plugin()

    result: Dict[str, Any] = {
        "exp_pred": {},
        "prop_pred": {},
        "exp_max": {},
        "prop_max": {},
        "pos": {},
    }
    with open_h5(path) as h5_file:
        _require_filter_available(h5_file)
        result["genes"] = _read_strings(_require_dataset(h5_file, "genes"))
        result["cell_types"] = _read_strings(_require_dataset(h5_file, "cell_types"))
        result["resolutions"] = np.asarray(_require_dataset(h5_file, "resolutions")[...])
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
                _require_dataset(h5_file, f"exp_max_{patch_size}")[...], dtype=np.float32
            )
            prop_max = np.asarray(
                _require_dataset(h5_file, f"prop_max_{patch_size}")[...], dtype=np.float32
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


__all__ = [
    "OPTIMIZED_H5_ENCODING",
    "available_patch_sizes",
    "iter_prediction_blocks",
    "load_highres_h5",
    "open_h5",
    "read_patch_locations",
    "read_prediction_matrix",
]
