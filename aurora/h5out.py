"""Write AURORA high-resolution predictions in the quantised HDF5 format.

This is the port of the writer that lives in
``code/predict_highres_TCGA_optimized.py`` (the script ``command.sh`` drives with
``--uint8_output --h5_layout spatial_gene_delta --h5_compression zstd``).  The
logic is deliberately kept identical so that any reader written against the
original -- ``code/highres_h5_loader.py``, now :mod:`aurora.h5in` -- accepts what
this module produces.

File layout
-----------

Root attributes::

    format                 = 'AURORA_highres_uint8'
    format_version         = uint32(2)
    prediction_layout      = 'spatial_gene_delta' | 'patch_major'
    compression_codec      = the codec name that was requested
    compression_level      = int
    logical_prediction_axes= 'rows,channels'
    rows_spatially_sorted  = True for 'spatial_gene_delta'
    row_order_direction    = 'stored_row_to_original_row'

Datasets, for every patch size ``ps``:

``genes``, ``cell_types``
    UTF-8 strings.
``resolutions``, ``patch_sizes``
    The prediction levels and the patch sizes present.
``exp_pred_<ps>``, ``prop_pred_<ps>``
    The quantised predictions, logically ``(rows, channels)``.
``exp_max_<ps>``, ``prop_max_<ps>``
    float32 per-channel maxima used to de-quantise.
``patch_loc_<ps>``
    ``(rows, 2)`` coordinates, with attribute ``row_layout``
    (``'spatial_sorted'`` or ``'original'``).
``row_order_<ps>``
    Present for ``spatial_gene_delta`` when ``store_row_order`` is set: the
    permutation mapping a stored row to its original row.

The ``spatial_gene_delta_v1`` encoding
--------------------------------------

``exp_pred_<ps>`` and ``prop_pred_<ps>`` are written **transposed** as
``(channels, rows)``, with the rows in a stable spatial raster order
(:func:`_spatial_row_order`).  Each block of ``delta_block_rows`` rows is
delta-coded independently along the row axis: the first row of a block stores
absolute values and every following row stores the difference from the previous
one, modulo 256 (u8 wraparound).  Blocks are therefore independently decodable,
which is what makes random block access possible without reading prior blocks.

Dequantisation is ``value_uint8 * (max / 255)`` per channel.

Dependencies
------------

Imported lazily: ``import aurora.h5out`` needs nothing installed.  ``h5py`` and
``numpy`` come from the ``core`` group through :func:`aurora.deps.require`, and
``hdf5plugin`` is only required for the ``zstd`` codec, where its absence is a
:class:`~aurora.errors.ConfigError` naming the fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .errors import ConfigError

#: Value of the ``encoding`` attribute on a delta-coded prediction dataset.
#: Read by :mod:`aurora.h5in` and by ``code/highres_h5_loader.py``.
OPTIMIZED_H5_ENCODING = "spatial_gene_delta_v1"

#: Layouts accepted by :func:`write_highres_uint8_h5`.
LAYOUT_SPATIAL_GENE_DELTA = "spatial_gene_delta"
LAYOUT_PATCH_MAJOR = "patch_major"

#: Default number of rows per delta block; one block is decoded independently.
DEFAULT_ROW_CHUNK = 16384

#: Default number of channels per HDF5 chunk.
DEFAULT_CHANNEL_CHUNK = 256


def _numpy() -> Any:
    from .deps import require

    return require("numpy", purpose="writing AURORA HDF5 predictions")


def _h5py() -> Any:
    from .deps import require

    return require("h5py", purpose="writing AURORA HDF5 predictions")


def _smallest_unsigned_dtype(max_value: int) -> Any:
    """Return the smallest unsigned dtype that can represent ``max_value``."""
    np = _numpy()
    max_value = int(max_value)
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_value <= np.iinfo(dtype).max:
            return dtype
    raise ValueError(f"Value {max_value} is too large for uint64")


def _spatial_row_order(patch_loc: Any) -> Any:
    """Stable raster order for (N, 2) coordinates.

    The second saved coordinate is the primary key and the first coordinate is
    the secondary key. Original row number is the final tie-breaker, making the
    result deterministic for duplicate coordinates.

    The returned array is indexed by *stored* row and yields the *original* row,
    which is exactly the ``stored_row_to_original_row`` mapping written to
    ``row_order_<ps>``.
    """
    np = _numpy()
    patch_loc = np.asarray(patch_loc)
    if patch_loc.ndim != 2 or patch_loc.shape[1] != 2:
        raise ValueError(f"patch_loc must have shape (N, 2), got {patch_loc.shape}")
    n_rows = patch_loc.shape[0]
    if n_rows == 0:
        return np.empty(0, dtype=np.int64)
    original_row = np.arange(n_rows, dtype=np.int64)
    return np.lexsort((original_row, patch_loc[:, 0], patch_loc[:, 1]))


def hdf5plugin_available() -> bool:
    """True when the optional Zstd HDF5 filter plugin can be imported."""
    try:
        import hdf5plugin  # noqa: F401
    except Exception:  # noqa: BLE001 - a broken install counts as absent
        return False
    return True


def default_codec() -> str:
    """The HDF5 codec to use when none was requested.

    ``zstd`` gives smaller files but needs ``hdf5plugin``, which is an optional extra:
    on a machine where it cannot be installed, writing with gzip keeps the pipeline
    working because h5py supports it natively.
    """
    return "zstd" if hdf5plugin_available() else "gzip"


def resolve_codec(requested: Optional[str]) -> str:
    """``requested`` if the user picked one, otherwise :func:`default_codec`."""
    if requested is None:
        return default_codec()
    text = str(requested).strip().lower()
    return text or default_codec()


def _h5_compression_kwargs(codec: str, level: int) -> Dict[str, Any]:
    """Build h5py dataset filter arguments.

    ``{'compression': ...}`` is spread straight into ``create_dataset``, so the
    returned dict is the whole filter specification.
    """
    codec = str(codec).lower()
    if codec == "zstd":
        if not (-131072 <= int(level) <= 22):
            raise ConfigError(
                f"Zstd compression level must be between -131072 and 22, got {level}",
                hint=(
                    "Pass --h5-compression-level within that range; the AURORA "
                    "default is 9."
                ),
            )
        try:
            import hdf5plugin
        except ImportError as exc:
            raise ConfigError(
                "the 'zstd' HDF5 compression codec requires the hdf5plugin package, "
                "which is not installed",
                hint=(
                    "Install it with:\n"
                    "  python -m pip install hdf5plugin\n"
                    "and import hdf5plugin before reading the resulting file. To avoid "
                    "the dependency entirely, write with "
                    "--h5-compression gzip or --h5-compression none."
                ),
            ) from exc
        return {"compression": hdf5plugin.Zstd(clevel=int(level))}
    if codec == "gzip":
        if not (0 <= int(level) <= 9):
            raise ConfigError(
                f"gzip compression level must be between 0 and 9, got {level}",
                hint="Pass --h5-compression-level between 0 and 9.",
            )
        return {"compression": "gzip", "compression_opts": int(level)}
    if codec == "none":
        return {}
    raise ConfigError(
        f"unsupported HDF5 compression codec: {codec}",
        hint=(
            "Supported codecs are 'zstd' (needs hdf5plugin), 'gzip' (built into "
            "h5py) and 'none'."
        ),
    )


def _set_prediction_encoding_attrs(
    dataset: Any,
    logical_shape: Sequence[int],
    delta_block_rows: int,
    row_order_dataset: str,
) -> None:
    """Attach the metadata needed to decode a transformed prediction dataset."""
    np = _numpy()
    dataset.attrs["encoding"] = OPTIMIZED_H5_ENCODING
    dataset.attrs["logical_shape"] = np.asarray(logical_shape, dtype=np.uint64)
    dataset.attrs["stored_axes"] = "channels,spatial_rows"
    dataset.attrs["delta_axis"] = "spatial_rows"
    dataset.attrs["delta_mode"] = "uint8_modulo_256"
    dataset.attrs["delta_block_rows"] = int(delta_block_rows)
    dataset.attrs["row_order_dataset"] = row_order_dataset or ""


def _write_spatial_gene_delta_dataset(
    hf: Any,
    name: str,
    data: Any,
    spatial_order: Any,
    compression_kwargs: Mapping[str, Any],
    row_chunk: int = DEFAULT_ROW_CHUNK,
    channel_chunk: int = DEFAULT_CHANNEL_CHUNK,
    row_order_dataset: str = "",
) -> Any:
    """Write a rows-by-channels uint8 matrix as spatial gene-major deltas.

    Stored shape is ``(channels, rows)``. Each row block starts with absolute
    values and then stores adjacent differences modulo 256, so blocks can be
    decoded independently and random block access does not need prior blocks.
    """
    np = _numpy()
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {data.shape}")
    if data.dtype != np.uint8:
        raise TypeError(f"{name} must be uint8, got {data.dtype}")

    n_rows, n_channels = data.shape
    spatial_order = np.asarray(spatial_order, dtype=np.int64)
    if spatial_order.shape != (n_rows,):
        raise ValueError(
            f"spatial order for {name} has shape {spatial_order.shape}; "
            f"expected {(n_rows,)}"
        )

    if n_rows == 0 or n_channels == 0:
        dataset = hf.create_dataset(name, shape=(n_channels, n_rows), dtype=np.uint8)
        _set_prediction_encoding_attrs(
            dataset, data.shape, max(1, int(row_chunk)), row_order_dataset
        )
        return dataset

    row_chunk = max(1, min(int(row_chunk), n_rows))
    channel_chunk = max(1, min(int(channel_chunk), n_channels))
    dataset = hf.create_dataset(
        name,
        shape=(n_channels, n_rows),
        dtype=np.uint8,
        chunks=(channel_chunk, row_chunk),
        **compression_kwargs,
    )
    _set_prediction_encoding_attrs(dataset, data.shape, row_chunk, row_order_dataset)

    for row_start in range(0, n_rows, row_chunk):
        row_end = min(row_start + row_chunk, n_rows)
        # Defaults create a ~46.9 MiB source block for 3,002 channels, matching
        # the benchmark sample while avoiding another full-size array copy.
        values = np.ascontiguousarray(data[spatial_order[row_start:row_end], :].T)
        encoded = np.empty_like(values)
        encoded[:, 0] = values[:, 0]
        if values.shape[1] > 1:
            np.subtract(values[:, 1:], values[:, :-1], out=encoded[:, 1:])

        for channel_start in range(0, n_channels, channel_chunk):
            channel_end = min(channel_start + channel_chunk, n_channels)
            dataset[channel_start:channel_end, row_start:row_end] = encoded[
                channel_start:channel_end, :
            ]
        del values, encoded

    return dataset


def _write_compressed_array(
    hf: Any,
    name: str,
    data: Any,
    compression_kwargs: Mapping[str, Any],
    target_chunk_bytes: int = 4 * 1024 * 1024,
    shuffle_multibyte: bool = True,
) -> Any:
    """Write a regular array with bounded chunks and the selected filter."""
    np = _numpy()
    data = np.asarray(data)
    if data.size == 0 or data.ndim == 0 or not compression_kwargs:
        return hf.create_dataset(name, data=data)

    kwargs = dict(compression_kwargs)
    if shuffle_multibyte and data.dtype.itemsize > 1:
        kwargs["shuffle"] = True

    if data.ndim == 1:
        elements = max(1, target_chunk_bytes // max(1, data.dtype.itemsize))
        chunks = (min(data.shape[0], elements),)
    elif data.ndim == 2:
        row_bytes = max(1, data.shape[1] * data.dtype.itemsize)
        rows = max(1, target_chunk_bytes // row_bytes)
        chunks = (min(data.shape[0], rows), data.shape[1])
    else:
        chunks = True
    return hf.create_dataset(name, data=data, chunks=chunks, **kwargs)


def write_highres_uint8_h5(
    h5_path: str | Path,
    gene_list: Sequence[str],
    major_cell_type: Sequence[str],
    resolutions: Sequence[int],
    patch_sizes: Sequence[int],
    exp_pred_mem: Mapping[int, Any],
    prop_pred_mem: Mapping[int, Any],
    exp_max_mem: Mapping[int, Any],
    prop_max_mem: Mapping[int, Any],
    patch_loc_mem: Mapping[int, Any],
    layout: str = LAYOUT_SPATIAL_GENE_DELTA,
    compression: str = "zstd",
    compression_level: int = 9,
    row_chunk: int = DEFAULT_ROW_CHUNK,
    channel_chunk: int = DEFAULT_CHANNEL_CHUNK,
    store_row_order: bool = True,
) -> Path:
    """Write quantized predictions using the selected HDF5 layout and filter.

    ``exp_pred_mem`` / ``prop_pred_mem`` map a patch size to a uint8
    ``(rows, channels)`` matrix in *original* row order, ``patch_loc_mem`` maps it
    to the matching ``(rows, 2)`` coordinates, and the ``*_max_mem`` mappings hold
    the float32 per-channel maxima used for de-quantisation.

    Returns the written path.

    The parent directory must already exist, as in the original: creating it here
    would hide a wrong output path.  The pipeline's stage directories are created
    by ``aurora.paths.RunLayout.ensure()``.

    Layouts:

    ``spatial_gene_delta`` (default)
        Rows are re-ordered into a stable spatial raster, delta-coded in blocks
        and stored transposed as ``(channels, rows)``, with the permutation in
        ``row_order_<ps>``.
    ``patch_major``
        The matrices are stored as given, with ``encoding='none'`` and no row
        order.
    """
    np = _numpy()
    h5py = _h5py()

    compression_kwargs = _h5_compression_kwargs(compression, compression_level)
    row_chunk = max(1, int(row_chunk))
    channel_chunk = max(1, int(channel_chunk))

    if layout not in (LAYOUT_SPATIAL_GENE_DELTA, LAYOUT_PATCH_MAJOR):
        raise ConfigError(
            f"unknown HDF5 layout: {layout}",
            hint=(
                f"Supported layouts are '{LAYOUT_SPATIAL_GENE_DELTA}' (spatially "
                f"delta-coded, the default) and '{LAYOUT_PATCH_MAJOR}'."
            ),
        )

    h5_path = Path(h5_path)

    # Keep several 4 MiB chunks resident while splitting one row block across
    # channel chunks.
    cache_bytes = max(64 * 1024 * 1024, row_chunk * channel_chunk * 4)
    with h5py.File(h5_path, "w", rdcc_nbytes=cache_bytes, rdcc_nslots=10007) as hf:
        hf.attrs["format"] = "AURORA_highres_uint8"
        hf.attrs["format_version"] = np.uint32(2)
        hf.attrs["prediction_layout"] = layout
        hf.attrs["compression_codec"] = compression
        hf.attrs["compression_level"] = int(compression_level)
        hf.attrs["logical_prediction_axes"] = "rows,channels"
        hf.attrs["rows_spatially_sorted"] = bool(layout == LAYOUT_SPATIAL_GENE_DELTA)
        hf.attrs["row_order_direction"] = "stored_row_to_original_row"

        hf.create_dataset(
            "genes",
            data=np.asarray(gene_list, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        hf.create_dataset(
            "cell_types",
            data=np.asarray(major_cell_type, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        hf.create_dataset("resolutions", data=np.asarray(resolutions))
        hf.create_dataset("patch_sizes", data=np.asarray(patch_sizes, dtype=np.int32))

        for ps in patch_sizes:
            print(f"    Writing optimized HDF5 patch_size {ps} ...", flush=True)
            exp_data = np.asarray(exp_pred_mem[ps])
            prop_data = np.asarray(prop_pred_mem[ps])
            patch_loc = np.asarray(patch_loc_mem[ps])
            if exp_data.shape[0] != patch_loc.shape[0]:
                raise ConfigError(
                    f"exp_pred_{ps} rows ({exp_data.shape[0]}) do not match "
                    f"patch_loc_{ps} rows ({patch_loc.shape[0]})",
                    hint=(
                        "Every prediction row must have one coordinate. If rows were "
                        "filtered (for example by the HistoSweep tissue mask), filter "
                        "the coordinates with the same flag so they stay aligned."
                    ),
                )
            if prop_data.shape[0] != patch_loc.shape[0]:
                raise ConfigError(
                    f"prop_pred_{ps} rows ({prop_data.shape[0]}) do not match "
                    f"patch_loc_{ps} rows ({patch_loc.shape[0]})",
                    hint=(
                        "Every prediction row must have one coordinate. If rows were "
                        "filtered (for example by the HistoSweep tissue mask), filter "
                        "the coordinates with the same flag so they stay aligned."
                    ),
                )

            if layout == LAYOUT_SPATIAL_GENE_DELTA:
                order = _spatial_row_order(patch_loc)
                row_order_name = f"row_order_{ps}" if store_row_order else ""
                _write_spatial_gene_delta_dataset(
                    hf,
                    f"exp_pred_{ps}",
                    exp_data,
                    order,
                    compression_kwargs,
                    row_chunk=row_chunk,
                    channel_chunk=channel_chunk,
                    row_order_dataset=row_order_name,
                )
                _write_spatial_gene_delta_dataset(
                    hf,
                    f"prop_pred_{ps}",
                    prop_data,
                    order,
                    compression_kwargs,
                    row_chunk=row_chunk,
                    channel_chunk=min(channel_chunk, max(1, prop_data.shape[1])),
                    row_order_dataset=row_order_name,
                )

                sorted_loc = np.ascontiguousarray(patch_loc[order])
                loc_ds = _write_compressed_array(
                    hf, f"patch_loc_{ps}", sorted_loc, compression_kwargs
                )
                loc_ds.attrs["row_layout"] = "spatial_sorted"
                loc_ds.attrs["row_order_dataset"] = row_order_name

                if store_row_order:
                    order_dtype = _smallest_unsigned_dtype(max(0, len(order) - 1))
                    order_ds = _write_compressed_array(
                        hf,
                        row_order_name,
                        order.astype(order_dtype, copy=False),
                        compression_kwargs,
                    )
                    order_ds.attrs["mapping"] = "stored_row_to_original_row"
                del order, sorted_loc
            elif layout == LAYOUT_PATCH_MAJOR:
                exp_ds = _write_compressed_array(
                    hf, f"exp_pred_{ps}", exp_data, compression_kwargs
                )
                prop_ds = _write_compressed_array(
                    hf, f"prop_pred_{ps}", prop_data, compression_kwargs
                )
                loc_ds = _write_compressed_array(
                    hf, f"patch_loc_{ps}", patch_loc, compression_kwargs
                )
                exp_ds.attrs["encoding"] = "none"
                prop_ds.attrs["encoding"] = "none"
                loc_ds.attrs["row_layout"] = "original"
            else:  # pragma: no cover - guarded above
                raise ConfigError(f"unknown HDF5 layout: {layout}")

            hf.create_dataset(
                f"exp_max_{ps}", data=np.asarray(exp_max_mem[ps], dtype=np.float32)
            )
            hf.create_dataset(
                f"prop_max_{ps}", data=np.asarray(prop_max_mem[ps], dtype=np.float32)
            )
        hf.flush()

    return h5_path


__all__ = [
    "DEFAULT_CHANNEL_CHUNK",
    "DEFAULT_ROW_CHUNK",
    "LAYOUT_PATCH_MAJOR",
    "LAYOUT_SPATIAL_GENE_DELTA",
    "OPTIMIZED_H5_ENCODING",
    "write_highres_uint8_h5",
]
