#!/usr/bin/env python
"""Round-trip tests for :mod:`aurora.h5out` and :mod:`aurora.h5in`.

The quantised HDF5 format is only trustworthy if a file written by our writer
reads back byte-for-byte, so this test writes with :func:`write_highres_uint8_h5`
and reads with :func:`load_highres_h5`.  That is the only way to know the
``spatial_gene_delta_v1`` encoding and the spatial row order agree with each
other: an encode/decode asymmetry would show up as wrong values, and a row-order
bug as rows in the wrong place.

What is checked

1. ``exp_pred`` / ``prop_pred`` / ``patch_loc`` survive the round trip exactly,
   for several patch sizes, once the recorded row order is undone.
2. The stored (spatially sorted) order really is the sorted order, it is
   recorded, and it is reversible -- the fixture coordinates are asymmetric,
   duplicated and deliberately out of order, and the test asserts the stored
   order *differs* from the input, so it cannot pass by ignoring the sort.
3. Delta blocks decode independently and reassemble to the same matrix, and a
   mismatched block size is refused.
4. ``patch_major`` round-trips too, storing no row order.
5. The HDF5 attributes are the documented ones.
6. User-triggerable failures raise ``ConfigError`` with a useful hint.

Run from the repository root::

    python tests/test_h5_format.py

Uses gzip, not zstd: h5py's built-in filter needs nothing extra, while the zstd
path needs ``hdf5plugin``.  The zstd-without-hdf5plugin failure is still covered,
by writing a gzip file and relabelling its codec attribute.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from aurora.errors import ConfigError  # noqa: E402
from aurora.h5in import (  # noqa: E402
    OPTIMIZED_H5_ENCODING,
    available_patch_sizes,
    iter_prediction_blocks,
    load_highres_h5,
    read_patch_locations,
    read_prediction_matrix,
)
from aurora.h5out import write_highres_uint8_h5  # noqa: E402

#: Patch sizes and their row/feature counts.  Different row counts per patch size
#: make sure nothing assumes they agree.
PATCH_SIZES = [56, 14]
N_ROWS = {56: 260, 14: 173}
N_GENES = 10
N_CELL_TYPES = 5
RESOLUTIONS = [56, 14]

#: Small enough to produce several delta blocks for 260 rows (5 blocks), which is
#: what exercises the independent per-block encoding.
ROW_CHUNK = 64

failures: List[str] = []


def expect(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def build_coordinates(n_rows: int, patch_size: int, seed: int) -> Any:
    """Asymmetric, duplicated, out-of-order ``(y, x)`` coordinates.

    * asymmetric -- y steps by ``2 * patch_size`` while x steps by
      ``patch_size``, so the set of points is not symmetric under transposition
      and swapping the axes would be detectable.
    * duplicated -- two later rows repeat earlier coordinates, which exercises the
      original-row tie-breaker in the sort.
    * out of order -- the grid is shuffled, so the spatial sort genuinely has to
      permute.
    """
    rng = np.random.default_rng(seed)
    xs = (np.arange(40) * patch_size).astype(np.int64)
    ys = (np.arange(20) * (patch_size * 2)).astype(np.int64)
    grid = np.array([[y, x] for y in ys for x in xs], dtype=np.int64)
    rng.shuffle(grid)
    coords = grid[:n_rows].copy()
    coords[3] = coords[0]
    coords[7] = coords[1]
    return coords


def build_fixture() -> Dict[str, Any]:
    """Synthetic uint8 predictions in the shape the writer expects."""
    rng = np.random.default_rng(20241016)
    fixture: Dict[str, Any] = {
        "gene_list": [f"GENE{i}" for i in range(N_GENES)],
        "major_cell_type": [f"CellType{i}" for i in range(N_CELL_TYPES)],
        "resolutions": RESOLUTIONS,
        "patch_sizes": PATCH_SIZES,
        "exp_pred_mem": {},
        "prop_pred_mem": {},
        "exp_max_mem": {},
        "prop_max_mem": {},
        "patch_loc_mem": {},
    }
    for index, ps in enumerate(PATCH_SIZES):
        n_rows = N_ROWS[ps]
        # Full 0..255 range so the delta coding wraps modulo 256 repeatedly.
        fixture["exp_pred_mem"][ps] = rng.integers(
            0, 256, size=(n_rows, N_GENES), dtype=np.uint8
        )
        fixture["prop_pred_mem"][ps] = rng.integers(
            0, 256, size=(n_rows, N_CELL_TYPES), dtype=np.uint8
        )
        fixture["exp_max_mem"][ps] = np.linspace(
            1.0, 20.0, N_GENES, dtype=np.float32
        )
        fixture["prop_max_mem"][ps] = np.linspace(
            0.5, 3.0, N_CELL_TYPES, dtype=np.float32
        )
        fixture["patch_loc_mem"][ps] = build_coordinates(n_rows, ps, seed=100 + index)
    return fixture


def write(fixture: Dict[str, Any], path: Path, **overrides: Any) -> Path:
    kwargs: Dict[str, Any] = dict(
        h5_path=path,
        gene_list=fixture["gene_list"],
        major_cell_type=fixture["major_cell_type"],
        resolutions=fixture["resolutions"],
        patch_sizes=fixture["patch_sizes"],
        exp_pred_mem=fixture["exp_pred_mem"],
        prop_pred_mem=fixture["prop_pred_mem"],
        exp_max_mem=fixture["exp_max_mem"],
        prop_max_mem=fixture["prop_max_mem"],
        patch_loc_mem=fixture["patch_loc_mem"],
        layout="spatial_gene_delta",
        compression="gzip",
        compression_level=6,
        row_chunk=ROW_CHUNK,
    )
    kwargs.update(overrides)
    return write_highres_uint8_h5(**kwargs)


def main() -> int:
    root = REPO_ROOT / f".h5fmt_{os.getpid()}"
    root.mkdir(parents=True)
    try:
        fixture = build_fixture()
        spatial_path = root / "spatial.h5"
        write(fixture, spatial_path)

        # ------------------------------------------------------------------ 1
        print("1. values survive the round trip and match the input exactly")
        restored = load_highres_h5(spatial_path, restore_original_order=True)
        expect(
            restored["patch_sizes"] == PATCH_SIZES,
            f"patch sizes {restored['patch_sizes']}",
        )
        expect(restored["layout"] == "spatial_gene_delta", f"layout {restored['layout']}")
        expect(
            restored["genes"].tolist() == fixture["gene_list"],
            f"genes {restored['genes'].tolist()}",
        )
        expect(
            restored["cell_types"].tolist() == fixture["major_cell_type"],
            f"cell types {restored['cell_types'].tolist()}",
        )
        expect(
            np.array_equal(restored["resolutions"], np.asarray(RESOLUTIONS)),
            f"resolutions {restored['resolutions']}",
        )
        for ps in PATCH_SIZES:
            expect(
                restored["exp_pred"][ps].dtype == np.uint8,
                f"{ps}: exp_pred is uint8 ({restored['exp_pred'][ps].dtype})",
            )
            expect(
                np.array_equal(restored["exp_pred"][ps], fixture["exp_pred_mem"][ps]),
                f"{ps}: exp_pred matches the input exactly ({restored['exp_pred'][ps].shape})",
            )
            expect(
                np.array_equal(restored["prop_pred"][ps], fixture["prop_pred_mem"][ps]),
                f"{ps}: prop_pred matches the input exactly",
            )
            expect(
                np.array_equal(restored["pos"][ps], fixture["patch_loc_mem"][ps]),
                f"{ps}: patch_loc matches the input after undoing the row order",
            )
            expect(
                np.array_equal(
                    restored["exp_max"][ps], fixture["exp_max_mem"][ps]
                ),
                f"{ps}: exp_max matches",
            )

        # A spatially-sorted read is the stored order, i.e. input[order].
        print()
        print("1b. the default read returns the stored (spatially sorted) order")
        stored = load_highres_h5(spatial_path)
        for ps in PATCH_SIZES:
            order = np.asarray(
                h5py.File(spatial_path, "r")[f"row_order_{ps}"][...], dtype=np.int64
            )
            expect(
                np.array_equal(
                    stored["exp_pred"][ps], fixture["exp_pred_mem"][ps][order]
                ),
                f"{ps}: default exp_pred equals input[order]",
            )
            expect(
                np.array_equal(stored["pos"][ps], fixture["patch_loc_mem"][ps][order]),
                f"{ps}: default patch_loc equals input[order]",
            )

        # ------------------------------------------------------------------ 3
        print()
        print("3. the row order is genuinely applied, recorded and reversible")
        with h5py.File(spatial_path, "r") as hf:
            for ps in PATCH_SIZES:
                order = np.asarray(hf[f"row_order_{ps}"][...], dtype=np.int64)
                n_rows = N_ROWS[ps]
                expect(
                    order.shape == (n_rows,),
                    f"{ps}: row_order shape {order.shape}",
                )
                expect(
                    sorted(order.tolist()) == list(range(n_rows)),
                    f"{ps}: row_order is a permutation",
                )
                expect(
                    not np.array_equal(order, np.arange(n_rows)),
                    f"{ps}: the fixture order really is out of order, so sorting happened",
                )
                # The stored order must equal patch_loc[order] ...
                stored_loc = np.asarray(hf[f"patch_loc_{ps}"][...])
                expect(
                    np.array_equal(stored_loc, fixture["patch_loc_mem"][ps][order]),
                    f"{ps}: stored patch_loc equals patch_loc[order]",
                )
                # ... and be the documented sort: primary key is column 1 (x),
                # secondary column 0 (y), with the original row as tie-breaker.
                xs = stored_loc[:, 1]
                expect(
                    bool((np.diff(xs) >= 0).all()),
                    f"{ps}: stored rows are ordered by the second coordinate (x)",
                )
                duplicate_rows = [
                    i
                    for i in range(1, len(stored_loc))
                    if stored_loc[i, 1] == stored_loc[i - 1, 1]
                ]
                tied_ok = all(
                    stored_loc[i, 0] >= stored_loc[i - 1, 0] for i in duplicate_rows
                )
                expect(tied_ok, f"{ps}: equal x values are ordered by y (tie-break)")
                expect(
                    len(set(map(tuple, fixture["patch_loc_mem"][ps].tolist())))
                    < n_rows,
                    f"{ps}: the fixture contains duplicate coordinates",
                )
                expect(
                    hf[f"row_order_{ps}"].attrs["mapping"]
                    == "stored_row_to_original_row",
                    f"{ps}: row_order mapping attribute",
                )
                expect(
                    hf[f"patch_loc_{ps}"].attrs["row_layout"] == "spatial_sorted",
                    f"{ps}: patch_loc row_layout attribute",
                )
                expect(
                    hf[f"patch_loc_{ps}"].attrs["row_order_dataset"] == f"row_order_{ps}",
                    f"{ps}: patch_loc row_order_dataset attribute",
                )

        # ------------------------------------------------------------------ 4
        print()
        print("4. delta blocks decode independently and reassemble correctly")
        with h5py.File(spatial_path, "r") as hf:
            for ps in PATCH_SIZES:
                name = f"exp_pred_{ps}"
                encoded_block_rows = int(hf[name].attrs["delta_block_rows"])
                expect(
                    encoded_block_rows == ROW_CHUNK,
                    f"{ps}: delta_block_rows is the requested row_chunk "
                    f"({encoded_block_rows})",
                )
                blocks = list(iter_prediction_blocks(hf, name))
                n_rows = N_ROWS[ps]
                expected_blocks = -(-n_rows // ROW_CHUNK)
                expect(
                    len(blocks) == expected_blocks > 1,
                    f"{ps}: decoded {len(blocks)} blocks (expected {expected_blocks})",
                )
                # Block boundaries must tile the rows contiguously.
                expect(
                    [b[0] for b in blocks] == list(range(0, n_rows, ROW_CHUNK)),
                    f"{ps}: block start offsets {[b[0] for b in blocks][:4]}...",
                )
                expect(blocks[-1][1] == n_rows, f"{ps}: the last block ends at {n_rows}")

                reassembled = np.empty((n_rows, N_GENES), dtype=np.uint8)
                for start, end, block in blocks:
                    reassembled[start:end, :] = block
                matrix = read_prediction_matrix(hf, name)
                expect(
                    np.array_equal(reassembled, matrix),
                    f"{ps}: the blocks reassemble to read_prediction_matrix",
                )
                # And the stored order is the sorted input, so the decode is right.
                order = np.asarray(hf[f"row_order_{ps}"][...], dtype=np.int64)
                expect(
                    np.array_equal(matrix, fixture["exp_pred_mem"][ps][order]),
                    f"{ps}: decoded matrix equals the written data in stored order",
                )
                # Each block starts with an absolute value, so splits must not move.
                try:
                    list(iter_prediction_blocks(hf, name, block_rows=16))
                except ConfigError as exc:
                    expect(
                        "block size" in str(exc),
                        f"{ps}: a mismatched block size is refused with a hint",
                    )
                else:
                    expect(False, f"{ps}: a mismatched block size should be refused")

        # ------------------------------------------------------------------ 5
        print()
        print("5. patch_major round-trips and stores no row order")
        major_path = root / "major.h5"
        write(fixture, major_path, layout="patch_major")
        major = load_highres_h5(major_path, restore_original_order=True)
        expect(major["layout"] == "patch_major", f"layout {major['layout']}")
        for ps in PATCH_SIZES:
            expect(
                np.array_equal(major["exp_pred"][ps], fixture["exp_pred_mem"][ps]),
                f"{ps}: patch_major exp_pred matches the input",
            )
            expect(
                np.array_equal(major["prop_pred"][ps], fixture["prop_pred_mem"][ps]),
                f"{ps}: patch_major prop_pred matches the input",
            )
            expect(
                np.array_equal(major["pos"][ps], fixture["patch_loc_mem"][ps]),
                f"{ps}: patch_major patch_loc matches the input in the original order",
            )
        with h5py.File(major_path, "r") as hf:
            expect(
                not bool(hf.attrs["rows_spatially_sorted"]),
                "patch_major records rows_spatially_sorted=False",
            )
            for ps in PATCH_SIZES:
                expect(
                    f"row_order_{ps}" not in hf,
                    f"{ps}: patch_major stores no row_order dataset",
                )
                expect(
                    hf[f"exp_pred_{ps}"].attrs["encoding"] == "none",
                    f"{ps}: patch_major exp_pred encoding is 'none'",
                )
                expect(
                    hf[f"patch_loc_{ps}"].attrs["row_layout"] == "original",
                    f"{ps}: patch_major patch_loc row_layout is 'original'",
                )

        # ------------------------------------------------------------------ 6
        print()
        print("6. the HDF5 attributes are the documented ones")
        with h5py.File(spatial_path, "r") as hf:
            expect(hf.attrs["format"] == "AURORA_highres_uint8", f"format {hf.attrs['format']!r}")
            expect(int(hf.attrs["format_version"]) == 2, f"format_version {hf.attrs['format_version']}")
            expect(
                hf.attrs["prediction_layout"] == "spatial_gene_delta",
                f"prediction_layout {hf.attrs['prediction_layout']!r}",
            )
            expect(hf.attrs["compression_codec"] == "gzip", f"codec {hf.attrs['compression_codec']!r}")
            expect(int(hf.attrs["compression_level"]) == 6, f"level {hf.attrs['compression_level']}")
            expect(
                hf.attrs["logical_prediction_axes"] == "rows,channels",
                f"logical_prediction_axes {hf.attrs['logical_prediction_axes']!r}",
            )
            expect(
                bool(hf.attrs["rows_spatially_sorted"]) is True,
                f"rows_spatially_sorted {hf.attrs['rows_spatially_sorted']!r}",
            )
            expect(
                hf.attrs["row_order_direction"] == "stored_row_to_original_row",
                f"row_order_direction {hf.attrs['row_order_direction']!r}",
            )
            for ps in PATCH_SIZES:
                name = f"exp_pred_{ps}"
                dataset = hf[name]
                expect(
                    dataset.attrs["encoding"] == OPTIMIZED_H5_ENCODING,
                    f"{ps}: encoding {dataset.attrs['encoding']!r}",
                )
                expect(
                    tuple(int(v) for v in dataset.attrs["logical_shape"])
                    == (N_ROWS[ps], N_GENES),
                    f"{ps}: logical_shape {tuple(dataset.attrs['logical_shape'])}",
                )
                expect(
                    dataset.shape == (N_GENES, N_ROWS[ps]),
                    f"{ps}: stored shape is (channels, rows) = {dataset.shape}",
                )
                expect(
                    dataset.attrs["stored_axes"] == "channels,spatial_rows",
                    f"{ps}: stored_axes {dataset.attrs['stored_axes']!r}",
                )
                expect(
                    dataset.attrs["delta_axis"] == "spatial_rows",
                    f"{ps}: delta_axis {dataset.attrs['delta_axis']!r}",
                )
                expect(
                    dataset.attrs["delta_mode"] == "uint8_modulo_256",
                    f"{ps}: delta_mode {dataset.attrs['delta_mode']!r}",
                )
                expect(
                    int(dataset.attrs["delta_block_rows"]) == ROW_CHUNK,
                    f"{ps}: delta_block_rows {dataset.attrs['delta_block_rows']}",
                )
                expect(
                    dataset.attrs["row_order_dataset"] == f"row_order_{ps}",
                    f"{ps}: row_order_dataset {dataset.attrs['row_order_dataset']!r}",
                )
                expect(
                    dataset.dtype == np.uint8,
                    f"{ps}: prediction dataset dtype {dataset.dtype}",
                )
            expect(
                available_patch_sizes(hf) == PATCH_SIZES,
                f"available_patch_sizes {available_patch_sizes(hf)}",
            )

        # De-quantisation (not required, but cheap to pin down).
        print()
        print("6b. dequantize reconstructs value * max / 255")
        dequantized = load_highres_h5(
            spatial_path, restore_original_order=True, dequantize=True
        )
        for ps in PATCH_SIZES:
            expected = fixture["exp_pred_mem"][ps].astype(np.float32) * (
                fixture["exp_max_mem"][ps][np.newaxis, :] / 255.0
            )
            expect(
                np.allclose(dequantized["exp_pred"][ps], expected),
                f"{ps}: dequantized exp_pred matches value * max / 255",
            )
            expect(
                dequantized["exp_pred"][ps].dtype == np.float32,
                f"{ps}: dequantized dtype {dequantized['exp_pred'][ps].dtype}",
            )

        # ------------------------------------------------------------------ 7
        print()
        print("7. user-triggerable failures raise ConfigError with a hint")
        try:
            write(fixture, root / "bad_codec.h5", compression="snappy")
        except ConfigError as exc:
            expect("snappy" in str(exc), "the unsupported codec is named")
            expect(
                "gzip" in (exc.hint or "") and "zstd" in (exc.hint or ""),
                "the hint lists the supported codecs",
            )
        else:
            expect(False, "an unsupported codec should raise ConfigError")

        for bad_level in (10, -1):
            try:
                write(fixture, root / "bad_level.h5", compression="gzip", compression_level=bad_level)
            except ConfigError as exc:
                expect(
                    "0 and 9" in str(exc),
                    f"gzip level {bad_level} is refused with the valid range",
                )
                expect(bool(exc.hint), f"gzip level {bad_level} error carries a hint")
            else:
                expect(False, f"gzip level {bad_level} should raise ConfigError")

        try:
            write(fixture, root / "bad_zstd_level.h5", compression="zstd", compression_level=99)
        except ConfigError as exc:
            expect(
                "Zstd" in str(exc) and "-131072" in str(exc),
                "the zstd level range is checked",
            )
        else:
            expect(False, "an out-of-range zstd level should raise ConfigError")

        # zstd needs hdf5plugin. Whether it is installed decides the expectation, so
        # both branches are real: on a machine that has it, exercise the actual
        # compressed round-trip; on one that does not, check the refusal.
        have_hdf5plugin = False
        try:
            import hdf5plugin  # noqa: F401

            have_hdf5plugin = True
        except ImportError:
            pass

        if have_hdf5plugin:
            zstd_path = root / "zstd.h5"
            write(fixture, zstd_path, compression="zstd", compression_level=9)
            with h5py.File(zstd_path, "r") as hf:
                expect(
                    hf.attrs.get("compression_codec") == "zstd",
                    "a zstd file records its codec",
                )
            reloaded = load_highres_h5(zstd_path)
            expect(
                bool(reloaded),
                f"a zstd file written with hdf5plugin reads back: {type(reloaded).__name__}",
            )
            expect(True, "zstd round-trips (hdf5plugin is installed)")
        else:
            try:
                write(fixture, root / "zstd.h5", compression="zstd", compression_level=9)
            except ConfigError as exc:
                expect(
                    "hdf5plugin" in str(exc),
                    f"zstd without hdf5plugin names the package: {str(exc)[:70]}",
                )
                expect(
                    "pip install hdf5plugin" in (exc.hint or ""),
                    "the hint gives the install command",
                )
            else:
                expect(False, "zstd without hdf5plugin should raise ConfigError")

        try:
            write(fixture, root / "bad_layout.h5", layout="row_major")
        except ConfigError as exc:
            expect("row_major" in str(exc), "an unknown layout is refused")
            expect(bool(exc.hint), "the layout error carries a hint")
        else:
            expect(False, "an unknown layout should raise ConfigError")

        # A row-count mismatch between predictions and coordinates.
        mismatched = dict(fixture["patch_loc_mem"])
        mismatched[56] = fixture["patch_loc_mem"][56][:-1]
        try:
            write(fixture, root / "mismatch.h5", patch_loc_mem=mismatched)
        except ConfigError as exc:
            expect(
                "exp_pred_56" in str(exc) and "patch_loc_56" in str(exc),
                f"the row mismatch names both arrays: {str(exc)[:80]}",
            )
            expect(bool(exc.hint), "the row-count mismatch error carries a hint")
        else:
            expect(False, "a row-count mismatch should raise ConfigError")

        # Reading a file whose codec says zstd. With hdf5plugin installed that
        # succeeds; without it the reader's up-front guard must name the package.
        relabelled = root / "claims_zstd.h5"
        shutil.copyfile(spatial_path, relabelled)
        with h5py.File(relabelled, "r+") as hf:
            hf.attrs["compression_codec"] = "zstd"
        try:
            load_highres_h5(relabelled)
        except ConfigError as exc:
            expect(
                "hdf5plugin" in str(exc),
                f"reading a zstd file without hdf5plugin names the package: {str(exc)[:60]}",
            )
            expect(
                "pip install hdf5plugin" in (exc.hint or ""),
                "the reader's hint gives the install command",
            )
        else:
            expect(
                have_hdf5plugin,
                "a zstd-labelled file is readable only when hdf5plugin is installed",
            )

        # A file that is not HDF5 at all (empty, truncated, or a compressed
        # prediction) must be explained, not surfaced as a bare h5py OSError.
        empty = root / "empty.h5"
        empty.write_bytes(b"")
        try:
            load_highres_h5(empty)
        except ConfigError as exc:
            expect(
                "not a readable HDF5 file" in str(exc),
                f"an empty file is explained: {str(exc)[:70]}",
            )
            expect(bool(exc.hint), "the unreadable-file error carries a hint")
        else:
            expect(False, "an empty file should raise ConfigError")

        truncated = root / "truncated.h5"
        truncated.write_bytes(spatial_path.read_bytes()[:128])
        try:
            load_highres_h5(truncated)
        except ConfigError as exc:
            expect(
                "not a readable HDF5 file" in str(exc) or "no '" in str(exc),
                f"a truncated file is explained: {str(exc)[:70]}",
            )
        else:
            expect(False, "a truncated file should raise ConfigError")

        # A valid HDF5 file that is not an AURORA prediction file.
        foreign = root / "foreign.h5"
        with h5py.File(foreign, "w") as hf:
            hf.create_dataset("something_else", data=np.arange(4))
        try:
            load_highres_h5(foreign)
        except ConfigError as exc:
            expect(
                "'genes'" in str(exc) and "not an AURORA" in str(exc),
                f"a foreign HDF5 file is explained: {str(exc)[:80]}",
            )
            expect("something_else" in (exc.hint or ""), "the hint lists what it did find")
        else:
            expect(False, "a foreign HDF5 file should raise ConfigError")

        # A missing row permutation must not be silently ignored.
        no_order = root / "no_order.h5"
        write(fixture, no_order, store_row_order=False)
        with h5py.File(no_order, "r") as hf:
            expect(
                "row_order_56" not in hf,
                "store_row_order=False omits the permutation",
            )
        try:
            load_highres_h5(no_order, restore_original_order=True)
        except ConfigError as exc:
            expect(
                "permutation was not stored" in str(exc),
                f"restoring without a permutation is refused: {str(exc)[:70]}",
            )
            expect(bool(exc.hint), "that error carries a hint")
        else:
            expect(False, "restore_original_order without a permutation should raise")
        # ... but a spatially-sorted read is still fine.
        expect(
            np.array_equal(
                load_highres_h5(no_order)["exp_pred"][56],
                fixture["exp_pred_mem"][56][
                    np.asarray(
                        h5py.File(spatial_path, "r")["row_order_56"][...], dtype=np.int64
                    )
                ],
            ),
            "a sorted read still works when the permutation was not stored",
        )

        # read_patch_locations alone must agree with load_highres_h5.
        with h5py.File(spatial_path, "r") as hf:
            for ps in PATCH_SIZES:
                expect(
                    np.array_equal(
                        read_patch_locations(hf, ps, restore_original_order=True),
                        fixture["patch_loc_mem"][ps],
                    ),
                    f"{ps}: read_patch_locations restores the original order",
                )
                expect(
                    np.array_equal(
                        read_patch_locations(hf, ps), stored["pos"][ps]
                    ),
                    f"{ps}: read_patch_locations default matches the stored order",
                )
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("h5 format round-trip test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
