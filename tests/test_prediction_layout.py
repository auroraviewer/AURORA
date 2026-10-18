#!/usr/bin/env python
"""Regression test for plotting: the layout code/plot.py can actually read.

``code/plot.py`` reads the legacy 3-D array layout and flips
``patches_loc_before_pad`` a second time before treating column 0 as ``y``.  The
modern layout that ``predict_lowres`` writes by default is therefore unreadable
to it, and an earlier revision stored ``patches_loc_before_pad`` in ``(y, x)``
order, which silently transposed every figure.

This test builds a synthetic modern prediction, projects it with
:func:`aurora.predictions.to_legacy` and then **replays code/plot.py's own
indexing** against the result -- including the coordinate flip -- so a wrong
orientation or a wrong array shape fails here instead of producing a mirrored
plot.

Needs NumPy only.

    python tests/test_prediction_layout.py
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from aurora.paths import RunLayout  # noqa: E402
from aurora.plotting import build_view, legacy_view_root  # noqa: E402
from aurora.predictions import (  # noqa: E402
    LAYOUT_LEGACY,
    LAYOUT_MODERN,
    choose_resolution,
    load_prediction,
    to_legacy,
    write_legacy_view,
)

RESOLUTIONS = [3584, 896, 224]
GENES = ["TMSB4X", "SFTPB", "CD74"]
CELL_TYPES = ["B_cell", "Epithelial", "T_cell"]

# True top-left corners of the 224 px patches, in (x, y).
TRUE_XY = np.array([[0, 0], [224, 0], [0, 224], [224, 224]], dtype=np.int64)

failures: list[str] = []


def expect(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def modern_payload() -> dict:
    """What predict_lowres writes with the default --output-format modern."""
    spots = len(TRUE_XY)
    exp = {
        3584: np.arange(spots * len(GENES), dtype=np.float32).reshape(spots, len(GENES)),
        896: np.arange(spots * len(GENES), dtype=np.float32).reshape(spots, len(GENES)) + 100,
        224: np.arange(spots * len(GENES), dtype=np.float32).reshape(spots, len(GENES)) + 200,
    }
    log_prop = {
        resolution: np.log(
            np.full((spots, len(CELL_TYPES)), 0.25, dtype=np.float32)
            + np.arange(len(CELL_TYPES), dtype=np.float32)
        )
        for resolution in RESOLUTIONS
    }
    return {
        "exp_pred": exp,
        "log_prop_pred": log_prop,
        # 'pos' is (y, x), matching code/predict.py
        "pos": {r: TRUE_XY[:, ::-1].copy() for r in RESOLUTIONS},
        # 'patches_loc_before_pad' must be (x, y), which is what code/plot.py expects
        "patches_loc_before_pad": {r: TRUE_XY.copy() for r in RESOLUTIONS},
        "genes": GENES,
        # code/predict.py stores cell types as an array (`.values.squeeze()`).
        "cell_types": np.asarray(CELL_TYPES),
        "resolutions": RESOLUTIONS,
    }


def legacy_payload() -> dict:
    """What predict_lowres writes with --output-format legacy."""
    spots = len(TRUE_XY)
    return {
        "exp_pred": np.arange(spots * len(RESOLUTIONS) * len(GENES), dtype=np.float32).reshape(
            spots, len(RESOLUTIONS), len(GENES)
        ),
        "prop_pred": np.full((spots, len(RESOLUTIONS), len(CELL_TYPES)), 0.3, dtype=np.float32),
        "patches_loc_before_pad": {r: TRUE_XY.copy() for r in RESOLUTIONS},
        "genes": GENES,
        "cell_types": np.asarray(CELL_TYPES),
        "resolutions": RESOLUTIONS,
    }


def replay_plot_py(payload: dict, resolution: int) -> list[tuple[int, int]]:
    """code/plot.py, lines 69-131, reduced to the calls it makes.

    Exercises **both** branches.  The gene branch and the cell-type branch index
    their label lists differently -- ``results['genes'].index(gene)`` against
    ``results['cell_types'].tolist().index(celltype)`` -- so a payload that plots
    genes happily still crashes on cell types, which is exactly what happened.

    Returns ``(x_start, y_start)`` per patch, after plot.py's own flip.
    """
    patch_loc_before_pad = payload["patches_loc_before_pad"][resolution] // 10
    patch_loc_before_pad = patch_loc_before_pad[:, [1, 0]]
    if resolution not in payload["resolutions"]:
        raise AssertionError(f"{resolution} not in {payload['resolutions']}")
    patch_size_index = payload["resolutions"].index(resolution)

    # --- gene branch (plot.py lines 77-103) -------------------------------- #
    pred = payload["exp_pred"]
    for gene in GENES:
        if gene not in payload["genes"]:
            raise AssertionError(f"gene {gene} missing from the payload")
        gene_idx = payload["genes"].index(gene)  # needs a list, not an array
        pred_use = pred[:, patch_size_index, gene_idx]
        pred_use = pred_use / pred_use.max()
        assert pred_use.shape[0] == len(patch_loc_before_pad), (
            f"{pred_use.shape[0]} predictions for {len(patch_loc_before_pad)} positions"
        )

    # --- cell-type branch (plot.py lines 104-131) -------------------------- #
    prop = np.exp(payload["prop_pred"])
    for celltype in CELL_TYPES:
        if celltype not in payload["cell_types"]:
            raise AssertionError(f"cell type {celltype} missing from the payload")
        # This is the line that failed: it requires cell_types to be an ARRAY.
        celltype_idx = payload["cell_types"].tolist().index(celltype)
        prop_use = prop[:, patch_size_index, celltype_idx]
        assert prop_use.shape[0] == len(patch_loc_before_pad)

    coordinates = []
    for i in range(len(patch_loc_before_pad)):
        loc_patch = patch_loc_before_pad[i]
        coordinates.append((max(int(loc_patch[1]), 0), max(int(loc_patch[0]), 0)))
    return coordinates


def main() -> int:
    root = REPO_ROOT / f".pred_layout_{os.getpid()}"
    root.mkdir()
    try:
        modern_path = root / "TCGA-X-pred.pickle"
        with open(modern_path, "wb") as handle:
            pickle.dump(modern_payload(), handle)
        legacy_path = root / "TCGA-Y-pred.pickle"
        with open(legacy_path, "wb") as handle:
            pickle.dump(legacy_payload(), handle)

        # (x, y) divided by 10, which is what code/plot.py works in.
        expected = [(int(x) // 10, int(y) // 10) for x, y in TRUE_XY]

        # ---------------------------------------------------------------- 1
        print("1. layouts are detected")
        modern = load_prediction(modern_path)
        legacy = load_prediction(legacy_path)
        expect(modern.layout == LAYOUT_MODERN, f"modern detected: {modern.layout}")
        expect(legacy.layout == LAYOUT_LEGACY, f"legacy detected: {legacy.layout}")
        expect(modern.resolutions == RESOLUTIONS, f"resolutions: {modern.resolutions}")
        expect(modern.genes == GENES, "genes carried over")

        # ---------------------------------------------------------------- 2
        print("2. positions are normalised to (x, y) from either key")
        expect(
            np.array_equal(modern.positions[224], TRUE_XY),
            "modern 'patches_loc_before_pad' used as-is",
        )
        expect(
            np.array_equal(legacy.positions[224], TRUE_XY),
            "legacy 'patches_loc_before_pad' used as-is",
        )
        # 'pos' alone is (y, x) and must be flipped.
        only_pos = modern_payload()
        del only_pos["patches_loc_before_pad"]
        only_pos_path = root / "TCGA-Z-pred.pickle"
        with open(only_pos_path, "wb") as handle:
            pickle.dump(only_pos, handle)
        flipped = load_prediction(only_pos_path)
        expect(
            np.array_equal(flipped.positions[224], TRUE_XY),
            f"'pos' flipped to (x, y): {flipped.positions[224].tolist()}",
        )
        expect(bool(flipped.warnings), "the flip is reported as a warning")

        # ---------------------------------------------------------------- 2b
        print("2b. a file written before the orientation fix is corrected")
        # The buggy writer stored the SAME (y, x) array under both keys.
        stale = modern_payload()
        stale["patches_loc_before_pad"] = {r: TRUE_XY[:, ::-1].copy() for r in RESOLUTIONS}
        stale_path = root / "TCGA-STALE-pred.pickle"
        with open(stale_path, "wb") as handle:
            pickle.dump(stale, handle)
        recovered = load_prediction(stale_path)
        expect(
            np.array_equal(recovered.positions[224], TRUE_XY),
            f"stale (y, x) file recovers (x, y): {recovered.positions[224].tolist()}",
        )
        expect(
            any("orientation" in w for w in recovered.warnings),
            f"the correction is reported: {recovered.warnings}",
        )
        recovered_coords = replay_plot_py(to_legacy(recovered, 224), 224)
        expect(
            recovered_coords == expected,
            f"and plots the right way round: {recovered_coords}",
        )

        # A correctly written file must NOT be flipped.
        fresh = load_prediction(modern_path)
        expect(
            np.array_equal(fresh.positions[224], TRUE_XY),
            "a correctly written file is left alone",
        )
        expect(
            not any("orientation" in w for w in fresh.warnings),
            f"and reports no correction: {fresh.warnings}",
        )

        # Disagreeing in an unexplained way must warn, not silently flip.
        odd = modern_payload()
        odd["patches_loc_before_pad"] = {r: (TRUE_XY + 7).copy() for r in RESOLUTIONS}
        odd_path = root / "TCGA-ODD-pred.pickle"
        with open(odd_path, "wb") as handle:
            pickle.dump(odd, handle)
        odd_prediction = load_prediction(odd_path)
        expect(
            any("neither orientation" in w for w in odd_prediction.warnings),
            f"an unclassifiable file warns: {odd_prediction.warnings}",
        )

        # ---------------------------------------------------------------- 3
        print("3. the modern file cannot be read by code/plot.py as-is")
        try:
            replay_plot_py(modern_payload(), 224)
        except (TypeError, KeyError, IndexError) as exc:
            expect(True, f"as expected, it fails: {type(exc).__name__}")
        else:
            expect(False, "a raw modern payload should not be plot-able")

        # ---------------------------------------------------------------- 4
        print("4. conversion produces a payload code/plot.py can index")
        for resolution in RESOLUTIONS:
            converted = to_legacy(modern, resolution)
            expect(
                converted["resolutions"] == [resolution],
                f"{resolution}: resolutions == [{resolution}]",
            )
            expect(
                converted["exp_pred"].shape == (len(TRUE_XY), 1, len(GENES)),
                f"{resolution}: exp_pred shape {converted['exp_pred'].shape}",
            )
            expect(
                converted["prop_pred"].shape == (len(TRUE_XY), 1, len(CELL_TYPES)),
                f"{resolution}: prop_pred shape {converted['prop_pred'].shape}",
            )
            expect(
                np.allclose(converted["prop_pred"], np.exp(modern.log_prop_pred[resolution])),
                f"{resolution}: prop_pred holds probabilities, not logs",
            )
            expect(
                isinstance(converted["genes"], list),
                f"{resolution}: genes is a list (plot.py calls .index())",
            )
            expect(
                hasattr(converted["cell_types"], "tolist"),
                f"{resolution}: cell_types is an array (plot.py calls .tolist()); "
                f"got {type(converted['cell_types']).__name__}",
            )
            coordinates = replay_plot_py(converted, resolution)
            expected = [(int(x) // 10, int(y) // 10) for x, y in TRUE_XY]
            expect(
                coordinates == expected,
                f"{resolution}: code/plot.py resolves {coordinates} (expected {expected})",
            )

        # The transposed orientation must be detectable.
        transposed = to_legacy(modern, 224)
        transposed["patches_loc_before_pad"] = {224: TRUE_XY[:, ::-1].copy()}
        wrong = replay_plot_py(transposed, 224)
        expect(
            wrong != expected,
            "a (y, x) 'patches_loc_before_pad' would produce transposed coordinates",
        )

        # ---------------------------------------------------------------- 5
        print("5. a legacy file is passed through unchanged")
        converted = to_legacy(legacy, 224)
        coordinates = replay_plot_py(converted, 224)
        expect(coordinates == expected, f"legacy 224 px resolves {coordinates}")

        # ---------------------------------------------------------------- 6
        print("6. resolution selection")
        expect(choose_resolution(modern) == 224, "defaults to the smallest resolution")
        expect(choose_resolution(modern, 896) == 896, "explicit resolution honoured")
        try:
            choose_resolution(modern, 999)
        except Exception as exc:  # noqa: BLE001
            expect("999" in str(exc), f"unknown resolution rejected: {exc}")
        else:
            expect(False, "an unknown resolution should be rejected")

        # ---------------------------------------------------------------- 7
        print("7. build_view produces the directory code/plot.py is pointed at")
        project = root / "project"
        run_root = project / "aurora_runs" / "luad_infer"
        source_dir = run_root / "03_predictions" / "lowres" / "pred"
        source_dir.mkdir(parents=True)
        for name in ("S1", "S2"):
            with open(source_dir / f"{name}-pred.pickle", "wb") as handle:
                pickle.dump(modern_payload(), handle)
        (run_root / "tmp").mkdir(parents=True, exist_ok=True)

        layout = RunLayout.from_root(run_root, project)
        view_root, notes = build_view(
            layout, layout.prediction_dir("lowres"), ["S1", "S2"], resolution=224
        )
        view_file = view_root / "pred" / "S1-pred.pickle"
        expect(view_root == legacy_view_root(layout), f"view root is {view_root}")
        for name in ("S1", "S2"):
            target = view_root / "pred" / f"{name}-pred.pickle"
            expect(target.is_file(), f"{name}: view file written")
            with open(target, "rb") as handle:
                payload = pickle.load(handle)
            coordinates = replay_plot_py(payload, 224)
            expect(coordinates == expected, f"{name}: view is plot-able and correctly oriented")
        expect(any("projected" in note for note in notes), f"notes explain the projection: {notes}")

        # Re-running must not redo the work.
        _, notes2 = build_view(
            layout, layout.prediction_dir("lowres"), ["S1", "S2"], resolution=224
        )
        expect(
            all("up to date" in note for note in notes2 if "projected" not in note),
            f"second run reuses the view: {notes2}",
        )

        # ---------------------------------------------------------------- 8
        print("8. a view left by an older converter is rebuilt, not reused")
        # The trap that kept a fixed bug alive: the view file existed and was
        # newer than its source, so an mtime-only check reused it.  A view written
        # by the old converter has NO build record next to it.

        def plant_stale_view() -> None:
            """Reproduce what the old converter left on disk."""
            with open(view_file, "rb") as handle:
                payload = pickle.load(handle)
            payload["cell_types"] = list(payload["cell_types"])  # the old bug
            with open(view_file, "wb") as handle:
                pickle.dump(payload, handle)
            # Newer than the source, so only the build record can save us.
            os.utime(view_file, (time.time() + 60, time.time() + 60))

        def planted_is_buggy() -> bool:
            with open(view_file, "rb") as handle:
                return not hasattr(pickle.load(handle)["cell_types"], "tolist")

        # (a) the user's exact situation: a view with no build record
        plant_stale_view()
        (view_root / "view.json").unlink(missing_ok=True)
        expect(planted_is_buggy(), "the planted view reproduces the old bug")
        _, notes3 = build_view(
            layout, layout.prediction_dir("lowres"), ["S1", "S2"], resolution=224
        )
        expect(
            any("no build record" in note for note in notes3),
            f"a view without a build record forces a rebuild: {notes3}",
        )
        expect(not planted_is_buggy(), "the stale view was overwritten")
        with open(view_file, "rb") as handle:
            replay_plot_py(pickle.load(handle), 224)

        # (b) a build record from an older converter version
        plant_stale_view()
        meta = json.loads((view_root / "view.json").read_text(encoding="utf-8"))
        meta["view_format"] = 1
        (view_root / "view.json").write_text(json.dumps(meta), encoding="utf-8")
        expect(planted_is_buggy(), "the planted view reproduces the old bug again")
        _, notes4 = build_view(
            layout, layout.prediction_dir("lowres"), ["S1", "S2"], resolution=224
        )
        expect(
            any("older converter" in note for note in notes4),
            f"an older view_format forces a rebuild: {notes4}",
        )
        expect(not planted_is_buggy(), "the older-format view was overwritten")

        # (c) the source prediction changed
        source_file = layout.prediction_output_dir("lowres") / "S1-pred.pickle"
        os.utime(source_file, (time.time() + 120, time.time() + 120))
        _, notes5 = build_view(
            layout, layout.prediction_dir("lowres"), ["S1", "S2"], resolution=224
        )
        expect(
            any("projected" in note and "S1" in note for note in notes5),
            f"a newer source invalidates only that sample: {notes5}",
        )
        expect(
            any("up to date" in note and "S2" in note for note in notes5),
            f"and the untouched sample is still reused: {notes5}",
        )

        # (d) changing the plotted resolution must invalidate the view
        _, notes6 = build_view(
            layout, layout.prediction_dir("lowres"), ["S1", "S2"], resolution=896
        )
        expect(
            any("projected" in note for note in notes6),
            f"a different --plot-resolution rebuilds: {notes6}",
        )
        with open(view_file, "rb") as handle:
            expect(
                pickle.load(handle)["resolutions"] == [896],
                "the rebuilt view uses the newly requested resolution",
            )

        # A legacy source is used directly, with no view.
        legacy_source = run_root / "03_predictions" / "legacy"
        (legacy_source / "pred").mkdir(parents=True)
        for name in ("S1",):
            with open(legacy_source / "pred" / f"{name}-pred.pickle", "wb") as handle:
                pickle.dump(legacy_payload(), handle)
        returned, notes3 = build_view(layout, legacy_source, ["S1"], resolution=224)
        expect(returned == legacy_source, f"a legacy source is used directly: {returned}")
        expect(any("already use the legacy layout" in note for note in notes3), f"{notes3}")

        # ---------------------------------------------------------------- 9
        print("9. high-resolution predictions are plottable too")
        # predict_highres names its file '<sample>-highres-pred.pickle', but
        # code/plot.py always opens '<sample>-pred.pickle', so the view renames.
        highres_dir = run_root / "03_predictions" / "highres" / "pred"
        highres_dir.mkdir(parents=True)
        highres_resolutions = [56, 14]
        spots_hr = len(TRUE_XY)
        highres_payload = {
            "exp_pred": {
                r: np.arange(spots_hr * len(GENES), dtype=np.float32).reshape(spots_hr, len(GENES))
                for r in highres_resolutions
            },
            "log_prop_pred": {
                r: np.log(np.full((spots_hr, len(CELL_TYPES)), 0.4, dtype=np.float32))
                for r in highres_resolutions
            },
            "pos": {r: TRUE_XY[:, ::-1].copy() for r in highres_resolutions},
            "patches_loc_before_pad": {r: TRUE_XY[:, ::-1].copy() for r in highres_resolutions},
            "genes": GENES,
            "cell_types": np.asarray(CELL_TYPES),
            "resolutions": highres_resolutions,
        }
        for name in ("S1", "S2"):
            with open(highres_dir / f"{name}-highres-pred.pickle", "wb") as handle:
                pickle.dump(highres_payload, handle)

        hr_root, hr_notes = build_view(
            layout,
            layout.prediction_dir("highres"),
            ["S1", "S2"],
            resolution=56,
            kind="highres",
        )
        expect(
            hr_root != view_root,
            f"the high-resolution view is separate from the low-resolution one: {hr_root}",
        )
        hr_file = hr_root / "pred" / "S1-pred.pickle"
        expect(
            hr_file.is_file(),
            "the high-resolution view uses the filename code/plot.py opens",
        )
        with open(hr_file, "rb") as handle:
            hr_view = pickle.load(handle)
        expect(
            hr_view["resolutions"] == [56],
            f"the high-resolution view records its own resolution: {hr_view['resolutions']}",
        )
        expect(
            hr_view["exp_pred"].shape == (spots_hr, 1, len(GENES)),
            f"high-res exp_pred shape {hr_view['exp_pred'].shape}",
        )
        expect(
            any("highres 56 px" in note for note in hr_notes),
            f"notes name the kind: {hr_notes}",
        )
        # The stale (y, x) alias in the high-res file must be corrected too.
        expect(
            np.array_equal(hr_view["patches_loc_before_pad"][56], TRUE_XY),
            f"high-res positions corrected to (x, y): {hr_view['patches_loc_before_pad'][56].tolist()}",
        )
        # And the whole plot.py contract, at the high-resolution level.
        coordinates = replay_plot_py(hr_view, 56)
        expected_hr = [(int(x) // 10, int(y) // 10) for x, y in TRUE_XY]
        expect(coordinates == expected_hr, f"high-res replays code/plot.py: {coordinates}")

        # The two views must not bleed into each other.  Step 8d last rebuilt the
        # low-resolution view at 896, so that is what it must still hold.
        with open(view_root / "pred" / "S1-pred.pickle", "rb") as handle:
            lowres_resolutions = pickle.load(handle)["resolutions"]
        expect(
            lowres_resolutions == [896],
            f"the low-resolution view is untouched by the high-res run: {lowres_resolutions}",
        )
        expect(
            lowres_resolutions != hr_view["resolutions"],
            "the two views hold different resolutions",
        )

        # ---------------------------------------------------------------- 10
        print("10. an unplottable compressed output is explained, not crashed on")
        (highres_dir / "S1-highres-pred.pickle").unlink()
        (highres_dir / "S1-highres-pred.npz.zst").write_bytes(b"")
        try:
            build_view(
                layout, layout.prediction_dir("highres"), ["S1"], resolution=56, kind="highres"
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc) + (getattr(exc, "hint", "") or "")
            expect(
                "compress-output" in message,
                f"the error explains the format problem: {message[:200]}",
            )
        else:
            expect(False, "a missing plottable file should raise")
        (highres_dir / "S1-highres-pred.npz.zst").unlink()

        # ---------------------------------------------------------------- 11
        print("11. the default HDF5 high-res output is plottable")
        # --uint8-output writes HDF5 by default now, so the plot stage has to read
        # it; otherwise making it the default would have broken plotting.
        from aurora.h5out import write_highres_uint8_h5

        rows, channels = 6, len(GENES)
        rng = np.random.default_rng(0)
        exp = rng.integers(0, 256, size=(rows, channels), dtype=np.uint8)
        prop = rng.integers(0, 256, size=(rows, 3), dtype=np.uint8)
        # (y, x), the order the high-resolution writer stores in patch_loc_<ps>.
        locs = np.array([[0, 0], [56, 0], [0, 56], [56, 56], [28, 28], [28, 84]], dtype=np.int64)
        for name in ("S1", "S2"):
            write_highres_uint8_h5(
                highres_dir / f"{name}-highres-pred.h5",
                gene_list=GENES,
                major_cell_type=np.asarray(CELL_TYPES),
                resolutions=highres_resolutions,
                patch_sizes=[56],
                exp_pred_mem={56: exp},
                prop_pred_mem={56: prop},
                exp_max_mem={56: np.full(channels, 255, dtype=np.float32)},
                prop_max_mem={56: np.full(3, 255, dtype=np.float32)},
                patch_loc_mem={56: locs},
                layout="spatial_gene_delta",
                compression="gzip",  # hdf5plugin/zstd is not installed here
                compression_level=6,
            )
        h5_root, h5_notes = build_view(
            layout,
            layout.prediction_dir("highres"),
            ["S1", "S2"],
            resolution=56,
            kind="highres",
            force=True,
        )
        h5_file = h5_root / "pred" / "S1-pred.pickle"
        expect(h5_file.is_file(), "an .h5 source produced a legacy view")
        with open(h5_file, "rb") as handle:
            from_h5 = pickle.load(handle)
        expect(
            from_h5["exp_pred"].shape == (rows, 1, len(GENES)),
            f"from-h5 exp_pred shape {from_h5['exp_pred'].shape}",
        )
        expect(
            hasattr(from_h5["cell_types"], "tolist"),
            "from-h5 cell_types is an array (code/plot.py calls .tolist())",
        )
        # The values must survive: the reader dequantizes by the per-channel max,
        # so with max=255 the recovered float equals the stored uint8 exactly.
        expect(
            np.allclose(from_h5["exp_pred"][:, 0, :], exp.astype(np.float32)),
            f"HDF5 values round-trip through the view (got {from_h5['exp_pred'][:, 0, :].tolist()})",
        )
        coordinates = replay_plot_py(from_h5, 56)
        # patch_loc is (y, x); code/plot.py resolves (x // downscale, y // downscale).
        expect(
            coordinates == [(int(loc[1]) // 10, int(loc[0]) // 10) for loc in locs],
            f"the HDF5 view is correctly oriented: {coordinates}",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("prediction layout test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
