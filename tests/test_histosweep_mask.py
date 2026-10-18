#!/usr/bin/env python
"""Tests for the HistoSweep tissue mask and how predict_highres applies it.

HistoSweep is a mandatory stage: ``predict_highres`` drops the points its mask
marks as background / low quality.  Two things can go quietly wrong:

* the **axis order**.  The mask is ``(row=y, col=x)`` while AURORA also has an
  ``(x, y)`` spelling of the same positions.  Mixing them filters the transpose,
  which produces a plausible-looking but wrong result -- so the asymmetry is
  tested explicitly with a mask that is tissue on one side only.
* **row alignment**.  Filtered points are dropped from the predictions *and*
  their positions together; if only one is filtered the arrays drift apart and
  every later point is plotted at the wrong place.

Needs NumPy and tifffile (to write the mask fixture).

    python tests/test_histosweep_mask.py
"""

from __future__ import annotations

import os
import shutil
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from aurora.errors import ConfigError  # noqa: E402
from aurora.masks import (  # noqa: E402
    apply_mask,
    histosweep_keep_flags,
    load_mask,
    mask_path,
    require_mask,
)

failures: list[str] = []

#: Real tifffile when available; otherwise a stub, so ``load_mask``'s own shape
#: handling is still exercised.  The real file read is covered by the stage on a
#: machine that has it.
try:
    import tifffile as _tifffile  # noqa: F401
except ImportError:
    _tifffile = None


class _StubTiffFile:
    store: dict = {}

    def __init__(self, path, mode: str = "r") -> None:
        self.path = str(path)

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def write(self, array) -> None:
        _StubTiffFile.store[self.path] = np.asarray(array)


def install_stub() -> bool:
    """Make ``import tifffile`` work.  Returns True when a stub was installed."""
    if _tifffile is not None:
        return False
    module = types.ModuleType("tifffile")
    module.TiffFile = _StubTiffFile
    module.imread = lambda path: _StubTiffFile.store[str(path)]
    sys.modules["tifffile"] = module
    return True


def write_tiff(path: Path, array) -> None:
    # tifffile dropped ``TiffFile(path, 'w')`` in newer releases, so prefer the
    # stable writer and fall back to the stub only when tifffile is unavailable.
    if _tifffile is not None:
        try:
            _tifffile.imwrite(str(path), array)
            return
        except Exception:
            pass
        tiff_file = getattr(_tifffile, "TiffFile", None)
        if tiff_file is not None:
            try:
                with tiff_file(str(path), "w") as tif:
                    tif.write(array)
                return
            except Exception:
                pass
    _StubTiffFile(str(path)).write(array)


def expect(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def main() -> int:
    stubbed = install_stub()
    root = REPO_ROOT / f".histomask_{os.getpid()}"
    rescaled = root / "02_features" / "UNI_multiscale_patches" / "wsis_rescaled"
    rescaled.mkdir(parents=True)
    sample = "S1"
    print(f"tifffile: {'stub' if stubbed else 'real'}")

    try:
        # ---------------------------------------------------------------- 1
        print("1. the mask path is the one run_HistoSweep.py writes")
        expected = rescaled / f"{sample}_HistoSweep_mask_rescaled.tif"
        expect(mask_path(rescaled, sample) == expected, f"mask_path -> {mask_path(rescaled, sample).name}")

        # ---------------------------------------------------------------- 2
        print("2. a missing mask is an error, not a warning")
        try:
            require_mask(rescaled, sample)
        except ConfigError as exc:
            expect("missing" in str(exc), f"explains the problem: {str(exc)[:80]}")
            expect(
                "histosweep" in (exc.hint or ""),
                "the hint says to run the histosweep stage",
            )
        else:
            expect(False, "a missing mask should raise")

        # ---------------------------------------------------------------- 3
        print("3. an asymmetric mask is filtered on the right axis")
        # A square 60x60 mask with tissue only where COLUMN < 25.  Square so that
        # transposing a point stays in bounds -- otherwise the (x, y) spelling
        # would just fall outside the mask and look the same.
        height = width = 60
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[:, :25] = 1
        write_tiff(root / "m.tif", mask)
        loaded = load_mask(root / "m.tif", np)
        expect(loaded.shape == (height, width), f"mask shape {loaded.shape}")
        expect(set(np.unique(loaded).tolist()) <= {0, 1}, "mask is 0/1")

        # (y, x) positions: column 10 is tissue, column 40 is not, at two rows.
        positions = np.array([[10, 10], [10, 40], [40, 10], [40, 40]], dtype=np.int64)
        keep = histosweep_keep_flags(positions, loaded, patch_size=14, np=np)
        expect(
            keep.tolist() == [True, False, True, False],
            f"kept {keep.tolist()} (expected column 10 kept, column 40 filtered)",
        )

        # Reading the same points as (x, y) filters by row instead of column, which
        # is a different answer -- so a mix-up cannot pass unnoticed.
        transposed = histosweep_keep_flags(positions[:, ::-1], loaded, patch_size=14, np=np)
        expect(
            transposed.tolist() == [True, True, False, False],
            f"the (x, y) spelling filters the transpose: {transposed.tolist()}",
        )
        expect(transposed.tolist() != keep.tolist(), "the two orders are distinguishable")

        # Out-of-bounds points are filtered rather than crashing.
        edge = np.array([[-5, 10], [10, -5], [height + 5, 10], [10, width + 5]], dtype=np.int64)
        expect(
            histosweep_keep_flags(edge, loaded, patch_size=14, np=np).tolist()
            == [False, False, False, False],
            "out-of-bounds points are filtered",
        )

        # ---------------------------------------------------------------- 4
        print("4. dropping points keeps predictions and positions aligned")
        # Patch size 14 snaps to multiples of 14: column 14 is tissue (14 < 25),
        # column 42 is not.
        spots = 3
        output = {
            14: np.array([[1.0, 2.0], [9.0, 9.0], [3.0, 4.0]], dtype=np.float32),
            56: np.array([[5.0, 6.0], [7.0, 8.0], [1.0, 1.0]], dtype=np.float32),
        }
        prop = {
            14: np.array([[0.1], [0.9], [0.3]], dtype=np.float32),
            56: np.array([[0.2], [0.8], [0.4]], dtype=np.float32),
        }
        pos = {
            14: np.array([[14, 14], [14, 42], [42, 14]], dtype=np.int64),
            56: np.array([[28, 14], [28, 42], [56, 14]], dtype=np.int64),
        }
        before = {k: v.copy() for k, v in output.items()}
        dropped = apply_mask(output, prop, pos, loaded, (14, 56), 14, np)

        expect(dropped == 2, f"2 points dropped (one per patch size), got {dropped}")
        for patch_size in (14, 56):
            expect(
                len(output[patch_size]) == 2
                and len(prop[patch_size]) == 2
                and len(pos[patch_size]) == 2,
                f"{patch_size}: all three arrays have 2 rows",
            )
        # The survivors are the two tissue points, in order, and they still line up.
        expect(
            output[14].tolist() == [[1.0, 2.0], [3.0, 4.0]],
            f"14: surviving predictions {output[14].tolist()}",
        )
        expect(
            pos[14].tolist() == [[14, 14], [42, 14]],
            f"14: surviving positions {pos[14].tolist()}",
        )
        expect(
            np.allclose(prop[14], np.array([[0.1], [0.3]], dtype=np.float32)),
            f"14: surviving proportions {prop[14].tolist()}",
        )
        expect(
            output[56].tolist() == [[5.0, 6.0], [1.0, 1.0]],
            f"56: surviving predictions {output[56].tolist()}",
        )
        # The input arrays were not mutated in place.
        expect(
            np.array_equal(output[14], before[14][[0, 2]]),
            "the original arrays were copied, not mutated in place",
        )

        # ---------------------------------------------------------------- 5
        print("5. a fully tissue mask drops nothing")
        full = np.ones((height, width), dtype=np.uint8)
        out2 = {14: np.array([[1.0, 2.0]], dtype=np.float32)}
        prop2 = {14: np.array([[0.5]], dtype=np.float32)}
        pos2 = {14: np.array([[14, 14]], dtype=np.int64)}
        expect(
            apply_mask(out2, prop2, pos2, full, (14,), 14, np) == 0,
            "nothing dropped",
        )
        expect(len(out2[14]) == 1, "the point is still there")

        # ---------------------------------------------------------------- 6
        print("6. shape handling of the mask reader")
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[..., 0] = 1
        write_tiff(root / "rgb.tif", rgb)
        loaded_rgb = load_mask(root / "rgb.tif", np)
        expect(loaded_rgb.shape == (4, 4), f"an RGB mask takes the first channel: {loaded_rgb.shape}")

        write_tiff(root / "fourd.tif", np.zeros((2, 3, 4, 4), dtype=np.uint8))
        try:
            load_mask(root / "fourd.tif", np)
        except ConfigError as exc:
            expect("2-D" in str(exc), f"a 4-D array is rejected: {str(exc)[:70]}")
        else:
            expect(False, "a 4-D array should be rejected")

        try:
            histosweep_keep_flags(np.array([1, 2, 3]), loaded, 14, np)
        except ConfigError as exc:
            expect("(N, 2)" in str(exc), f"bad positions are rejected: {str(exc)[:70]}")
        else:
            expect(False, "a 1-D position array should be rejected")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("histosweep mask test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
