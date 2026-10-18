"""Apply HistoSweep tissue-quality filtering to AURORA rescaled WSIs.

Intended to run *before* predict_highres.py.  For every sample in the predict
sample sheet it sweeps the already-rescaled H&E image

    <interim>/UNI_multiscale_patches/wsis_rescaled/<sample>_rescaled.tif

with the HistoSweep pipeline (density + texture + ratio filtering) and writes
an AURORA-compatible tissue mask as

    <interim>/UNI_multiscale_patches/wsis_rescaled/<sample>_HistoSweep_mask_rescaled.tif

HistoSweep pads the image on the right and the bottom; this script crops that
padding off the produced mask so it stays pixel-aligned with the original
<sample>_rescaled.tif (the mask has the exact same shape and the same
1 = tissue / 0 = background convention that feature extraction expects).

The ``UNI_multiscale_patches`` folder name shown below is configurable with
``--patches_dir_name`` (default ``UNI_multiscale_patches``) and must match the
folder used by AURORA rescale/extract and by the predict step.

HistoSweep functions are imported directly (no subprocesses), so the hardcoded
cluster paths inside HistoSweep_single.py are never used and no temporary
full-size image copies are made.

After all samples finish, the HistoSweep native outputs under
<pred_path>/HistoSweep are removed (pass --keep-native-outputs to retain them);
the per-sample AURORA masks in wsis_rescaled/ are kept.

Examples
--------
# All samples in the same sheet used by predict_highres
python run_HistoSweep_pred.py -p ./ -i AURORA_interim -t AURORA_pred \
    --sample_sheet Test_samples.csv

# A single selected sample
python run_HistoSweep_pred.py -p ./ --samples TENX23

# Recompute an existing sample with a looser density threshold
python run_HistoSweep_pred.py -p ./ --samples TENX23 --density_thresh 150 --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
DEFAULT_HISTOSWEEP_DIR = _HERE / "HistoSweep"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run HistoSweep on AURORA rescaled WSIs and write per-sample "
            "AURORA-compatible tissue masks."
        )
    )
    # Mirror the path arguments of predict_highres.py
    parser.add_argument("--project_path", "-p", type=str, default="./",
                        help="Main directory path (same as predict_highres).")
    parser.add_argument("--interim_path", "-i", type=str, default="AURORA_interim",
                        help="Interim data directory (same as predict_highres).")
    parser.add_argument("--patches_dir_name", type=str, default="UNI_multiscale_patches",
                        help="Name of the subdirectory under --interim_path that holds "
                             "the rescaled WSIs and HistoSweep masks "
                             "(default: UNI_multiscale_patches).")
    parser.add_argument("--pred_path", "-t", type=str, default="AURORA_pred",
                        help="Directory for HistoSweep native outputs (same as predict_highres).")
    parser.add_argument("--sample_sheet", type=str, default=None,
                        help="CSV with a 'Sample_name' column (same one passed to predict_highres).")
    parser.add_argument("--samples", nargs="+", type=str, default=None,
                        help="Optional explicit sample names; defaults to all rows in --sample_sheet.")
    parser.add_argument("--histosweep_path", type=str, default=str(DEFAULT_HISTOSWEEP_DIR),
                        help="Path to the HistoSweep package directory.")

    # HistoSweep parameters (defaults mirror HistoSweep/run_HistoSweep_single.py)
    parser.add_argument("--pixel_size", type=float, default=0.5,
                        help="Target pixel size in microns (record only; the input is already rescaled).")
    parser.add_argument("--density_thresh", type=float, default=100,
                        help="Density threshold; increase to remove more artifacts (e.g. fiducials).")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Size of one square super-pixel used throughout processing.")
    parser.add_argument("--min_size", type=int, default=10,
                        help="Minimum super-pixel cluster kept by background cleaning.")
    parser.add_argument("--clean_background", dest="clean_background",
                        action="store_false", default=True,
                        help="Disable cleaning of isolated debris and small specs outside tissue.")

    parser.add_argument("--force", action="store_true",
                        help="Recompute even if <sample>_HistoSweep_mask_rescaled.tif already exists.")
    parser.add_argument("--mask-suffix", type=str, default="HistoSweep",
                        help="Suffix of the generated mask file name (before '_mask_rescaled.tif').")
    parser.add_argument("--keep-native-outputs", action="store_true",
                        help="Keep HistoSweep's native outputs (mask.png, mask-small.png, plots) "
                             "under <pred_path>/HistoSweep/. By default they are removed after "
                             "all samples finish.")
    return parser


def _load_rgb(path: Path) -> np.ndarray:
    """Read an image as an HxWx3 uint8 array (handles gray and RGBA tifs)."""
    import tifffile

    img = np.asarray(tifffile.imread(str(path)))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    elif img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape {img.shape} for {path}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _force_utf8_stdio() -> None:
    """Reconfigure stdio to UTF-8 so HistoSweep's emoji prints don't crash.

    HistoSweep prints characters such as \u2705 (\N{WHITE HEAVY CHECK MARK}) and
    \U0001F389 in several places.  On Windows with a GBK/cp936 locale, writing
    those to the default console encoding raises UnicodeEncodeError and aborts
    the run.  Reconfiguring the streams to UTF-8 (with lossy replacement for
    any truly unrepresentable byte) makes the pipeline robust on any locale.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _ensure_pyvips_stub() -> None:
    """Make HistoSweep's ``from utils import ...`` importable without pyvips.

    HistoSweep's ``utils.py`` does ``import pyvips`` at module load, but pyvips
    is only actually used by its rescale/preprocess helpers (which this script
    deliberately bypasses).  If pyvips is not installed, register a minimal stub
    module so the HistoSweep metric/mask functions import cleanly.  The stub's
    ``Image`` attribute stays None - it is only referenced at call time inside
    functions this script never invokes.
    """
    try:
        import pyvips  # noqa: F401
    except ImportError:
        pass
    else:
        return

    import types

    stub = types.ModuleType("pyvips")
    stub.Image = None
    sys.modules.setdefault("pyvips", stub)


def _pad_right_bottom(img: np.ndarray, patch_size: int, fill: int = 255) -> np.ndarray:
    """Pad on the right/bottom so H and W are multiples of patch_size (like HistoSweep)."""
    h, w = img.shape[:2]
    pad_h = (-h) % patch_size
    pad_w = (-w) % patch_size
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(
        img,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=fill,
    )


def run_histosweep_for_sample(
    sample: str,
    image_path: Path,
    out_dir: Path,
    mask_path: Path,
    *,
    patch_size: int,
    density_thresh: float,
    clean_background: bool,
    min_size: int,
    pixel_size: float,
) -> None:
    """Run the HistoSweep pipeline on one sample and save the AURORA mask."""
    import tifffile
    from PIL import Image

    # HistoSweep modules are imported lazily and must resolve against the
    # HistoSweep package directory (inserted at sys.path[0] in main()).
    from computeMetrics import compute_metrics_fast
    from densityFiltering import compute_low_density_mask
    from textureAnalysis import run_texture_analysis_optimized
    from ratioFiltering import run_ratio_filtering
    from generateMask import generate_final_mask
    from saveParameters import saveParams

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== HistoSweep: {sample} ===")
    image = _load_rgb(image_path)
    orig_h, orig_w = image.shape[:2]
    image_padded = _pad_right_bottom(image, patch_size)
    if image_padded.shape[:2] != (orig_h, orig_w):
        print(f"    Padded {image.shape[:2]} -> {image_padded.shape[:2]} (right/bottom)")
    del image

    # 1) Per-super-pixel metrics (std, z_v, ratio) on the 16x16 grid
    he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = (
        compute_metrics_fast(image_padded, patch_size=patch_size)
    )

    # 2) Density filtering
    mask1_lowdensity = compute_low_density_mask(
        z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh
    )

    # 3) Texture analysis (GLCM + GMM, removes artifacts such as fiducials)
    mask1_updated = run_texture_analysis_optimized(
        prefix=sample,
        image=image_padded,
        tissue_mask=mask1_lowdensity,
        output_dir=str(out_dir),
        patch_size=patch_size,
        glcm_levels=64,
    )

    # 4) Ratio filtering (Otsu on normalized std/mean ratio)
    mask2_lowratio, otsu_thresh = run_ratio_filtering(ratio_norm_, mask1_updated)
    print(f"    Otsu ratio threshold: {otsu_thresh:.4f}")

    # Record parameters and produce HistoSweep's native outputs
    # (mask.png, mask-small.png, texture-analysis plots) for visual inspection.
    saveParams(
        sample, str(out_dir), False, True, pixel_size, density_thresh,
        clean_background, min_size, patch_size, pixel_size,
    )
    generate_final_mask(
        prefix=sample,
        he=image_padded,
        mask1_updated=mask1_updated,
        mask2=mask2_lowratio,
        output_dir=str(out_dir),
        clean_background=clean_background,
        super_pixel_size=patch_size,
        minSize=min_size,
    )

    # Rebuild the full-resolution mask from HistoSweep's (tiny) grid-level PNG,
    # crop the right/bottom padding, and convert to AURORA's 0/1 convention
    # (1 = tissue kept, 0 = filtered background) matching <sample>_rescaled.tif.
    mask_small = np.asarray(Image.open(out_dir / "mask-small.png"))
    tissue_grid = mask_small > 0
    mask_full = np.kron(
        tissue_grid.astype(np.uint8),
        np.ones((patch_size, patch_size), dtype=np.uint8),
    )
    aurora_mask = mask_full[:orig_h, :orig_w].astype(np.uint8)

    # Sanity check: the mask must be pixel-aligned with the original image.
    if aurora_mask.shape != (orig_h, orig_w):
        raise RuntimeError(
            f"Mask shape {aurora_mask.shape} != image shape {(orig_h, orig_w)} for {sample}"
        )

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    bigtiff = orig_h > 40000 or orig_w > 60000
    tifffile.imwrite(str(mask_path), aurora_mask, compression="zlib", bigtiff=bigtiff)

    # Verify round-trip
    check = np.asarray(tifffile.imread(str(mask_path)))
    if check.shape != (orig_h, orig_w) or not np.array_equal(check, aurora_mask):
        raise RuntimeError(f"Mask round-trip verification failed for {sample}")

    print(f"    Saved mask: {mask_path}")
    print(f"    Tissue fraction: {float(aurora_mask.mean()):.3%} "
          f"({int(aurora_mask.sum())} of {aurora_mask.size} pixels)")


def _dir_size_mb(path: Path) -> float:
    """Total size in MB of every file under ``path`` (0.0 if it does not exist)."""
    total = 0
    if path.is_dir():
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    return total / 1e6


def main(argv=None) -> int:
    _force_utf8_stdio()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sample_sheet is None and args.samples is None:
        parser.error("either --sample_sheet or --samples must be provided")
    if args.patch_size <= 0:
        parser.error("--patch_size must be positive")
    if args.density_thresh <= 0:
        parser.error("--density_thresh must be positive")
    if args.min_size <= 0:
        parser.error("--min_size must be positive")

    if args.samples is not None:
        samples = list(args.samples)
    else:
        import pandas as pd

        meta = pd.read_csv(os.path.join(args.project_path, args.sample_sheet), header=0)
        if "Sample_name" not in meta.columns:
            raise ValueError(
                f"--sample_sheet must contain a 'Sample_name' column; got {list(meta.columns)}"
            )
        samples = list(meta["Sample_name"].values)

    histosweep_dir = Path(args.histosweep_path).resolve()
    if not histosweep_dir.is_dir():
        raise FileNotFoundError(f"HistoSweep directory not found: {histosweep_dir}")
    sys.path.insert(0, str(histosweep_dir))
    # HistoSweep's utils.py imports pyvips unconditionally; make that import
    # succeed even when pyvips is not installed (rescaling/preprocessing that
    # would need it are bypassed by this script).
    _ensure_pyvips_stub()

    wsis_dir = (
        Path(args.project_path)
        / args.interim_path
        / args.patches_dir_name
        / "wsis_rescaled"
    )
    if not wsis_dir.is_dir():
        raise FileNotFoundError(
            f"Rescaled WSI directory not found: {wsis_dir}. Run AURORA rescale.py first."
        )

    native_out_root = Path(args.project_path) / args.pred_path / "HistoSweep"
    print(f"HistoSweep package : {histosweep_dir}")
    print(f"Samples to process : {len(samples)}")
    print(f"Rescaled WSI dir   : {wsis_dir}")
    print(f"Native outputs dir : {native_out_root}")

    failed = []
    for sample in samples:
        image_path = wsis_dir / f"{sample}_rescaled.tif"
        if not image_path.exists():
            print(f"\n[skip] {sample}: no rescaled image at {image_path}")
            failed.append((sample, "missing rescaled image"))
            continue

        mask_path = wsis_dir / f"{sample}_{args.mask_suffix}_mask_rescaled.tif"
        if mask_path.exists() and not args.force:
            print(f"\n[skip] {sample}: {mask_path.name} already exists (use --force to recompute)")
            continue

        out_dir = native_out_root / sample
        try:
            run_histosweep_for_sample(
                sample,
                image_path,
                out_dir,
                mask_path,
                patch_size=args.patch_size,
                density_thresh=args.density_thresh,
                clean_background=args.clean_background,
                min_size=args.min_size,
                pixel_size=args.pixel_size,
            )
        except Exception as exc:  # noqa: BLE001 - keep sweeping the other samples
            print(f"\n[error] {sample}: {exc!r}")
            failed.append((sample, repr(exc)))

    print("\n===== HistoSweep summary =====")
    for sample in samples:
        mask_path = wsis_dir / f"{sample}_{args.mask_suffix}_mask_rescaled.tif"
        print(f"  {sample}: {'OK' if mask_path.exists() else 'FAILED'}")

    # The per-sample AURORA masks in wsis_rescaled/ are the deliverable; the
    # HistoSweep native outputs are only intermediates, so clean them up once
    # all samples have finished (opt out with --keep-native-outputs).
    if not args.keep_native_outputs:
        size_mb = _dir_size_mb(native_out_root)
        shutil.rmtree(native_out_root, ignore_errors=True)
        print(f"\nCleaned up HistoSweep native outputs: {native_out_root} "
              f"({size_mb:.2f} MB removed)")

    if failed:
        print("\nFailures:")
        for sample, reason in failed:
            print(f"  {sample}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
