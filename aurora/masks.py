"""The HistoSweep tissue mask, and how ``predict_highres`` applies it.

``code/run_HistoSweep.py`` sweeps the rescaled H&E image and writes a 0/1 tissue
mask next to it::

    <interim>/UNI_multiscale_patches/wsis_rescaled/<sample>_HistoSweep_mask_rescaled.tif

``predict_highres`` then drops the prediction points that fall outside the tissue:
the filtered rows are zeroed and removed, which shrinks the saved arrays and their
positions together, so the remaining rows stay aligned.

This mirrors ``code/predict_highres_TCGA.py`` (``load_histosweep_mask`` and
``histosweep_keep_flags``), with the axis order spelled out rather than left to a
comment -- see :func:`histosweep_keep_flags`.

Standard library only; NumPy is imported through the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import ConfigError
from .naming import HISTOSWEEP_MASK_SUFFIX, histosweep_mask_path


def mask_path(rescaled_dir: Path, sample: str) -> Path:
    """Where the HistoSweep mask for ``sample`` lives."""
    return histosweep_mask_path(rescaled_dir, sample)


def require_mask(rescaled_dir: Path, sample: str) -> Path:
    """Return the mask path, or raise explaining how to produce it.

    HistoSweep is a mandatory stage, so a missing mask is an error rather than a
    warning: silently predicting without it would quietly change which patches
    are kept.
    """
    path = mask_path(rescaled_dir, sample)
    if path.is_file():
        return path
    raise ConfigError(
        f"the HistoSweep mask for '{sample}' is missing: {path}",
        hint=(
            "HistoSweep is a required step because the high-resolution stage applies "
            "its tissue mask. Run it first:\n"
            "  python -m aurora run --stage histosweep --auto-prerequisites\n"
            "It writes "
            f"'<sample>_{HISTOSWEEP_MASK_SUFFIX}_mask_rescaled.tif' next to the rescaled slide."
        ),
    )


def load_mask(path: Path, np: Any) -> Any:
    """Read a HistoSweep 0/1 mask as an ``(H, W)`` uint8 array.

    A 3-D array is reduced with ``[..., 0]``, exactly as
    ``code/predict_highres_TCGA.py`` does, which handles an ``(H, W, 3)`` RGB
    TIFF.  Note that this also flattens an ``(planes, H, W)`` stack to its first
    plane -- HistoSweep writes 2-D masks, so anything else is unexpected, but the
    reduction is the reference behaviour and is kept for parity.
    """
    import tifffile

    mask = np.asarray(tifffile.imread(str(path)))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ConfigError(
            f"{path} is not a 2-D tissue mask (shape {mask.shape})",
            hint="Re-run the histosweep stage; the mask should be a single 2-D plane.",
        )
    return (mask > 0).astype(np.uint8)


def histosweep_keep_flags(positions_yx: Any, mask: Any, patch_size: int, np: Any) -> Any:
    """Boolean keep-flag per point against a HistoSweep tissue mask.

    ``positions_yx`` is ``(N, 2)`` with **column 0 = y (mask row)** and
    **column 1 = x (mask column)**.

    The caller passes the same post-flip dictionary that the prediction file
    stores under ``pos``, which is exactly this order; passing the ``(x, y)``
    spelling would filter the transpose, so the two must not be mixed up.  Both
    the point and the mask are snapped onto the ``patch_size`` super-pixel grid,
    where the mask is constant, so small alignment offsets do not matter.
    Out-of-bounds points are treated as filtered.
    """
    positions = np.asarray(positions_yx)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ConfigError(
            f"expected an (N, 2) position array, got shape {positions.shape}",
            hint="positions must be in (y, x) order for the HistoSweep mask.",
        )
    rows = positions[:, 0].astype(np.int64)
    cols = positions[:, 1].astype(np.int64)
    height, width = mask.shape
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    keep = np.zeros(positions.shape[0], dtype=bool)
    if in_bounds.any():
        snapped_y = np.clip((rows[in_bounds] // patch_size) * patch_size, 0, height - 1)
        snapped_x = np.clip((cols[in_bounds] // patch_size) * patch_size, 0, width - 1)
        keep[in_bounds] = mask[snapped_y, snapped_x] > 0
    return keep


def apply_mask(
    output_dedup: dict,
    prop_dedup: dict,
    positions_dedup: dict,
    mask: Any,
    patch_sizes,
    patch_size_for_mask: int,
    np: Any,
) -> int:
    """Zero and drop the points outside the mask.

    Every dict is updated in place, so the three arrays for a patch size always
    have the same row count afterwards.  Returns the number of points dropped, so
    the run can report how much of the prediction was filtered out.
    """
    dropped = 0
    for patch_size in patch_sizes:
        if patch_size not in output_dedup:
            continue
        keep = histosweep_keep_flags(
            positions_dedup[patch_size], mask, patch_size=patch_size_for_mask, np=np
        )
        if keep.all():
            continue
        # Zero first, then drop the rows that became entirely zero.  A point is
        # kept when at least one gene survives, which is what the TCGA script
        # does; positions are filtered by the same mask so they stay aligned.
        exp = np.array(output_dedup[patch_size], copy=True)
        prop = np.array(prop_dedup[patch_size], copy=True)
        exp[~keep] = 0.0
        prop[~keep] = 0.0
        row_keep = exp.any(axis=1)
        dropped += int((~row_keep).sum())
        output_dedup[patch_size] = exp[row_keep]
        prop_dedup[patch_size] = prop[row_keep]
        positions_dedup[patch_size] = np.asarray(positions_dedup[patch_size])[row_keep]
    return dropped
