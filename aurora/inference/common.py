"""Shared plumbing for the native prediction stages.

Kept deliberately small: argument parsing, run-context reconstruction, the
stage gate check, and the pickle writers.  The numerical work lives in the
legacy helpers (``train_utils.load_data_pred`` / ``load_data2_pred`` and
``model.AURORA``) exactly as in ``code/predict.py`` and
``code/predict_highres.py``, so predictions are unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .._legacy import ensure_code_on_path
from ..bundle import CombinedCheckpoint
from ..checkpoint import CheckpointAssets, load_assets
from ..deps import require
from ..errors import AuroraError, ConfigError, StagePrerequisiteError
from ..paths import RunLayout
from ..samples import SampleSheet, read_sample_sheet
from ..spec import StageContext
from ..state import RunState, assert_prerequisites

#: Output pickle formats understood by :func:`write_prediction_pickle`.
FORMAT_MODERN = "modern"
FORMAT_LEGACY = "legacy"
OUTPUT_FORMATS = (FORMAT_MODERN, FORMAT_LEGACY)


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by both predictors."""
    parser.add_argument("--run-root", type=str, required=True, help="Run directory (the layout root)")
    parser.add_argument("--model", type=str, required=True, help="Combined AURORA checkpoint (.pth)")
    parser.add_argument("--samples", type=str, required=True, help="Sample sheet, relative to the run root")
    parser.add_argument("--interim-dir", type=str, default="02_features", help="Feature directory, relative to the run root")
    parser.add_argument("--pred-dir", type=str, required=True, help="Output directory, relative to the run root")
    parser.add_argument("--batch-size", type=int, default=100, help="Samples per outer batch")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', 'cuda' or 'cuda:0'")
    parser.add_argument("--num-workers", type=int, default=4, help="Worker processes for the KD-tree helpers")
    parser.add_argument(
        "--uint8-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Quantise the prediction to uint8 (default: on). "
            "Pass --no-uint8-output for full-precision floats."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Recompute samples whose output already exists")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many samples (debugging)")


def add_lowres_arguments(parser: argparse.ArgumentParser) -> None:
    add_run_arguments(parser)
    parser.add_argument("--bulk", type=str, default=None, help="Bulk expression matrix (rows = genes, columns = samples)")
    parser.add_argument("--no-bulk", action="store_true", help="Do not condition on bulk expression (model must not use it)")
    parser.add_argument(
        "--only-224",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep only the finest (224 px) low-resolution level (default: on), matching "
            "code/predict.py --only_224. Pass --no-only-224 to keep every level."
        ),
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default=FORMAT_MODERN,
        choices=list(OUTPUT_FORMATS),
        help=(
            "modern: dict-of-patch-sizes layout written by code/predict.py. "
            "legacy: the 3-D array layout expected by code/plot.py."
        ),
    )


def add_highres_arguments(parser: argparse.ArgumentParser) -> None:
    add_run_arguments(parser)
    parser.add_argument("--lowres-dir", type=str, default="03_predictions/lowres", help="Low-resolution prediction directory")
    parser.add_argument(
        "--lowres-state",
        type=str,
        default="run.json",
        help="Run state file used to prove the low-resolution stage ran first",
    )
    parser.add_argument("--compress-output", action="store_true", help="Write .npz.zst instead of .h5")
    parser.add_argument(
        "--h5-layout",
        type=str,
        default="spatial_gene_delta",
        choices=["spatial_gene_delta", "patch_major"],
        help="HDF5 layout for the uint8 output (default: spatial_gene_delta)",
    )
    parser.add_argument(
        "--h5-compression",
        type=str,
        default="zstd",
        choices=["zstd", "gzip", "none"],
        help="HDF5 filter for the uint8 output (default: zstd; needs hdf5plugin)",
    )
    parser.add_argument(
        "--h5-compression-level",
        type=int,
        default=9,
        help="Level for --h5-compression (default: 9)",
    )
    parser.add_argument(
        "--rescaled-dir",
        type=str,
        default="02_features/UNI_multiscale_patches/wsis_rescaled",
        help="Directory holding the rescaled slides and the HistoSweep tissue masks",
    )
    parser.add_argument(
        "--histosweep-patch-size",
        type=int,
        default=14,
        help=(
            "Super-pixel size used to align prediction points with the HistoSweep "
            "mask; matches code/predict_highres_TCGA.py"
        ),
    )
    parser.add_argument("--skip-prerequisite-check", action="store_true", help=argparse.SUPPRESS)


# --------------------------------------------------------------------------- #
# Device / context
# --------------------------------------------------------------------------- #


def resolve_device(spec: str) -> Any:
    """Turn ``'auto'`` / ``'cpu'`` / ``'cuda:0'`` into a ``torch.device``."""
    torch = require("torch", purpose="running a prediction stage")
    if spec in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError(
            f"--device {spec!r} was requested but CUDA is not available",
            hint="Use --device cpu, or install a CUDA-enabled PyTorch build.",
        )
    return device


def load_run_layout(run_root: str) -> RunLayout:
    root = Path(run_root).expanduser().resolve()
    if not (root / "run.json").is_file():
        raise AuroraError(
            f"{root} does not look like an AURORA run directory (no run.json)",
            hint=(
                "Create the run with:\n"
                "  python -m aurora run --run <name> --samples <samples.csv> --model <combined.pth>"
            ),
        )
    return RunLayout.from_root(root)


def open_checkpoint(path: str) -> CombinedCheckpoint:
    return CombinedCheckpoint.open(path)


def build_stage_context(
    layout: RunLayout,
    checkpoint: CombinedCheckpoint,
    sheet: SampleSheet,
    assets: CheckpointAssets,
    options: Optional[Dict[str, Any]] = None,
) -> StageContext:
    """Reconstruct the context the gate needs, outside the CLI."""
    manifest = checkpoint.manifest
    lowres_params = dict(manifest.stages["lowres"].params) if "lowres" in manifest.stages else {}
    highres_params = dict(manifest.stages["highres"].params) if "highres" in manifest.stages else {}
    for stage, params in (("lowres", lowres_params), ("highres", highres_params)):
        if stage in manifest.stages:
            entry = manifest.stages[stage]
            if entry.patch_sizes and "patch_size" not in params:
                params["patch_size"] = list(entry.patch_sizes)
            if entry.img_emb_dim is not None and "img_emb_dim" not in params:
                params["img_emb_dim"] = int(entry.img_emb_dim)
    del assets
    return StageContext(
        layout=layout,
        samples=sheet,
        lowres_params=lowres_params,
        highres_params=highres_params,
        options=dict(options or {}),
    )


def enforce_prerequisites(
    layout: RunLayout,
    checkpoint: CombinedCheckpoint,
    sheet: SampleSheet,
    assets: CheckpointAssets,
    stage: str,
    options: Optional[Dict[str, Any]] = None,
) -> None:
    """Refuse to run ``stage`` when its prerequisites have not completed.

    This is the direct enforcement of requirement #2, and it also protects users
    who bypass the CLI and call the predictor module by hand.
    """
    state = RunState.load(layout, create=False)
    if checkpoint.model_name and state.model_info.get("model_name") not in (None, "", checkpoint.model_name):
        # A different model in the same run directory: still gated, just note it.
        pass
    ctx = build_stage_context(layout, checkpoint, sheet, assets, options)
    assert_prerequisites(state, ctx, stage)


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def resolve_bulk_matrix(
    checkpoint: CombinedCheckpoint,
    sheet: SampleSheet,
    run_root: Path,
    bulk_argument: Optional[str],
    no_bulk: bool,
) -> Optional[Any]:
    """Locate and load the bulk expression matrix.

    Raises a precise error rather than silently predicting with an all-zero
    bulk vector, which would quietly degrade every prediction.
    """
    if no_bulk:
        return None
    pd = require("pandas", purpose="reading the bulk expression matrix")

    candidates: List[Tuple[str, Any]] = []
    if bulk_argument:
        candidates.append(("--bulk", Path(bulk_argument)))
    default = run_root / "config" / "bulk.csv"
    candidates.append(("run config/bulk.csv", default))
    if "bulk_expression" in checkpoint.manifest.assets:
        import io

        candidates.append(("checkpoint asset", io.BytesIO(checkpoint.asset_bytes("bulk_expression"))))

    last_error: Optional[Exception] = None
    for label, source in candidates:
        try:
            if hasattr(source, "read"):
                bulk = pd.read_csv(source, index_col=0)
            else:
                if not Path(source).is_file():
                    continue
                bulk = pd.read_csv(source, index_col=0)
        except Exception as exc:  # pragma: no cover - depends on user data
            last_error = exc
            continue
        needed = [name for name in sheet.bulk_sample_names if name]
        missing = [name for name in needed if name not in bulk.columns]
        if missing:
            last_error = ConfigError(
                f"the bulk matrix from {label} is missing {len(missing)} column(s) required by the "
                f"sample sheet: {', '.join(missing[:4])}"
                + (f" (+{len(missing) - 4} more)" if len(missing) > 4 else "")
            )
            continue
        print(f"Using bulk expression matrix from {label} ({bulk.shape[0]} genes x {bulk.shape[1]} samples)")
        return bulk

    detail = f" Last error: {last_error}" if last_error else ""
    raise ConfigError(
        "the low-resolution model is conditioned on bulk expression, but no usable bulk matrix "
        f"was found.{detail}",
        hint=(
            "Pass the bulk expression matrix with --bulk (rows = genes, columns = sample ids) and "
            "make sure every value of the 'Bulk_sample_name' column of the sample sheet is a column "
            "of that matrix. Alternatively pass --no-bulk if the checkpoint was trained without it."
        ),
    )


def write_prediction_pickle(payload: Dict[str, Any], path: Path) -> Path:
    """Write a prediction dictionary with the legacy pickle helper."""
    ensure_code_on_path()
    import utils  # legacy helper: plain pickle.dump, creates parent directories

    path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_pickle(payload, str(path))
    return path


def expected_output(
    pred_dir: Path,
    sample: str,
    *,
    highres: bool,
    uint8_output: bool = False,
    compress_output: bool = False,
) -> Path:
    """The file the stage will write for ``sample``.

    Resume support and output verification both go through this, so a stage can
    never skip a sample because a file in a *different* format happens to exist.
    """
    from ..naming import highres_output_path, lowres_output_path

    if highres:
        return highres_output_path(pred_dir, sample, uint8_output, compress_output)
    return lowres_output_path(pred_dir, sample)


def read_sample_sheet_for_run(run_root: Path, relative: str) -> SampleSheet:
    path = Path(relative)
    if not path.is_absolute():
        path = run_root / relative
    return read_sample_sheet(path, kind="inference")


def describe_plan(stage: str, sheet: SampleSheet, device: Any, extra: Sequence[str] = ()) -> None:
    print(f"[{stage}] samples={len(sheet)} device={device}")
    for line in extra:
        print(f"[{stage}] {line}")


def fail(message: str, hint: Optional[str] = None) -> "NoReturn":  # type: ignore[valid-type]
    raise AuroraError(message, hint=hint)


__all__ = [
    "FORMAT_MODERN",
    "FORMAT_LEGACY",
    "OUTPUT_FORMATS",
    "add_run_arguments",
    "add_lowres_arguments",
    "add_highres_arguments",
    "resolve_device",
    "load_run_layout",
    "open_checkpoint",
    "build_stage_context",
    "enforce_prerequisites",
    "resolve_bulk_matrix",
    "write_prediction_pickle",
    "expected_output",
    "read_sample_sheet_for_run",
    "describe_plan",
    "StagePrerequisiteError",
    "load_assets",
]
