"""Materialising everything a stage expects inside the run directory.

The legacy scripts resolve their inputs relative to ``main_dir``, so the run
directory has to contain (or link to) the files they read:

* ``config/samples.csv``            -- canonical inference sheet
* ``config/train_samples.csv``      -- the training sheet, unchanged
* ``config/params_lowres.json``     -- low-resolution model parameters
* ``config/bulk.csv``               -- bulk expression matrix, when given
* ``checkpoints/params_highres.json``         -- train_highres.py reads it from ``-t``
* ``checkpoints/gene_list.csv``, ``cell_types.csv``,
  ``gene_normalize_factor.csv``               -- read from ``{train_path}/``

Small files are copied; large ones are hard-linked when possible so a big
expression matrix is not duplicated on disk.  Nothing is ever deleted.

Standard library only.
"""

from __future__ import annotations

import json
import os
import shutil
import typing
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .bundle import CombinedCheckpoint
from .checkpoint import CheckpointAssets
from .config import RunConfig
from .errors import ConfigError
from .naming import resolve_sheet_images
from .paths import RunLayout
from .samples import SampleSheet, write_sample_sheet

#: Files above this size are hard-linked instead of copied when possible.
DEFAULT_COPY_LIMIT_MB = 64.0

#: Canonical inference sheet; every stage indexes it by ``Sample_name``.
SAMPLES_FILENAME = "samples.csv"
#: The same sheet with ``Image_name`` resolved to the file that exists on disk.
RESOLVED_SAMPLES_FILENAME = "samples_resolved.csv"
#: The original Visium training sheet, untouched.
TRAIN_SAMPLES_FILENAME = "train_samples.csv"

#: Asset key -> file name written into ``<run>/checkpoints``.
CHECKPOINT_ASSET_FILES: Mapping[str, str] = {
    "gene_list": "gene_list.csv",
    "cell_types": "cell_types.csv",
    "gene_norm_factor": "gene_normalize_factor.csv",
}


def materialize_file(
    source: Path,
    destination: Path,
    mode: str = "auto",
    copy_limit_mb: float = DEFAULT_COPY_LIMIT_MB,
) -> Tuple[Path, str]:
    """Put ``source`` at ``destination``; return ``(destination, how)``.

    ``mode`` is ``'auto'``, ``'copy'`` or ``'link'``.  ``'auto'`` copies small
    files and hard-links large ones, falling back to a copy when the file
    system does not support links.
    """
    source = Path(source).resolve()
    destination = Path(destination)
    if not source.is_file():
        raise ConfigError(f"{source} is not a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return destination, "existing"

    size_mb = source.stat().st_size / (1024 * 1024)
    want_link = mode == "link" or (mode == "auto" and size_mb > copy_limit_mb)
    if want_link:
        try:
            if destination.exists():
                destination.unlink()
            os.link(source, destination)
            return destination, "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return destination, "copy"


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(tmp, path)
    return path


def column_notes(sheet: SampleSheet, filename: str) -> List[str]:
    """Explain which optional columns were written, and which were dropped.

    The legacy scripts branch on *column presence*, so a column that was dropped
    or completed with a default is worth stating explicitly.
    """
    from .samples import COL_BULK, COL_FOLDER, COL_PIXEL_SIZE

    columns = sheet.output_columns()
    notes: List[str] = [f"{filename}: columns {', '.join(columns)}"]
    dropped = [
        name
        for name, present in (
            (COL_FOLDER, COL_FOLDER in sheet.columns),
            (COL_PIXEL_SIZE, COL_PIXEL_SIZE in sheet.columns),
            (COL_BULK, COL_BULK in sheet.columns),
        )
        if present and name not in columns
    ]
    for name in dropped:
        if name == COL_FOLDER:
            notes.append(
                f"dropped the empty '{COL_FOLDER}' column: code/rescale.py selects the "
                "per-slide-subdirectory layout from the presence of that column, and an "
                "all-empty one would make it join a literal 'nan' into every path"
            )
        else:
            notes.append(f"dropped the empty '{name}' column")
    if COL_FOLDER in columns and not sheet.has_folders:
        notes.append(f"'{COL_FOLDER}' completed with '.' for the rows that had no folder")
    if COL_BULK in columns and not sheet.has_bulk_names:
        notes.append(f"'{COL_BULK}' completed with Sample_name for the rows that had none")
    return notes


def materialize_sample_sheets(
    layout: RunLayout,
    sheet: SampleSheet,
    mode: str,
    training_sheet: Optional[Path] = None,
    image_root: Optional[Path] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Write the sample sheets the stages read.

    Two sheets are written in inference mode:

    ``config/samples.csv``
        the canonical sheet, unchanged.  Used by the stages that index by
        ``Sample_name`` (the predictors, ``plot``, ``histosweep``).
    ``config/samples_resolved.csv``
        the same sheet with ``Image_name`` set to the file name that really
        exists on disk.  Used by ``rescale`` and the feature extractors, because
        the legacy scripts derive both the image path *and* the tissue-mask path
        from that column.

    Returns ``(written, notes)``.
    """
    written: Dict[str, str] = {}
    notes: List[str] = []
    canonical = layout.config_dir / SAMPLES_FILENAME
    write_sample_sheet(sheet, canonical)
    written["samples"] = canonical.as_posix()
    notes.extend(column_notes(sheet, canonical.name))

    if mode == "training":
        source = training_sheet or sheet.path
        if source is None:
            raise ConfigError(
                "training mode needs the original training sample sheet",
                hint="Pass --samples <Training_samples.csv>.",
            )
        destination = layout.config_dir / TRAIN_SAMPLES_FILENAME
        _, how = materialize_file(Path(source), destination, mode="copy")
        written["train_samples"] = f"{destination.as_posix()} ({how})"
        return written, notes

    if image_root is None:
        return written, notes
    resolved_sheet, resolutions, notes = resolve_sheet_images(sheet, Path(image_root))
    resolved_path = layout.config_dir / RESOLVED_SAMPLES_FILENAME
    write_sample_sheet(resolved_sheet, resolved_path)
    written["samples_resolved"] = resolved_path.as_posix()
    missing = [r for r in resolutions if not r.found]
    if missing:
        notes.append(
            f"{len(missing)} of {len(resolutions)} slide(s) named by the sheet were not found "
            f"under {image_root}; the rescale stage will list the paths it tried"
        )
    return written, notes


#: The params keys that name a file the training scripts open.
PARAMS_ASSET_KEYS: Mapping[str, str] = {
    "gene_list": "gene_list_file",
    "gene_norm_factor": "gene_norm_factor_file",
    "cell_types": "celltype_file",
}

#: Where each stage's script resolves those keys, as a prefix from the run root.
#:
#: ``code/train.py`` does ``f"{main_dir}/{gene_list_file}"`` with ``main_dir = -p``
#: (the run root), while ``code/train_highres.py`` and
#: ``code/prepare_highres_training.py`` do ``f"{main_dir}/{train_path}/{...}"``
#: (``-p/<checkpoints>``).  The paths in the checkpoint's JSON
#: (``LUAD_gene.csv``, ``/LUAD_gene.csv``) match neither, so they are rewritten
#: here for each stage -- otherwise training dies on a missing CSV.
PARAMS_ASSET_BASE: Mapping[str, str] = {
    "lowres": "checkpoints",   # train.py: run-root relative
    "highres": "",             # train_highres.py: relative to <run>/checkpoints
}

#: Params the user may override, and the type each value is coerced to.
PARAMS_TYPES: Mapping[str, typing.Callable[[str], Any]] = {
    "num_epochs": int,
    "exp_epochs": int,
    "batch_size": int,
    "num_workers": int,
    "img_emb_dim": int,
    "mse_weight": float,
    "prop_weight": float,
    "focal_gamma": float,
    "lr": float,
    "lr_prop": float,
    "weight_decay": float,
    "stepLR_step_exp": int,
    "stepLR_gamma_exp": float,
    "stepLR_step_prop": int,
    "stepLR_gamma_prop": float,
    "pixel_size": float,
    "superpatch_size": int,
    "superpatch_stride": int,
    "superpatch_minsize": int,
    "superpatch_minlowres_patch": int,
    "use_bulk": bool,
    "pred_celltype": bool,
    "bidirectional": bool,
    "retrain_cellprop_only": bool,
    "use_context_loss": bool,
}


def _coerce_param(key: str, raw: str) -> Any:
    """Turn a ``KEY=VALUE`` string into the type the scripts expect."""
    kind = PARAMS_TYPES.get(key)
    if kind is bool:
        lowered = raw.strip().lower()
        if lowered in ("true", "t", "yes", "y", "1", "on"):
            return True
        if lowered in ("false", "f", "no", "n", "0", "off"):
            return False
        raise ConfigError(
            f"--param {key}={raw} is not a boolean",
            hint="Use true or false.",
        )
    if kind is int:
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"--param {key}={raw} is not an integer") from exc
    if kind is float:
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"--param {key}={raw} is not a number") from exc
    if kind is None:
        raise ConfigError(
            f"--param {key} is not a known training parameter",
            hint="Known parameters:\n  " + ", ".join(sorted(PARAMS_TYPES)),
        )
    return raw


def parse_param_overrides(pairs: Sequence[str]) -> Dict[str, Any]:
    """``["num_epochs=50", "pred_celltype=false"]`` -> ``{"num_epochs": 50, ...}``."""
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ConfigError(
                f"--param {pair!r} is not in KEY=VALUE form",
                hint='For example: --param num_epochs=100 --param pred_celltype=false',
            )
        if key not in PARAMS_TYPES:
            raise ConfigError(
                f"unknown training parameter {key!r}",
                hint=(
                    "Known parameters:\n  " + ", ".join(sorted(PARAMS_TYPES)) + "\n"
                    "Run 'python -m aurora params --run <run>' to see the current values."
                ),
            )
        overrides[key] = _coerce_param(key, value)
    return overrides


def stage_params(
    stage_params: Mapping[str, Any],
    stage: str,
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """One stage's parameters with the asset paths fixed and overrides applied."""
    params = dict(stage_params)
    base = PARAMS_ASSET_BASE.get(stage, "")
    for asset_key, param_key in PARAMS_ASSET_KEYS.items():
        filename = CHECKPOINT_ASSET_FILES[asset_key]
        params[param_key] = f"{base}/{filename}" if base else filename
    # Only keys this stage already has: `train.py` and `train_highres.py` read
    # different subsets, and inventing keys would be noise in the JSON.
    for key, value in overrides.items():
        if key in params or key in PARAMS_TYPES:
            params[key] = value
    return params


def _assets_supplied(config: RunConfig) -> bool:
    """True when every asset a checkpoint would have carried was given explicitly.

    This is what makes a training run possible without a checkpoint.  It is
    deliberately not just ``config.mode == "training"``: ``preprocess`` also runs in
    training mode and needs none of these.
    """
    return all(
        [
            config.gene_list_path,
            config.cell_type_path,
            config.norm_factor_path,
        ]
    )


def materialize_asset_overrides(layout: RunLayout, config: RunConfig) -> Dict[str, str]:
    """Copy the three assets from ``--gene-list``/``--cell-types``/``--norm-factors``.

    The path taken when training without a checkpoint: the checkpoint would only
    have carried these, so they have to be given explicitly instead.
    """
    sources = {
        "gene_list": config.gene_list_path,
        "cell_types": config.cell_type_path,
        "gene_norm_factor": config.norm_factor_path,
    }
    missing = [
        f"  --{key.replace('_', '-')}  ({CHECKPOINT_ASSET_FILES[key]})"
        for key, value in sources.items()
        if not value
    ]
    if missing:
        raise ConfigError(
            "training without --model needs the three assets that a checkpoint would "
            "have supplied, and these are missing:\n" + "\n".join(missing),
            hint=(
                "Either pass the combined checkpoint:\n"
                "  --model trained_model/AURORA_LUAD_combined.pth\n"
                "or supply all three CSVs so the run is configured from your own panel:\n"
                "  --gene-list genes.csv --cell-types cell_types.csv "
                "--norm-factors norm_factors.csv\n"
                "Nothing is downloaded or trained until they are given."
            ),
        )
    checkpoints_dir = layout.checkpoints_dir
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    for key, filename in CHECKPOINT_ASSET_FILES.items():
        destination = checkpoints_dir / filename
        _, how = materialize_file(Path(sources[key]), destination, mode="copy")
        written[key] = (
            f"{destination.as_posix()} ({how} from --{key.replace('_', '-')})"
        )
    return written


def load_run_params(layout: RunLayout, config: RunConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The low- and high-resolution parameters this run is actually using.

    ``prepare_run`` writes ``config/params_lowres.json`` and
    ``checkpoints/params_highres.json`` from the checkpoint, from the packaged
    defaults, or from ``--param``.  Stages read the parameters from those files, so the
    in-memory :class:`~aurora.spec.StageContext` has to read them too: building it from
    an empty dict made ``features_lowres`` fail with "the combined checkpoint does not
    record the low-resolution patch sizes" in a run that has no checkpoint at all.
    """
    from .defaults import DEFAULT_HIGHRES_PARAMS, DEFAULT_LOWRES_PARAMS

    def read(path: Path, defaults: Mapping[str, Any], stage: str) -> Dict[str, Any]:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = None
            if isinstance(data, dict) and data:
                return data
        return stage_params(defaults, stage, config.param_overrides)

    return (
        read(layout.config_dir / "params_lowres.json", DEFAULT_LOWRES_PARAMS, "lowres"),
        read(layout.checkpoints_dir / "params_highres.json", DEFAULT_HIGHRES_PARAMS, "highres"),
    )


def materialize_default_params(
    layout: RunLayout, config: RunConfig
) -> Dict[str, str]:
    """Write the parameter files from :mod:`aurora.defaults`, no checkpoint needed."""
    from .defaults import DEFAULT_HIGHRES_PARAMS, DEFAULT_LOWRES_PARAMS

    written: Dict[str, str] = {}
    for name, params, destination in (
        (
            "params_lowres",
            stage_params(DEFAULT_LOWRES_PARAMS, "lowres", config.param_overrides),
            layout.config_dir / "params_lowres.json",
        ),
        (
            "params_highres",
            stage_params(DEFAULT_HIGHRES_PARAMS, "highres", config.param_overrides),
            layout.checkpoints_dir / "params_highres.json",
        ),
    ):
        written[name] = _write_stage_params(destination, params, config)
    # Same reason as in materialize_checkpoint_assets: prepare_highres_training
    # rebuilds the low-res network, so it needs the low-res architecture with
    # checkpoints-relative asset paths.
    written["params_highres_training"] = _write_stage_params(
        layout.checkpoints_dir / "params_highres_training.json",
        stage_params(DEFAULT_LOWRES_PARAMS, "highres", config.param_overrides),
        config,
    )
    return written


def materialize_checkpoint_assets(
    layout: RunLayout,
    checkpoint: CombinedCheckpoint,
    assets: CheckpointAssets,
    config: RunConfig,
) -> Dict[str, str]:
    """Extract the checkpoint's assets and parameters into the run directory."""
    checkpoints_dir = layout.checkpoints_dir
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    for key, filename in CHECKPOINT_ASSET_FILES.items():
        destination = checkpoints_dir / filename
        if key in checkpoint.manifest.assets:
            checkpoint.write_asset_to(key, destination)
            written[key] = f"{destination.as_posix()} (from checkpoint)"
            continue
        override = {
            "gene_list": config.gene_list_path,
            "cell_types": config.cell_type_path,
            "gene_norm_factor": config.norm_factor_path,
        }.get(key)
        if override:
            _, how = materialize_file(Path(override), destination, mode="copy")
            written[key] = f"{destination.as_posix()} ({how} from --{key.replace('_', '-')})"
    missing = [k for k in CHECKPOINT_ASSET_FILES if k not in written]
    if missing:
        raise ConfigError(
            f"could not provide these assets: {', '.join(missing)}",
            hint=(
                "Rebuild the combined checkpoint with the assets bundled:\n"
                "  python tools/merge_aurora_checkpoints.py --lowres … --highres … "
                "--gene-list trained_model/LUAD_gene.csv "
                "--cell-types trained_model/LUAD_cell_types.csv "
                "--norm-factors trained_model/LUAD_gene_normalize_factor.csv --out …"
            ),
        )

    lowres_params = stage_params(
        checkpoint.manifest.params_for("lowres"), "lowres", config.param_overrides
    )
    highres_params = (
        stage_params(
            checkpoint.manifest.params_for("highres"), "highres", config.param_overrides
        )
        if "highres" in checkpoint.manifest.stages
        else {}
    )
    for name, params, destination in (
        ("params_lowres", lowres_params, layout.config_dir / "params_lowres.json"),
        ("params_highres", highres_params, checkpoints_dir / "params_highres.json"),
    ):
        if not params:
            continue
        written[name] = _write_stage_params(destination, params, config)

    # `prepare_highres_training.py` rebuilds the **low-resolution** network and loads
    # the low-res weights into it, so its parameters must describe the architecture
    # that was actually trained -- including any --param or hand edit, such as
    # pred_celltype=false -- while the file is opened as
    # `{main}/{train_path}/{parameter_json}`, which needs the checkpoints-relative
    # asset spelling. Handing it params_highres.json produced
    # "size mismatch for lstm.weight_ih_l0: [1024,1536] vs [1024,3072]" plus a 0-vs-10
    # cell-type decoder.
    effective_lowres = layout.config_dir / "params_lowres.json"
    if effective_lowres.is_file():
        try:
            trained_arch = json.loads(effective_lowres.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            trained_arch = dict(lowres_params)
        written["params_highres_training"] = _write_stage_params(
            checkpoints_dir / "params_highres_training.json",
            stage_params(trained_arch, "highres", {}),
            config,
        )
    return written


def _write_stage_params(destination: Path, params: Mapping[str, Any], config: RunConfig) -> str:
    """Write a stage's parameter file, respecting hand edits.

    These are the knobs people are meant to change, so a re-run must not silently
    throw an edit away -- but it must still keep the *asset paths* correct, and an
    explicit ``--param`` has to reach the file the script reads or it would have no
    effect at all.  So:

    * missing file          -> write it in full;
    * ``--refresh-params``  -> write it in full, discarding edits;
    * ``--param`` given     -> merge the overrides into the existing file;
    * otherwise             -> keep every value, but repair the asset paths.
    """
    existing: Dict[str, Any] = {}
    if destination.is_file():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    if not existing or config.refresh_params:
        write_json(destination, dict(params))
        return destination.as_posix()

    merged = dict(existing)
    asset_keys = set(PARAMS_ASSET_KEYS.values())
    repaired = [key for key in asset_keys if merged.get(key) != params.get(key)]
    for key in asset_keys:
        merged[key] = params[key]
    applied = []
    for key, value in config.param_overrides.items():
        if merged.get(key) != value:
            applied.append(key)
        merged[key] = value
    write_json(destination, merged)
    notes = []
    if repaired:
        notes.append("asset paths corrected: " + ", ".join(sorted(repaired)))
    if applied:
        notes.append("--param applied: " + ", ".join(sorted(applied)))
    if notes:
        return f"{destination.as_posix()} ({'; '.join(notes)})"
    return f"{destination.as_posix()} (kept)"


def materialize_bulk_matrix(layout: RunLayout, config: RunConfig) -> Optional[str]:
    """Copy/link the bulk expression matrix into ``config/bulk.csv``."""
    if config.bulk_path is None:
        return None
    destination = layout.config_dir / "bulk.csv"
    _, how = materialize_file(Path(config.bulk_path), destination, mode="auto")
    return f"{destination.as_posix()} ({how})"


def materialize_training_inputs(layout: RunLayout, config: RunConfig) -> Dict[str, str]:
    """Link the raw/processed training data the legacy scripts expect."""
    written: Dict[str, str] = {}
    raw = config.raw_data_path
    processed = config.processed_data_path
    for name, value in (("raw_data", raw), ("processed_data", processed)):
        if not value:
            continue
        source = Path(value).expanduser()
        if not source.is_absolute():
            source = config.resolved_project() / source
        if source.is_dir():
            written[name] = source.as_posix()
    return written


def prepare_run(
    layout: RunLayout,
    config: RunConfig,
    sheet: SampleSheet,
    checkpoint: Optional[CombinedCheckpoint],
    assets: Optional[CheckpointAssets],
    require_checkpoint: bool = True,
) -> Dict[str, Any]:
    """Create the layout and everything the stages read.  Returns a report.

    ``checkpoint`` is ``None`` for a run whose stages never read one (today that
    is ``preprocess`` alone); the bundled assets are then skipped rather than
    demanded.
    """
    layout.ensure(config.mode)
    report: Dict[str, Any] = {"layout": layout.describe(), "samples": {}, "notes": []}
    written, notes = materialize_sample_sheets(
        layout,
        sheet,
        config.mode,
        training_sheet=Path(config.samples_path) if config.samples_path else None,
        image_root=config.resolved_image_root() if config.mode != "training" else None,
    )
    report["samples"] = written
    report["notes"].extend(notes)
    if checkpoint is not None:
        report["checkpoint_assets"] = materialize_checkpoint_assets(
            layout, checkpoint, assets, config
        )
        # Surface the interesting cases: a corrected asset path or an applied
        # --param.  A file that was simply kept needs no announcement.
        for name, detail in report["checkpoint_assets"].items():
            if "corrected" in detail or "applied" in detail:
                report["notes"].append(f"{name}: {detail}")
    elif require_checkpoint:
        raise ConfigError(
            "this run needs the combined checkpoint but none was provided",
            hint="Pass --model, or let AURORA find the only combined .pth automatically.",
        )
    elif config.mode == "training" and _assets_supplied(config):
        # Training without a checkpoint: it would only have supplied the three assets
        # and the parameter template, so take those from the command line and the
        # packaged defaults instead.  No weights are involved either way.
        report["checkpoint_assets"] = materialize_asset_overrides(layout, config)
        report["checkpoint_assets"].update(materialize_default_params(layout, config))
        report["notes"].append(
            "no checkpoint: parameters come from aurora/defaults.py, assets from the "
            "--gene-list/--cell-types/--norm-factors files"
        )
    else:
        report["checkpoint_assets"] = {}
        report["notes"].append(
            "no checkpoint is needed by these stages, so its bundled assets were not copied"
        )
    bulk = materialize_bulk_matrix(layout, config)
    if bulk:
        report["bulk"] = bulk
    if config.mode == "training":
        report["training_inputs"] = materialize_training_inputs(layout, config)
    return report
