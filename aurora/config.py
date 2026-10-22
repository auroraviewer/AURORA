"""Run configuration: one typed object for everything the CLI can set.

The old scripts each had their own argparse block, so the same concept (the
interim directory, the sample sheet, the pixel size) had a different name in
every file.  :class:`RunConfig` collects all of them once; the stage command
builders translate it into whatever each legacy script expects.

Standard library only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .errors import ConfigError
from .paths import RunLayout
from .samples import SampleSheet

#: Default relative directory holding the whole-slide images.
DEFAULT_IMAGE_PATH = "wsis"

#: Default name of the segmentation output directory inside a run.
DEFAULT_PRESET = "tcga.csv"

#: Gene-type table used by ``prepare_st`` to keep the protein-coding genes.
DEFAULT_GENETYPE = "TCGA_genetype.csv"

#: Default target pixel size in micrometres (matches the trained models).
DEFAULT_PIXEL_SIZE = 0.5


def _resolve_existing(path: Path, what: str, hint: str = "") -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    if not resolved.exists():
        raise ConfigError(
            f"{what} {resolved} does not exist",
            hint=hint or "Check the path you passed on the command line.",
        )
    return resolved


@dataclass
class RunConfig:
    """Everything needed to execute a pipeline stage."""

    # ---------------------------------------------------------------- inputs
    project_path: Path = field(default_factory=lambda: Path.cwd())
    model_path: Optional[Path] = None
    samples_path: Optional[Path] = None
    run_name: str = "run"
    run_root: Optional[Path] = None
    mode: str = "inference"

    # --------------------------------------------------------------- runtime
    device: str = "auto"
    dry_run: bool = False
    force: bool = False
    keep_going: bool = False
    auto_prerequisites: bool = False
    log_to_console: bool = True

    # ------------------------------------------------------------- inference
    image_path: str = DEFAULT_IMAGE_PATH
    preset: Optional[str] = None
    uni_local_path: Optional[str] = None
    hf_token: Optional[str] = None
    batch_size: int = 128
    num_workers: int = 4
    stride: int = 112
    target_pixel_size: float = DEFAULT_PIXEL_SIZE
    plot: bool = False
    compile_mode: str = "off"
    samples_subset: List[str] = field(default_factory=list)
    prediction_batch_size: int = 100
    #: Quantise predictions to uint8.  At high resolution this also selects the
    #: compact HDF5 writer over the float pickle, matching command.sh.
    uint8_output: bool = True
    #: Keep only the finest (224 px) low-resolution level, as command.sh does.
    only_224: bool = True
    #: High-resolution HDF5 layout, filter and level (command.sh defaults).
    h5_layout: str = "spatial_gene_delta"
    #: Empty means "let AURORA choose": zstd when hdf5plugin is installed, else gzip.
    h5_compression: str = ""
    h5_compression_level: int = 9
    output_format: str = "modern"
    prediction_compression: bool = False
    gene_list_path: Optional[Path] = None
    cell_type_path: Optional[Path] = None
    #: Gene-type table for ``prepare_st``; defaults to the shipped
    #: ``examples/TCGA_genetype.csv`` and is only used by the training stages.
    genetype_path: Optional[Path] = None
    norm_factor_path: Optional[Path] = None
    #: Training parameter overrides from ``--param KEY=VALUE``.  Applied to the
    #: per-stage parameter files, so no JSON editing is needed for a one-off change.
    param_overrides: Dict[str, Any] = field(default_factory=dict)
    #: Regenerate ``params_lowres.json`` / ``params_highres.json`` from the
    #: checkpoint, discarding hand edits.
    refresh_params: bool = False
    bulk_path: Optional[Path] = None

    # -------------------------------------------------------------- training
    processed_data_path: str = "processed_data"
    raw_data_path: str = "raw_data"
    #: Training output directory inside the run.  It must be ``checkpoints``:
    #: train_highres.py and prepare_highres_training.py read the gene list, the
    #: cell types, the parameters and the low-resolution prior from
    #: ``<run>/<train_dir>/``, and that is also where the combined checkpoint is
    #: written.
    train_dir: str = "checkpoints"
    parameter_json: Optional[Path] = None
    train_samples: Optional[Path] = None
    train_sample_col: int = 0
    finetune_gene_list: Optional[Path] = None
    num_epochs: Optional[int] = None
    smooth_counts: bool = False
    model_tag: str = ""

    # ------------------------------------------------------- optional stages
    gene_names: List[str] = field(default_factory=list)
    cell_types: List[str] = field(default_factory=list)
    plot_resolution: int = 224
    #: Figure pixels = image pixels / this.  The figures must be at least 1/7 of
    #: the rescaled slide, so this must stay <= 7 (7 is the default); smaller
    #: values produce larger figures.
    plot_downscale: int = 7
    #: Which prediction stage the plot stage reads: 'lowres' or 'highres'.
    plot_predictions: str = "lowres"
    histosweep_path: Optional[str] = None
    #: Super-pixel size for aligning high-res predictions with the HistoSweep mask.
    histosweep_patch_size: int = 14

    # ------------------------------------------------------------------- misc
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- factories
    def layout(self) -> RunLayout:
        return RunLayout.create(self.project_path, self.run_name, self.run_root)

    def resolved_project(self) -> Path:
        return _resolve_existing(self.project_path, "project path")

    def resolved_model(self) -> Path:
        if self.model_path is None:
            raise ConfigError(
                "--model is required for this command",
                hint=(
                    "Point it at a *combined* checkpoint, for example:\n"
                    "  --model trained_model/AURORA_LUAD_combined.pth"
                ),
            )
        return _resolve_existing(
            Path(self.model_path),
            "combined checkpoint",
            hint=(
                "Build one from the legacy pair with:\n"
                "  python tools/merge_aurora_checkpoints.py --lowres <lowres.pth> "
                "--highres <highres.pth> --out trained_model/AURORA_<CANCER>_combined.pth"
            ),
        )

    def resolved_samples(self) -> Path:
        if self.samples_path is None:
            recorded = self.layout().config_dir / "samples.csv"
            if recorded.is_file():
                return recorded
            raise ConfigError(
                "--samples is required for this command",
                hint=(
                    "Pass the metadata CSV:\n"
                    "  --samples /path/to/Inference_samples.csv\n"
                    "Re-running an existing run picks up the sheet it already recorded "
                    f"({recorded}), but that file does not exist yet."
                ),
            )
        return _resolve_existing(Path(self.samples_path), "sample sheet")

    def resolved_preset(self) -> Path:
        """Resolve the segmentation preset, searching the shipped examples."""
        from ._legacy import repo_root

        candidates: List[Path] = []
        if self.preset:
            raw = Path(self.preset).expanduser()
            if raw.is_absolute():
                candidates.append(raw)
            else:
                candidates.extend(
                    [
                        self.resolved_project() / raw,
                        Path.cwd() / raw,
                        repo_root() / "examples" / raw,
                        repo_root() / raw,
                    ]
                )
        else:
            candidates.append(repo_root() / "examples" / DEFAULT_PRESET)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        searched = "\n".join(f"  - {c}" for c in candidates)
        raise ConfigError(
            f"segmentation preset {self.preset or DEFAULT_PRESET!r} was not found",
            hint=(
                "Pass an explicit path, e.g. --preset examples/tcga.csv\nSearched:\n" + searched
            ),
        )

    def resolved_genetype(self) -> Optional[Path]:
        """The gene-type table ``prepare_st`` needs, searched like the preset.

        Returns ``None`` when nothing exists, so the stage can say so itself; the
        shipped ``examples/TCGA_genetype.csv`` normally answers this.
        """
        from ._legacy import repo_root

        name = self.genetype_path or DEFAULT_GENETYPE
        raw = Path(name).expanduser()
        if raw.is_absolute():
            candidates = [raw]
        else:
            candidates = [
                self.resolved_project() / raw,
                Path.cwd() / raw,
                repo_root() / "examples" / raw,
                repo_root() / raw,
            ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None

    def resolved_raw_data(self) -> Path:
        """The directory of raw Visium sample folders, searched like the preset.

        ``--raw-data`` (default ``raw_data``) is an *input*, so it belongs to the
        project rather than to the run directory.  An absolute path is used as
        given; a relative one is looked for under the project, then the current
        directory, then the repository, which is where people actually put it.
        """
        from ._legacy import repo_root

        raw = Path(self.raw_data_path).expanduser()
        if raw.is_absolute():
            candidates = [raw]
        else:
            candidates = [
                self.resolved_project() / raw,
                Path.cwd() / raw,
                repo_root() / raw,
            ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        # Nothing exists yet: return the least surprising absolute path so the
        # error can name one place instead of three.
        return (self.resolved_project() / raw).resolve() if not raw.is_absolute() else raw

    def raw_data_candidates(self) -> List[Path]:
        """Every place :meth:`resolved_raw_data` looks, for the error message."""
        from ._legacy import repo_root

        raw = Path(self.raw_data_path).expanduser()
        if raw.is_absolute():
            return [raw]
        return [
            (self.resolved_project() / raw).resolve(),
            (Path.cwd() / raw).resolve(),
            (repo_root() / raw).resolve(),
        ]

    def resolved_image_root(self) -> Path:
        raw = Path(self.image_path).expanduser()
        if raw.is_absolute():
            return raw.resolve()
        return (self.resolved_project() / raw).resolve()

    def resolved_processed_data(self) -> Path:
        raw = Path(self.processed_data_path).expanduser()
        if raw.is_absolute():
            return raw.resolve()
        return (self.resolved_project() / raw).resolve()

    def device_string(self) -> str:
        if self.device and self.device != "auto":
            return self.device
        try:
            import importlib.util

            if importlib.util.find_spec("torch") is not None:
                import torch  # type: ignore

                return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch is optional here
            pass
        return "cpu"

    # -------------------------------------------------------------- context
    def stage_options(self) -> Dict[str, Any]:
        """The free-form option bag handed to :class:`aurora.spec.StageContext`."""
        return {
            "device": self.device_string(),
            "project_path": str(self.resolved_project()) if self.project_path else "",
            "image_path": self.image_path,
            "preset": self.preset,
            "uni_local_path": self.uni_local_path,
            "hf_token": self.hf_token,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "stride": self.stride,
            "target_pixel_size": self.target_pixel_size,
            "plot": self.plot,
            "compile_mode": self.compile_mode,
            "processed_data_path": self.processed_data_path,
            "raw_data_path": self.raw_data_path,
            "train_dir": self.train_dir,
            "parameter_json": str(self.parameter_json) if self.parameter_json else None,
            "train_samples": str(self.train_samples) if self.train_samples else None,
            "train_sample_col": self.train_sample_col,
            "finetune_gene_list": str(self.finetune_gene_list) if self.finetune_gene_list else None,
            "num_epochs": self.num_epochs,
            "smooth_counts": self.smooth_counts,
            "prediction_batch_size": self.prediction_batch_size,
            "uint8_output": self.uint8_output,
            "only_224": self.only_224,
            "h5_layout": self.h5_layout,
            "h5_compression": self.h5_compression,
            "h5_compression_level": self.h5_compression_level,
            "output_format": self.output_format,
            "compress_output": self.prediction_compression,
            "gene_names": list(self.gene_names),
            "cell_types": list(self.cell_types),
            "plot_resolution": self.plot_resolution,
            "plot_downscale": self.plot_downscale,
            "plot_predictions": self.plot_predictions,
            "force_plots": bool(self.force),
            "dry_run": bool(self.dry_run),
            "histosweep_path": self.histosweep_path,
            "histosweep_patch_size": self.histosweep_patch_size,
            "model_path": str(self.model_path) if self.model_path else "",
            "run_name": self.run_name,
            "mode": self.mode,
            "combined_name": self.extra.get("combined_name", "aurora.pth"),
            **self.extra,
        }

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = value.as_posix()
        return data

    def describe(self) -> str:
        lines = [
            f"mode            : {self.mode}",
            f"project         : {self.resolved_project() if self.project_path else '(unset)'}",
        ]
        if self.model_path:
            if self.mode == "training":
                # Training never loads these weights; the file supplies configuration.
                lines.append(f"config from     : {self.model_path}")
            else:
                lines.append(f"model           : {self.model_path}")
        if self.samples_path:
            lines.append(f"samples         : {self.samples_path}")
        lines.append(f"run             : {self.run_name}")
        if self.run_root:
            lines.append(f"run root        : {self.run_root}")
        if self.mode == "inference":
            lines.append(f"image path      : {self.image_path}")
            lines.append(f"target pixel um : {self.target_pixel_size}")
            lines.append(f"device          : {self.device_string()}")
        else:
            lines.append(f"processed data  : {self.processed_data_path}")
        return "\n".join(lines)


def select_samples(sheet: SampleSheet, subset: Sequence[str]) -> SampleSheet:
    """Apply ``--sample`` filters to a sheet."""
    if not subset:
        return sheet
    return sheet.with_names(list(subset))
