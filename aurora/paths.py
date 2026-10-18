"""Canonical output-directory layout for the AURORA unified code system.

Requirement #4 of the reorganisation: the old pipeline scattered its outputs
across the project root under ad-hoc names (``AURORA_interim``,
``AURORA_pred/pred``, ``AURORA_pred/processed_pkl``,
``AURORA_pred/processed_pkl_test_highres``, ``wsis_segmentation``,
``AURORA_plots``, ``debug``, ``Breast`` ...).  Everything now lives under a
single, self-describing run directory::

    <project>/aurora_runs/<run_name>/
        run.json                  # stage state, see aurora.state
        config/                   # copies/links of the inputs actually used
        logs/                     # one log file per stage
        01_segmentation/
            masks/                # <slide>.geojson
            contours/
            patches/              # <sample>.h5  (training only)
            process_list_autogen.csv
            viz/
        02_features/
            UNI_multiscale_patches/
                wsis_rescaled/            # <sample>_rescaled.tif + _mask_rescaled.tif
                check_figure/
                locs/                     # training only
                emb_3584/ emb_896/ emb_224/     # low-resolution UNI2-h features
                emb_56/  emb_14/                # high-resolution features
                rescale_parameters_<ts>.csv
        03_predictions/
            lowres/{pred,processed_pkl}
            highres/{pred,processed_pkl_test_highres}
        04_histosweep/
        05_plots/
        tmp/

The two intermediate directory names that the legacy scripts hard-code
(``UNI_multiscale_patches`` and ``processed_pkl`` / ``processed_pkl_test_highres``)
are deliberately preserved as *inner* directory names so that the unchanged
legacy scripts keep working when they are pointed at a single stage directory.
Only the top level is reorganised.

This module is pure standard library.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .errors import LayoutError
from .version import LAYOUT_FORMAT, LAYOUT_FORMAT_VERSION

#: Directory created under the project path that holds every run.
RUNS_DIRNAME = "aurora_runs"

#: Names of the stage directories, in pipeline order.
STAGE_DIRS: Dict[str, str] = {
    "segmentation": "01_segmentation",
    "features": "02_features",
    "predictions": "03_predictions",
    "histosweep": "04_histosweep",
    "plots": "05_plots",
}

#: Sub-directories of ``01_segmentation``.
SEGMENTATION_SUBDIRS: Tuple[str, ...] = ("masks", "contours", "patches", "viz")

#: Sub-directories of ``02_features/UNI_multiscale_patches``.
FEATURE_SUBDIRS: Tuple[str, ...] = ("wsis_rescaled", "check_figure", "locs")

#: Directory names the legacy scripts hard-code underneath a stage directory.
LEGACY_FEATURE_BUNDLE_DIRNAME = "UNI_multiscale_patches"
LEGACY_LOWRES_CACHE_DIRNAME = "processed_pkl"
LEGACY_HIGHRES_CACHE_DIRNAME = "processed_pkl_test_highres"

#: Output sub-directory name for predictions (legacy default is ``pred``).
PRED_SUBDIR = "pred"

#: Directory holding model weights, their assets and the training outputs.
CHECKPOINTS_DIRNAME = "checkpoints"

_SLUG_RE = re.compile(r"[^0-9A-Za-z._-]+")
_RUN_NAME_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]*$")


def slugify(value: str, fallback: str = "run") -> str:
    """Turn an arbitrary string into a safe single path component."""
    slug = _SLUG_RE.sub("-", str(value).strip()).strip("-.")
    slug = slug or fallback
    return slug[:120]


def validate_run_name(name: str) -> str:
    """Return ``name`` if it is a safe directory name, else raise."""
    if not name or not _RUN_NAME_RE.match(name) or name in (".", ".."):
        raise LayoutError(
            f"invalid run name {name!r}",
            hint=(
                "Run names must start with a letter or digit and contain only "
                "letters, digits, '.', '_' and '-'. Example: --run BRCA_cohort_2025"
            ),
        )
    return name


@dataclass(frozen=True)
class RunLayout:
    """Resolves every path the new pipeline is allowed to write to.

    Instances are cheap and immutable; build one with :meth:`create` (or
    :meth:`from_root` when the directory already exists).
    """

    #: Absolute path of the user's project directory (holds the input data).
    project_path: Path
    #: Name of the run; the run directory is ``project_path/aurora_runs/<name>``.
    run_name: str
    #: Absolute path of the run directory.
    root: Path
    #: Set when the run directory lives somewhere other than the default.
    root_is_custom: bool = False

    # ---------------------------------------------------------------- builders
    @staticmethod
    def default_root(project_path: os.PathLike | str, run_name: str) -> Path:
        validate_run_name(run_name)
        return Path(project_path).expanduser().resolve() / RUNS_DIRNAME / run_name

    @classmethod
    def create(
        cls,
        project_path: os.PathLike | str,
        run_name: str,
        root: Optional[os.PathLike | str] = None,
    ) -> "RunLayout":
        """Build a layout object, resolving the default root when ``root`` is None."""
        project = Path(project_path).expanduser().resolve()
        validate_run_name(run_name)
        if root is None:
            layout = cls(project, run_name, cls.default_root(project, run_name), False)
        else:
            layout = cls(project, run_name, Path(root).expanduser().resolve(), True)
        return layout

    @classmethod
    def from_root(cls, root: os.PathLike | str, project_path: Optional[os.PathLike | str] = None) -> "RunLayout":
        """Rebuild a layout from an existing run directory."""
        root_path = Path(root).expanduser().resolve()
        project = (
            Path(project_path).expanduser().resolve()
            if project_path is not None
            else root_path.parent.parent
        )
        return cls(project, root_path.name, root_path, root_path != cls.default_root(project, root_path.name))

    # ------------------------------------------------------------------ layout
    @property
    def run_json(self) -> Path:
        return self.root / "run.json"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def checkpoints_dir(self) -> Path:
        """Where model weights and their assets live.

        The training stages use this directory as their ``train_path``, so the
        gene list, the cell types, the parameters and the trained weights are all
        in one place -- and it is where ``combine_checkpoints`` writes the
        combined checkpoint that inference then consumes.
        """
        return self.root / CHECKPOINTS_DIRNAME

    @property
    def train_dir(self) -> Path:
        """Training output inside the run; the same directory as ``checkpoints``.

        The training scripts take ``-t`` (``train_path``) and write their interim
        and final weights there, which is why it must stay the checkpoint directory.
        """
        return self.checkpoints_dir

    def stage_dir(self, stage_group: str) -> Path:
        try:
            return self.root / STAGE_DIRS[stage_group]
        except KeyError as exc:  # pragma: no cover - programming error
            raise LayoutError(
                f"unknown stage group {stage_group!r}",
                hint=f"Known groups: {', '.join(STAGE_DIRS)}",
            ) from exc

    # -- segmentation ------------------------------------------------------- #
    @property
    def segmentation_dir(self) -> Path:
        return self.stage_dir("segmentation")

    @property
    def masks_dir(self) -> Path:
        return self.segmentation_dir / "masks"

    @property
    def contours_dir(self) -> Path:
        return self.segmentation_dir / "contours"

    @property
    def patches_dir(self) -> Path:
        return self.segmentation_dir / "patches"

    @property
    def segmentation_viz_dir(self) -> Path:
        return self.segmentation_dir / "viz"

    # -- features ----------------------------------------------------------- #
    @property
    def features_dir(self) -> Path:
        """The ``-o`` value handed to the legacy feature scripts."""
        return self.stage_dir("features")

    @property
    def feature_bundle_dir(self) -> Path:
        """``02_features/UNI_multiscale_patches`` -- legacy ``save_dir``."""
        return self.features_dir / LEGACY_FEATURE_BUNDLE_DIRNAME

    @property
    def rescaled_dir(self) -> Path:
        return self.feature_bundle_dir / "wsis_rescaled"

    @property
    def check_figure_dir(self) -> Path:
        return self.feature_bundle_dir / "check_figure"

    def embeddings_dir(self, patch_size: int) -> Path:
        return self.feature_bundle_dir / f"emb_{int(patch_size)}"

    # -- predictions -------------------------------------------------------- #
    @property
    def predictions_dir(self) -> Path:
        return self.stage_dir("predictions")

    def prediction_dir(self, resolution: str) -> Path:
        """``03_predictions/<lowres|highres>`` -- the legacy ``-t`` value."""
        if resolution not in ("lowres", "highres"):
            raise LayoutError(
                f"unknown resolution {resolution!r}",
                hint="Use 'lowres' or 'highres'.",
            )
        return self.predictions_dir / resolution

    def prediction_output_dir(self, resolution: str) -> Path:
        return self.prediction_dir(resolution) / PRED_SUBDIR

    def prediction_cache_dir(self, resolution: str) -> Path:
        return self.prediction_dir(resolution) / (
            LEGACY_LOWRES_CACHE_DIRNAME if resolution == "lowres" else LEGACY_HIGHRES_CACHE_DIRNAME
        )

    # -- optional stages ---------------------------------------------------- #
    @property
    def histosweep_dir(self) -> Path:
        return self.stage_dir("histosweep")

    @property
    def plots_dir(self) -> Path:
        return self.stage_dir("plots")

    # ------------------------------------------------------------------ helpers
    def all_directories(self) -> List[Path]:
        """Every directory :meth:`ensure` creates."""
        dirs = [
            self.root,
            self.config_dir,
            self.logs_dir,
            self.tmp_dir,
            self.state_dir,
            self.segmentation_dir,
            self.segmentation_viz_dir,
            self.features_dir,
            self.feature_bundle_dir,
            self.rescaled_dir,
            self.check_figure_dir,
            self.predictions_dir,
            self.prediction_dir("lowres"),
            self.prediction_output_dir("lowres"),
            self.prediction_cache_dir("lowres"),
            self.prediction_dir("highres"),
            self.prediction_output_dir("highres"),
            self.prediction_cache_dir("highres"),
            self.histosweep_dir,
            self.plots_dir,
        ]
        dirs.extend(self.segmentation_dir / sub for sub in SEGMENTATION_SUBDIRS)
        return dirs

    def shared_directories(self) -> List[Path]:
        """Directories every run needs, whatever the mode."""
        return [
            self.root,
            self.config_dir,
            self.logs_dir,
            self.tmp_dir,
            self.state_dir,
            self.checkpoints_dir,
        ]

    def inference_directories(self) -> List[Path]:
        """Directories only an inference run fills."""
        dirs = [
            self.segmentation_dir,
            self.segmentation_viz_dir,
            self.rescaled_dir,
            self.check_figure_dir,
            self.predictions_dir,
            self.prediction_dir("lowres"),
            self.prediction_output_dir("lowres"),
            self.prediction_cache_dir("lowres"),
            self.prediction_dir("highres"),
            self.prediction_output_dir("highres"),
            self.prediction_cache_dir("highres"),
            self.histosweep_dir,
            self.plots_dir,
        ]
        dirs.extend(self.segmentation_dir / sub for sub in SEGMENTATION_SUBDIRS)
        return dirs

    def directories_for(self, mode: Optional[str] = None) -> List[Path]:
        """The directories a run of ``mode`` uses, in a stable order.

        Feature extraction and the rescaled slides serve both modes, so they are
        always created; the inference-only ones are not.
        """
        dirs = self.shared_directories() + [self.features_dir, self.feature_bundle_dir]
        if mode != "training":
            dirs += self.inference_directories()
        ordered: List[Path] = []
        for directory in dirs:
            if directory not in ordered:
                ordered.append(directory)
        return ordered

    def ensure(self, mode: Optional[str] = None) -> "RunLayout":
        """Create the run's directories (idempotent).

        ``mode`` scopes what is created.  Without it every directory in the layout
        is made.  With ``"training"`` the inference-only ones -- ``01_segmentation``,
        ``03_predictions``, ``04_histosweep``, ``05_plots`` -- are left out, so a
        training run is not littered with empty directories for stages it never
        runs.
        """
        for directory in self.directories_for(mode) if mode else self.all_directories():
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def relativize(self, path: Path) -> str:
        """Path relative to the run root, using ``/`` separators.

        Legacy scripts build paths as ``f"{main_dir}/{arg}"``, so the run root
        is passed as ``main_dir`` and every other path is handed over in this
        form.
        """
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def stage_args(self) -> Dict[str, str]:
        """The relative arguments every stage implementation needs."""
        return {
            "run_root": self.root.as_posix(),
            "config_dir": self.relativize(self.config_dir),
            "logs_dir": self.relativize(self.logs_dir),
            "segmentation_dir": self.relativize(self.segmentation_dir),
            "masks_dir": self.relativize(self.masks_dir),
            "patches_dir": self.relativize(self.patches_dir),
            "features_dir": self.relativize(self.features_dir),
            "feature_bundle_dir": self.relativize(self.feature_bundle_dir),
            "rescaled_dir": self.relativize(self.rescaled_dir),
            "lowres_pred_dir": self.relativize(self.prediction_dir("lowres")),
            "highres_pred_dir": self.relativize(self.prediction_dir("highres")),
            "histosweep_dir": self.relativize(self.histosweep_dir),
            "plots_dir": self.relativize(self.plots_dir),
        }

    def describe(self) -> Dict[str, object]:
        """JSON-serialisable description, written into ``run.json``."""
        return {
            "format": LAYOUT_FORMAT,
            "format_version": LAYOUT_FORMAT_VERSION,
            "project_path": self.project_path.as_posix(),
            "run_name": self.run_name,
            "run_root": self.root.as_posix(),
            "custom_root": self.root_is_custom,
            "directories": {
                "segmentation": self.relativize(self.segmentation_dir),
                "masks": self.relativize(self.masks_dir),
                "patches": self.relativize(self.patches_dir),
                "features": self.relativize(self.features_dir),
                "feature_bundle": self.relativize(self.feature_bundle_dir),
                "rescaled": self.relativize(self.rescaled_dir),
                "predictions_lowres": self.relativize(self.prediction_dir("lowres")),
                "predictions_highres": self.relativize(self.prediction_dir("highres")),
                "histosweep": self.relativize(self.histosweep_dir),
                "plots": self.relativize(self.plots_dir),
                "logs": self.relativize(self.logs_dir),
            },
        }


# --------------------------------------------------------------------------- #
# Legacy layout detection / migration
# --------------------------------------------------------------------------- #

#: Directory names the *old* pipeline wrote into the project root.
LEGACY_OUTPUT_HINTS: Dict[str, str] = {
    "AURORA_interim": "low-resolution intermediate data (rescaled images + embeddings)",
    "AURORA_pred": "predictions and caches",
    "wsis_segmentation": "tissue segmentation masks/contours",
    "processed_data": "HEST-1K formatted training data",
    "raw_data": "raw Visium inputs",
    "AURORA_plots": "figures",
    "test_smooth_pred": "ad-hoc test output",
}


@dataclass
class LegacyLayoutReport:
    """What the old, flat output layout looks like inside a project directory."""

    project_path: Path
    found: Dict[str, Path] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def any_found(self) -> bool:
        return bool(self.found)

    def describe(self) -> str:
        if not self.found:
            return f"No legacy AURORA output directories found under {self.project_path}."
        lines = [f"Legacy AURORA output directories under {self.project_path}:"]
        for name, path in sorted(self.found.items()):
            lines.append(f"  - {path}  ({LEGACY_OUTPUT_HINTS.get(name, 'unknown purpose')})")
        return "\n".join(lines)


def detect_legacy_layout(project_path: os.PathLike | str, extra_names: Sequence[str] = ()) -> LegacyLayoutReport:
    """Find directories produced by the *old* pipeline inside ``project_path``."""
    project = Path(project_path).expanduser().resolve()
    report = LegacyLayoutReport(project_path=project)
    names = list(LEGACY_OUTPUT_HINTS) + list(extra_names)
    if not project.is_dir():
        report.notes.append(f"{project} is not a directory")
        return report
    for name in names:
        candidate = project / name
        if candidate.is_dir():
            report.found[name] = candidate
    return report


#: Mapping from a legacy directory to the stage group that now owns it.
LEGACY_TO_STAGE: Dict[str, str] = {
    "wsis_segmentation": "segmentation",
    "AURORA_interim": "features",
    "AURORA_pred": "predictions",
    "AURORA_plots": "plots",
}


def plan_legacy_migration(project_path: os.PathLike | str, run_name: str) -> List[Tuple[Path, Path]]:
    """Return ``(source, destination)`` pairs for :mod:`aurora.migrate`.

    Nothing is moved or deleted by this function; it only computes a plan.  The
    actual copy is performed by ``aurora migrate-layout``.
    """
    project = Path(project_path).expanduser().resolve()
    layout = RunLayout.create(project, run_name)
    report = detect_legacy_layout(project)
    plan: List[Tuple[Path, Path]] = []
    for name, source in report.found.items():
        stage = LEGACY_TO_STAGE.get(name)
        if stage is None:
            continue
        if name == "AURORA_interim":
            destination = layout.feature_bundle_dir
            # the legacy bundle may sit directly in the interim dir
            inner = source / LEGACY_FEATURE_BUNDLE_DIRNAME
            plan.append(((inner if inner.is_dir() else source), destination))
        elif name == "AURORA_pred":
            for resolution, dirname in (
                ("lowres", LEGACY_LOWRES_CACHE_DIRNAME),
                ("highres", LEGACY_HIGHRES_CACHE_DIRNAME),
            ):
                cache = source / dirname
                if cache.is_dir():
                    plan.append((cache, layout.prediction_cache_dir(resolution)))
            pred = source / PRED_SUBDIR
            if pred.is_dir():
                plan.append((pred, layout.prediction_output_dir("lowres")))
        elif name == "wsis_segmentation":
            plan.append((source, layout.segmentation_dir))
        elif name == "AURORA_plots":
            plan.append((source, layout.plots_dir))
    return plan


def write_layout_stamp(layout: RunLayout) -> Path:
    """Write ``state/layout.json`` describing the layout (idempotent).

    Creates only ``state/``: calling ``layout.ensure()`` here used to fill in every
    directory of the layout, including the inference-only ones, so a training run
    ended up with empty ``01_segmentation`` and ``05_plots``.
    """
    layout.state_dir.mkdir(parents=True, exist_ok=True)
    path = layout.state_dir / "layout.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(layout.describe(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)
    return path
