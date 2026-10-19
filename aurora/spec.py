"""Declarative description of every pipeline stage.

A stage spec answers four questions:

* **what** it does (``summary``),
* **which dependency group** it needs (``requires_group``),
* **what must have run before it** (``prerequisites``) -- this is where the
  reorganisation's requirement #2 lives: ``predict_highres`` lists
  ``predict_lowres``, so the high-resolution model can never run on its own,
* **which files it produces** (``outputs``), so a stage can only be recorded as
  complete when its artefacts really exist on disk.

The prerequisite graph is also stored inside every combined checkpoint
(``manifest.requires``), and :func:`aurora.state.assert_prerequisites` checks the
two agree, so a checkpoint that claims a different ordering cannot silently
bypass the gate.

Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .errors import ConfigError, format_error_list
from .paths import CHECKPOINTS_DIRNAME, RunLayout
from .samples import SampleSheet

# --------------------------------------------------------------------------- #
# Stage context
# --------------------------------------------------------------------------- #


@dataclass
class StageContext:
    """Everything a stage needs to know about the run it belongs to."""

    layout: RunLayout
    samples: SampleSheet
    #: Model parameters of the low-resolution stage (from the combined checkpoint).
    lowres_params: Dict[str, object] = field(default_factory=dict)
    #: Model parameters of the high-resolution stage.
    highres_params: Dict[str, object] = field(default_factory=dict)
    #: Free-form per-stage options coming from the CLI.
    options: Dict[str, object] = field(default_factory=dict)
    #: The :class:`aurora.config.RunConfig` that produced this context.  Typed
    #: access for the command builders; ``options`` stays the untyped view.
    config: Any = None

    # -- sample helpers ----------------------------------------------------- #
    @property
    def sample_names(self) -> List[str]:
        return self.samples.sample_names

    def image_stem(self, sample_name: str) -> str:
        """Name the segmentation stage uses for a slide (extension stripped once)."""
        sample = self.samples.by_name(sample_name)
        image = sample.image_name or sample.sample_name
        return image.rsplit(".", 1)[0] if "." in image else image

    @property
    def lowres_patch_sizes(self) -> List[int]:
        sizes = self.lowres_params.get("patch_size") or self.lowres_params.get("patch_sizes")
        if not sizes:
            raise ConfigError(
                "the run does not record the low-resolution patch sizes",
                hint=(
                    "Pass --param patch_size=3584,896,224, or use a combined checkpoint "
                    "whose manifest records them."
                ),
            )
        return [int(x) for x in sizes]  # type: ignore[arg-type]

    @property
    def highres_patch_sizes(self) -> List[int]:
        sizes = self.highres_params.get("patch_size") or self.highres_params.get("patch_sizes")
        if not sizes:
            raise ConfigError(
                "the run does not record the high-resolution patch sizes",
                hint=(
                    "Pass --param patch_size=56,14, or use a combined checkpoint whose "
                    "manifest records them."
                ),
            )
        return [int(x) for x in sizes]  # type: ignore[arg-type]

    def option(self, key: str, default: object = None) -> object:
        return self.options.get(key, default)


# --------------------------------------------------------------------------- #
# Stage specification
# --------------------------------------------------------------------------- #

#: Stage categories, used by the CLI to group the ``--help`` output.
MODE_INFERENCE = "inference"
MODE_TRAINING = "training"
MODE_BOTH = "both"

#: Output resolver signature.
OutputResolver = Callable[[StageContext], List[Path]]

#: Grouped-output resolver: each inner list holds the acceptable paths for one
#: output, the first entry being the canonical one that gets recorded.
OutputGroupResolver = Callable[[StageContext], List[List[Path]]]

#: Post-run hook: called after a stage's commands succeed and before its outputs
#: are verified.  Returns human-readable notes for the stage log.
PostRunHook = Callable[[StageContext], List[str]]


@dataclass(frozen=True)
class StageSpec:
    """One runnable step of the pipeline."""

    name: str
    summary: str
    mode: str
    #: Stage names that must be ``complete`` before this stage may start.
    prerequisites: Tuple[str, ...] = ()
    #: Dependency group (see :mod:`aurora.deps`) needed to execute the stage.
    requires_group: str = "inference"
    #: Concrete files this stage must have produced when it reports success.
    outputs: OutputResolver = lambda ctx: []
    #: Directories that must exist and be non-empty when the stage reports success.
    output_dirs: OutputResolver = lambda ctx: []
    #: Optional richer form of :attr:`outputs` for stages whose artefacts may be
    #: spelled in more than one way (see :mod:`aurora.naming`).  When set, each
    #: inner list is a set of acceptable paths and the first entry is canonical.
    output_groups: Optional[OutputGroupResolver] = None
    #: Optional hook run after the commands succeed, to normalise what they
    #: produced (e.g. link a mask to its canonical name).
    post_run: Optional[PostRunHook] = None
    #: ``True`` for stages that are useful but not part of the default chain.
    optional: bool = False
    #: Name of the legacy script this stage drives, when it drives one.
    legacy_script: str = ""
    #: ``True`` when the stage runs in-process inside the aurora package.
    native: bool = False
    #: Hash of the arguments that change a stage's *content*, if any.
    #:
    #: A stage is normally reused when it completed and its outputs are present, which
    #: is wrong for a stage whose output depends on the command line: asking ``plot``
    #: for different genes must not return the previous figures.  When this is set, a
    #: run is only reused if the recorded fingerprint still matches.
    args_fingerprint: Optional[Callable[[StageContext], str]] = None
    #: ``True`` when the stage consumes the combined checkpoint -- its model
    #: parameters, its bundled gene list / cell types / normalisation factors, or
    #: its weights.  A stage that only converts raw data into another format does
    #: not, and should not force the user to name a ``.pth``.
    needs_model: bool = True
    #: Prerequisites that apply only in training mode, when they differ.
    #:
    #: ``rescale`` is the reason this exists: the inference invocation passes
    #: ``--seg_path`` and therefore needs the tissue mask from ``segment``, while the
    #: training invocation works from ``processed_data`` and passes no mask at all.
    #: Leaving it unconditional dragged ``segment`` -- and its requirement for a
    #: ``wsis/`` directory of slides -- into every training run.
    #:
    #: ``None`` means "same as ``prerequisites``"; ``()`` means "none in training".
    training_prerequisites: Optional[Tuple[str, ...]] = None
    #: Environment group name reported by ``aurora doctor``.
    docs_anchor: str = ""

    def __post_init__(self) -> None:
        """Reject a literal list where a resolver is required.

        ``outputs=[]`` instead of ``outputs=lambda ctx: []`` only fails when the
        stage actually runs, as ``TypeError: 'list' object is not callable`` far
        from the mistake.  Fail at import time instead, for every stage -- the
        training stages are not covered by the inference-focused tests.
        """
        for field_name in ("outputs", "output_dirs"):
            value = getattr(self, field_name)
            if not callable(value):
                raise TypeError(
                    f"stage '{self.name}': {field_name} must be a callable taking a "
                    f"StageContext, got {type(value).__name__}; "
                    f"write 'lambda ctx: [...]' instead of a bare list"
                )
        if self.output_groups is not None and not callable(self.output_groups):
            raise TypeError(
                f"stage '{self.name}': output_groups must be a callable taking a StageContext, "
                f"got {type(self.output_groups).__name__}"
            )

    def describe(self) -> str:
        flags = []
        if self.optional:
            flags.append("optional")
        if self.native:
            flags.append("native")
        if self.legacy_script:
            flags.append(f"drives {self.legacy_script}")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        return f"{self.name:<24} {self.summary}{suffix}"


# --------------------------------------------------------------------------- #
# Output resolvers
# --------------------------------------------------------------------------- #


def _segmentation_outputs(ctx: StageContext) -> List[Path]:
    """Canonical mask per sample: keyed by ``Sample_name``.

    AURORA keys every per-sample artefact by ``Sample_name``.  The only place
    that disagrees is ``code/create_patches_fp.py``, which names its masks after
    the slide file; :func:`aurora.naming.reconcile_segmentation_names` links the
    two spellings together after the stage runs, and
    :func:`aurora.naming.segmentation_output_groups` lets the gate accept either.
    """
    return [
        ctx.layout.masks_dir / f"{name}.geojson" for name in ctx.sample_names
    ]


def _segmentation_output_groups(ctx: StageContext) -> List[List[Path]]:
    """Acceptable mask paths per sample, ``Sample_name`` spelling first."""
    from .naming import segmentation_output_groups

    return segmentation_output_groups(ctx.layout, ctx.samples)


def _reconcile_segmentation(ctx: StageContext) -> List[str]:
    """Link ``<image_stem>.geojson`` and ``<Sample_name>.geojson`` together."""
    from .naming import reconcile_segmentation_names

    return reconcile_segmentation_names(ctx.layout, ctx.samples)


def _rescaled_outputs(ctx: StageContext) -> List[Path]:
    rescaled = ctx.layout.rescaled_dir
    paths: List[Path] = []
    for name in ctx.sample_names:
        paths.append(rescaled / f"{name}_rescaled.tif")
        paths.append(rescaled / f"{name}_mask_rescaled.tif")
    return paths


def _lowres_feature_outputs(ctx: StageContext) -> List[Path]:
    paths: List[Path] = []
    for patch_size in ctx.lowres_patch_sizes:
        directory = ctx.layout.embeddings_dir(patch_size)
        paths.extend(directory / f"{name}.npz" for name in ctx.sample_names)
    return paths


def _highres_feature_outputs(ctx: StageContext) -> List[Path]:
    paths: List[Path] = []
    for patch_size in ctx.highres_patch_sizes:
        directory = ctx.layout.embeddings_dir(patch_size)
        paths.extend(directory / f"{name}.npz" for name in ctx.sample_names)
    return paths


def _lowres_prediction_outputs(ctx: StageContext) -> List[Path]:
    directory = ctx.layout.prediction_output_dir("lowres")
    return [directory / f"{name}-pred.pickle" for name in ctx.sample_names]


def _highres_prediction_outputs(ctx: StageContext) -> List[Path]:
    """The exact file ``predict_highres`` writes, given the two output flags.

    ``predict_highres`` selects its filename from ``--uint8-output`` *and*
    ``--compress-output`` (see :data:`aurora.naming.HIGHRES_SUFFIXES`), so the
    declared output has to use the same helper or the stage fails on a file that
    was never meant to be written.
    """
    from .naming import highres_output_path

    directory = ctx.layout.prediction_output_dir("highres")
    uint8 = bool(ctx.option("uint8_output", False))
    compress = bool(ctx.option("compress_output", False))
    return [
        highres_output_path(directory, name, uint8, compress) for name in ctx.sample_names
    ]


def _histosweep_outputs(ctx: StageContext) -> List[Path]:
    """The tissue masks HistoSweep leaves behind, which predict_highres reads.

    Not HistoSweep's own intermediate outputs: ``run_HistoSweep.py`` deletes those
    after a successful run, so verifying them would fail every time.
    """
    from .naming import histosweep_mask_path

    return [histosweep_mask_path(ctx.layout.rescaled_dir, name) for name in ctx.sample_names]


def _plot_output_dir(ctx: StageContext) -> Path:
    """Where the plot stage writes: high-resolution figures get their own folder.

    Both kinds would otherwise write ``<sample>-<gene>-<resolution>pixels.png``
    into the same tree, and the stage verifies this directory is non-empty.
    """
    kind = str(ctx.option("plot_predictions", "lowres"))
    if kind == "highres":
        return ctx.layout.plots_dir / "highres"
    return ctx.layout.plots_dir


def _plot_outputs(ctx: StageContext) -> List[Path]:
    return [_plot_output_dir(ctx)]


def _plot_args_fingerprint(ctx: StageContext) -> str:
    """Hash of what the plot stage was asked to draw.

    The figures are chosen entirely on the command line, so completing the stage once
    must not make it reusable for a different gene or cell-type list: that returned the
    previous figures and reported success.  Anything that changes the pictures is part
    of the fingerprint.
    """
    import hashlib

    parts = [
        str(ctx.option("plot_predictions", "lowres")),
        str(ctx.option("plot_resolution", "")),
        str(ctx.option("plot_downscale", "")),
        # Sorted: the order on the command line does not change the figures.
        ",".join(sorted(str(g) for g in (ctx.option("gene_names") or []))),
        ",".join(sorted(str(c) for c in (ctx.option("cell_types") or []))),
        ",".join(sorted(ctx.sample_names)),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _train_path(ctx: StageContext) -> Path:
    """Training output directory (`-t`), which also holds the checkpoint assets."""
    name = str(ctx.option("train_dir", CHECKPOINTS_DIRNAME))
    return ctx.layout.root / name


def _lowres_checkpoint_outputs(ctx: StageContext) -> List[Path]:
    # train.py writes AURORA_model_weights[_smooth].pth into <train_path>.
    name = (
        "AURORA_model_weights_smooth.pth"
        if ctx.option("smooth_counts")
        else "AURORA_model_weights.pth"
    )
    return [_train_path(ctx) / name]


def _highres_checkpoint_outputs(ctx: StageContext) -> List[Path]:
    # train_highres.py writes AURORA_highres_model_weights.pth into <train_path>.
    return [_train_path(ctx) / "AURORA_highres_model_weights.pth"]


#: What the combine stage writes, and what the downloader calls the ready-to-use file.
DEFAULT_COMBINED_NAME = "aurora.pth"

#: Name used by earlier versions.  Runs that finished before the change keep working,
#: because their checkpoint is still recognised as this stage's output.
LEGACY_COMBINED_NAME = "aurora_combined.pth"


def _combined_checkpoint_outputs(ctx: StageContext) -> List[Path]:
    name = str(ctx.option("combined_name", DEFAULT_COMBINED_NAME))
    path = _train_path(ctx) / name
    if not path.exists() and name == DEFAULT_COMBINED_NAME:
        # A run finished before the rename: its result is still a valid checkpoint.
        legacy = _train_path(ctx) / LEGACY_COMBINED_NAME
        if legacy.is_file():
            return [legacy]
    return [path]


def _prepare_highres_training_outputs(ctx: StageContext) -> List[Path]:
    # prepare_highres_training.py writes the low-resolution prior that
    # train_highres.py consumes, into <train_path>/train_pred_lowres/.
    directory = _train_path(ctx) / "train_pred_lowres"
    return [directory / f"{name}-pred.pickle" for name in ctx.sample_names]


def _prepare_st_outputs(ctx: StageContext) -> List[Path]:
    # `prepare_st.py` writes into `<output_path>/cnts/` as `<sample>-cnts.tsv`
    # (``save_folder = f"{main}/{output_path}/cnts/"``), not a pickle under the
    # feature bundle -- the bundle only holds the UNI embeddings and locs.
    base = ctx.layout.features_dir
    return [base / "cnts" / f"{name}-cnts.tsv" for name in ctx.sample_names]


def _smooth_cnts_outputs(ctx: StageContext) -> List[Path]:
    # `smooth_cnt.py` writes `<output_path>/<--output_folder>/<sample>-cnts.tsv`, and
    # its default folder name is `cnts_smooth`.
    base = ctx.layout.features_dir
    return [base / "cnts_smooth" / f"{name}-cnts.tsv" for name in ctx.sample_names]


def _dir_output(path_getter: Callable[[StageContext], Path]) -> OutputResolver:
    def resolver(ctx: StageContext) -> List[Path]:
        return [path_getter(ctx)]

    return resolver


# --------------------------------------------------------------------------- #
# The stage table
# --------------------------------------------------------------------------- #

STAGE_SPECS: Tuple[StageSpec, ...] = (
    # ----------------------------- inference ------------------------------ #
    StageSpec(
        name="segment",
        summary="Segment tissue on the whole-slide images (masks + contours)",
        mode=MODE_BOTH,
        prerequisites=(),
        requires_group="inference",
        outputs=_segmentation_outputs,
        output_groups=_segmentation_output_groups,
        post_run=_reconcile_segmentation,
        output_dirs=_dir_output(lambda ctx: ctx.layout.masks_dir),
        legacy_script="create_patches_fp.py",
        docs_anchor="inference-segment",
    ),
    StageSpec(
        name="rescale",
        summary="Rescale slides to the model pixel size and cut the tissue bounding box",
        mode=MODE_BOTH,
        # Inference rescales the slide and crops to the tissue mask from `segment`.
        # Training rescales the HEST-1K images in processed_data and passes no mask,
        # so it needs no segmentation and no wsis/ directory.
        prerequisites=("segment",),
        training_prerequisites=(),
        requires_group="inference",
        outputs=_rescaled_outputs,
        output_dirs=_dir_output(lambda ctx: ctx.layout.rescaled_dir),
        legacy_script="rescale.py",
        docs_anchor="inference-rescale",
    ),
    # Defined here, right after the stage it depends on, so STAGE_ORDER reflects
    # real dependency order: --auto-prerequisites plans missing stages in
    # STAGE_ORDER, and leaving HistoSweep in the optional block at the bottom
    # scheduled it after predict_lowres even though it only needs the rescaled
    # slide.
    StageSpec(
        name="histosweep",
        summary=(
            "Sweep the rescaled H&E for tissue quality and write the mask that "
            "predict_highres applies"
        ),
        mode=MODE_BOTH,
        prerequisites=("rescale",),
        requires_group="histosweep",
        outputs=_histosweep_outputs,
        optional=False,
        legacy_script="run_HistoSweep.py",
        docs_anchor="inference-histosweep",
    ),
    StageSpec(
        name="features_lowres",
        summary="Extract multi-scale UNI2-h features (3584/896/224 px by default)",
        mode=MODE_BOTH,
        prerequisites=("rescale",),
        requires_group="inference",
        outputs=_lowres_feature_outputs,
        output_dirs=_dir_output(lambda ctx: ctx.layout.feature_bundle_dir),
        legacy_script="extract_features.py",
        docs_anchor="inference-features",
    ),
    StageSpec(
        name="features_highres",
        summary="Extract high-resolution (56/14 px) features from the rescaled slides",
        mode=MODE_BOTH,
        prerequisites=("rescale", "features_lowres"),
        requires_group="inference",
        outputs=_highres_feature_outputs,
        legacy_script="extract_features_highres.py",
        docs_anchor="inference-features-highres",
    ),
    StageSpec(
        name="predict_lowres",
        summary="Run the low-resolution model to predict expression and cell-type proportions",
        mode=MODE_BOTH,
        prerequisites=("features_lowres",),
        requires_group="inference",
        outputs=_lowres_prediction_outputs,
        output_dirs=_dir_output(lambda ctx: ctx.layout.prediction_output_dir("lowres")),
        native=True,
        docs_anchor="inference-predict",
    ),
    StageSpec(
        name="predict_highres",
        summary="Run the high-resolution model to refine the low-resolution prediction",
        mode=MODE_BOTH,
        # >>> Requirement #2: the high-resolution model may never run alone. <<<
        # ``histosweep`` is a prerequisite because the high-resolution stage
        # applies its tissue mask.  Both are enforced by the fail-closed gate.
        prerequisites=("predict_lowres", "features_highres", "histosweep"),
        requires_group="inference",
        outputs=_highres_prediction_outputs,
        output_dirs=_dir_output(lambda ctx: ctx.layout.prediction_output_dir("highres")),
        native=True,
        docs_anchor="inference-predict-highres",
    ),
    StageSpec(
        name="plot",
        summary="Render gene-expression and cell-type figures",
        mode=MODE_BOTH,
        # Choosing highres does not remove the lowres requirement: the
        # high-resolution prediction is a refinement of the low-resolution one,
        # and predict_highres already fails closed without it.
        prerequisites=("predict_lowres",),
        requires_group="inference",
        outputs=_plot_outputs,
        output_dirs=_dir_output(_plot_output_dir),
        optional=True,
        args_fingerprint=_plot_args_fingerprint,
        legacy_script="plot.py",
        docs_anchor="inference-plot",
    ),
    # ------------------------------ training ------------------------------ #
    StageSpec(
        name="preprocess",
        summary="Convert raw Visium data into HEST-1K format",
        mode=MODE_TRAINING,
        prerequisites=(),
        requires_group="hest",
        outputs=lambda ctx: [],
        output_dirs=_dir_output(lambda ctx: ctx.layout.root / "processed_data"),
        optional=True,
        # It reads raw Visium data and writes HEST-1K; no checkpoint is involved.
        needs_model=False,
        legacy_script="HEST_preprocess.py",
        docs_anchor="training-preprocess",
    ),
    StageSpec(
        name="prepare_st",
        summary="Prepare spot-level expression targets from the processed data",
        mode=MODE_TRAINING,
        prerequisites=("rescale", "features_lowres"),
        requires_group="training",
        outputs=_prepare_st_outputs,
        legacy_script="prepare_st.py",
        docs_anchor="training-prepare-st",
    ),
    StageSpec(
        name="deconvolution",
        summary="Estimate cell-type proportions per spot with RCTD",
        mode=MODE_TRAINING,
        prerequisites=("preprocess",),
        requires_group="deconvolution",
        outputs=_dir_output(lambda ctx: ctx.layout.feature_bundle_dir / "deconvolution"),
        output_dirs=_dir_output(lambda ctx: ctx.layout.feature_bundle_dir / "deconvolution"),
        optional=True,
        legacy_script="get_deconvolution.R",
        docs_anchor="training-deconvolution",
    ),
    StageSpec(
        name="smooth_cnts",
        summary="Build smoothed expression targets for training",
        mode=MODE_TRAINING,
        prerequisites=("prepare_st",),
        requires_group="training",
        outputs=_smooth_cnts_outputs,
        legacy_script="smooth_cnt.py",
        docs_anchor="training-smooth-counts",
    ),
    StageSpec(
        name="train_lowres",
        summary="Train the low-resolution model",
        mode=MODE_TRAINING,
        prerequisites=("features_lowres", "prepare_st"),
        requires_group="training",
        outputs=_lowres_checkpoint_outputs,
        legacy_script="train.py",
        docs_anchor="training-lowres",
    ),
    StageSpec(
        name="finetune",
        summary="Fine-tune a trained model on additional genes",
        mode=MODE_TRAINING,
        prerequisites=("train_lowres",),
        requires_group="training",
        outputs=lambda ctx: [_train_path(ctx) / "AURORA_model_weights_finetuned.pth"],
        optional=True,
        legacy_script="finetune.py",
        docs_anchor="training-finetune",
    ),
    StageSpec(
        name="prepare_highres_training",
        summary="Assemble the high-resolution training targets from low-resolution predictions",
        mode=MODE_TRAINING,
        prerequisites=("train_lowres", "features_highres"),
        requires_group="training",
        outputs=_prepare_highres_training_outputs,
        legacy_script="prepare_highres_training.py",
        docs_anchor="training-prepare-highres",
    ),
    StageSpec(
        name="train_highres",
        summary="Train the high-resolution refinement model",
        mode=MODE_TRAINING,
        prerequisites=("prepare_highres_training",),
        requires_group="training",
        outputs=_highres_checkpoint_outputs,
        legacy_script="train_highres.py",
        docs_anchor="training-highres",
    ),
    StageSpec(
        name="combine_checkpoints",
        summary="Merge the trained low- and high-resolution weights into one combined .pth",
        mode=MODE_TRAINING,
        prerequisites=("train_lowres", "train_highres"),
        requires_group="core",
        outputs=_combined_checkpoint_outputs,
        native=True,
        docs_anchor="training-combine",
    ),
)

#: Name -> spec.
STAGES: Dict[str, StageSpec] = {spec.name: spec for spec in STAGE_SPECS}

#: Stage names in declaration (= pipeline) order.
STAGE_ORDER: Tuple[str, ...] = tuple(spec.name for spec in STAGE_SPECS)


def get_stage(name: str) -> StageSpec:
    """Look up a stage, raising a helpful error for unknown names."""
    try:
        return STAGES[name]
    except KeyError as exc:
        raise ConfigError(
            f"unknown stage {name!r}",
            hint="Known stages:\n" + format_error_list(STAGE_ORDER),
        ) from exc


def stages_for_mode(mode: str) -> List[StageSpec]:
    """Every stage usable in ``'inference'`` or ``'training'`` mode."""
    if mode not in (MODE_INFERENCE, MODE_TRAINING):
        raise ConfigError(
            f"unknown mode {mode!r}",
            hint=f"Use '{MODE_INFERENCE}' or '{MODE_TRAINING}'.",
        )
    return [spec for spec in STAGE_SPECS if spec.mode in (mode, MODE_BOTH)]


def stage_names_for_mode(mode: str, include_optional: bool = True) -> List[str]:
    return [spec.name for spec in stages_for_mode(mode) if include_optional or not spec.optional]


#: The default inference chain, in order.  ``predict_highres`` is included.
#: ``histosweep`` is mandatory: it masks out background / low-quality tissue and
#: ``predict_highres`` applies that mask, so skipping it would silently change
#: which patches are predicted.
INFERENCE_PIPELINE: Tuple[str, ...] = (
    "segment",
    "rescale",
    "histosweep",
    "features_lowres",
    "features_highres",
    "predict_lowres",
    "predict_highres",
)

#: The two-stage prediction chain that requirement #2 is about.
PREDICTION_CHAIN: Tuple[str, ...] = ("predict_lowres", "predict_highres")

#: The default training chain, in order.
TRAINING_PIPELINE: Tuple[str, ...] = (
    "rescale",
    "features_lowres",
    "prepare_st",
    "smooth_cnts",
    "train_lowres",
    "features_highres",
    "prepare_highres_training",
    "train_highres",
    "combine_checkpoints",
)


def resolve_execution_order(
    requested: Sequence[str],
    include_prerequisites: bool = True,
    mode: Optional[str] = None,
) -> List[str]:
    """Expand ``requested`` into a runnable, dependency-ordered list.

    Raises :class:`~aurora.errors.ConfigError` when a stage name is unknown, and
    detects cycles in the prerequisite graph.
    """
    ordered: List[str] = []
    visiting: List[str] = []

    def visit(name: str) -> None:
        spec = get_stage(name)
        if name in ordered:
            return
        if name in visiting:
            cycle = " -> ".join(visiting[visiting.index(name):] + [name])
            raise ConfigError(
                f"the stage prerequisite graph contains a cycle: {cycle}",
                hint="This is a bug in aurora/spec.py; please report it.",
            )
        visiting.append(name)
        if include_prerequisites:
            for prerequisite in prerequisites_for(name, mode):
                visit(prerequisite)
        visiting.pop()
        ordered.append(name)

    for name in requested:
        visit(name)
    return [n for n in STAGE_ORDER if n in ordered]


def prerequisites_for(stage: str, mode: Optional[str] = None) -> Tuple[str, ...]:
    """The prerequisites of ``stage`` in ``mode``.

    ``rescale`` needs ``segment`` only for inference; a training run must not be
    given an inference-only prerequisite, because that drags in ``wsis/`` and fails
    on data that is already in HEST-1K format.
    """
    spec = get_stage(stage)
    if mode == MODE_TRAINING and spec.training_prerequisites is not None:
        return tuple(spec.training_prerequisites)
    return tuple(spec.prerequisites)


def prerequisites_of(
    stage: str, recursive: bool = False, mode: Optional[str] = None
) -> List[str]:
    """Prerequisites of ``stage``, in pipeline order."""
    spec = get_stage(stage)
    if not recursive:
        return [n for n in STAGE_ORDER if n in prerequisites_for(stage, mode)]
    collected: List[str] = []
    for prerequisite in prerequisites_for(stage, mode):
        for name in prerequisites_of(prerequisite, recursive=True, mode=mode):
            if name not in collected:
                collected.append(name)
        if prerequisite not in collected:
            collected.append(prerequisite)
    return [n for n in STAGE_ORDER if n in collected]


def dependents_of(stage: str) -> List[str]:
    """Stages that would break if ``stage`` were re-run or removed."""
    return [
        spec.name
        for spec in STAGE_SPECS
        if stage in spec.prerequisites or stage in (spec.training_prerequisites or ())
    ]


def stages_needing_model(stages: Sequence[str]) -> List[str]:
    """The stages in ``stages`` (plus prerequisites) that need a checkpoint.

    Used to decide whether ``--model`` is required at all: a run that only
    converts raw data should not have to name a ``.pth``.  Prerequisites count,
    because ``--auto-prerequisites`` may add them.
    """
    considered = resolve_execution_order(stages, include_prerequisites=True)
    return [name for name in considered if get_stage(name).needs_model]


def requires_model(stages: Sequence[str]) -> bool:
    """``True`` when running ``stages`` will need the combined checkpoint."""
    return bool(stages_needing_model(stages))


def stage_requires_group(stage: str) -> str:
    return get_stage(stage).requires_group


def describe_stages(mode: Optional[str] = None, include_optional: bool = True) -> str:
    """Human-readable table used by ``aurora stages``."""
    specs = STAGE_SPECS if mode is None else stages_for_mode(mode)
    if not include_optional:
        specs = [s for s in specs if not s.optional]
    lines: List[str] = []
    for spec in specs:
        lines.append(spec.describe())
        requires = prerequisites_for(spec.name, mode)
        if requires:
            lines.append(f"{'':<24}   requires: {', '.join(requires)}")
        elif spec.prerequisites:
            # inference-only prerequisites, shown so the difference is visible
            lines.append(
                f"{'':<24}   requires: none in training "
                f"(inference: {', '.join(spec.prerequisites)})"
            )
    return "\n".join(lines)


def declared_output_groups(ctx: StageContext, stage: str) -> List[List[Path]]:
    """Acceptable paths for each declared output of ``stage``, canonical first.

    For most stages a group holds exactly one path.  Stages that accept more than
    one spelling (see :mod:`aurora.naming`) list the alternatives here.
    """
    spec = get_stage(stage)
    if spec.output_groups is not None:
        return spec.output_groups(ctx)
    return [[path] for path in spec.outputs(ctx)]


def check_outputs_exist(ctx: StageContext, stage: str) -> Tuple[bool, List[Path]]:
    """Return ``(all_present, missing)`` for the declared outputs of ``stage``.

    An output counts as present when *any* acceptable spelling of it exists; the
    reported ``missing`` paths are always the canonical ones.
    """
    spec = get_stage(stage)
    missing: List[Path] = []
    for group in declared_output_groups(ctx, stage):
        if not any(path.exists() for path in group):
            missing.append(group[0])
    for directory in spec.output_dirs(ctx):
        if not directory.is_dir() or not any(directory.iterdir()):
            missing.append(directory)
    return (not missing), missing


# --------------------------------------------------------------------------- #
# Third-party packages each stage actually imports
# --------------------------------------------------------------------------- #
#
# This is the precise per-stage view of the inference/training split and it is
# what 'aurora run' checks before starting a stage.  It is a strict subset of the
# stage's dependency group: 'segment' needs OpenSlide but not PyTorch, so an
# inference-only user who only wants masks does not need torch installed.
#
# tools/audit_inference_deps.py cross-checks these lists against the real import
# graph of the inference entry points.

STAGE_PACKAGES: Dict[str, Tuple[str, ...]] = {
    "segment": ("numpy", "pandas", "tqdm", "cv2", "openslide", "matplotlib", "PIL", "h5py", "scipy", "yaml"),
    "rescale": ("numpy", "pandas", "tqdm", "tifffile", "rasterio", "cv2", "openslide", "PIL"),
    "features_lowres": (
        "torch", "numpy", "pandas", "tqdm", "tifffile", "cv2", "timm", "huggingface_hub", "PIL", "scipy",
        "torchvision",
    ),
    "features_highres": (
        "torch", "numpy", "pandas", "tqdm", "tifffile", "cv2", "timm", "huggingface_hub", "PIL", "yaml",
        "scipy", "torchvision",
    ),
    "predict_lowres": ("torch", "numpy", "pandas", "tqdm", "scipy"),
    "predict_highres": ("torch", "numpy", "pandas", "h5py", "tqdm", "scipy"),
    "histosweep": ("numpy", "pandas", "skimage", "matplotlib", "PIL", "tifffile"),
    "plot": ("numpy", "pandas", "cv2", "matplotlib", "tifffile"),
    "preprocess": ("pandas", "pyvips", "hest"),
    "prepare_st": ("numpy", "pandas", "scanpy", "tqdm"),
    "deconvolution": (),
    "smooth_cnts": ("numpy", "pandas", "scanpy"),
    "train_lowres": ("torch", "numpy", "pandas", "tqdm", "tensorboard", "scipy", "tifffile", "cv2"),
    "finetune": ("torch", "numpy", "pandas", "tqdm", "tensorboard", "scipy", "tifffile", "cv2"),
    "prepare_highres_training": ("torch", "numpy", "pandas", "tqdm", "scipy"),
    "train_highres": ("torch", "numpy", "pandas", "tqdm", "tensorboard", "scipy", "tifffile", "cv2"),
    "combine_checkpoints": (),
}


def stage_packages(stage: str) -> Tuple[str, ...]:
    """Third-party import names ``stage`` needs (empty means standard library only)."""
    return STAGE_PACKAGES.get(stage, ())


def stage_is_inference_only(stage: str) -> bool:
    """True when the stage can run in an inference-only installation."""
    from .deps import group_of

    return all(group_of(name) in ("core", "inference") for name in stage_packages(stage))


def run_post_hook(ctx: StageContext, stage: str) -> List[str]:
    """Run the stage's post-run normalisation hook, if it has one."""
    spec = get_stage(stage)
    if spec.post_run is None:
        return []
    return list(spec.post_run(ctx))

