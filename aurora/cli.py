"""Command line interface of the AURORA unified code system.

    python -m aurora <command> [options]

Commands
--------
``run``             execute pipeline stages (inference or training)
``status``          show what a run has already completed
``stages``          list the stages, their prerequisites and their packages
``inspect``         describe a combined checkpoint
``validate``        verify that a combined checkpoint is self-consistent
``migrate-layout``  copy an old flat output tree into the new run layout
``doctor``          environment and dependency report

Everything is dependency-light: ``--help`` and ``doctor`` work in a bare
interpreter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

from . import bundle as bundle_mod
from ._legacy import describe_environment, repo_root
from .config import RunConfig
from .deps import (
    GROUP_PURPOSE,
    GROUP_REQUIREMENTS_FILE,
    check_group,
    format_missing,
    group_of,
)
from .errors import AuroraError, ConfigError
from .paths import RunLayout, detect_legacy_layout, plan_legacy_migration
from .spec import (
    INFERENCE_PIPELINE,
    MODE_INFERENCE,
    MODE_TRAINING,
    STAGE_ORDER,
    TRAINING_PIPELINE,
    describe_stages,
    get_stage,
    stage_packages,
)
from .version import __version__

PROG = "aurora"

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_UNEXPECTED = 3


# --------------------------------------------------------------------------- #
# Shared argument groups
# --------------------------------------------------------------------------- #


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project",
        "-p",
        type=str,
        default=".",
        help="Project directory that holds the input data (default: current directory)",
    )
    parser.add_argument("--run", type=str, default="run", help="Name of the run (default: run)")
    parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help="Explicit run directory (default: <project>/aurora_runs/<run>)",
    )


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Combined AURORA checkpoint (.pth). Required by 'run' and 'inspect'.",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Sample sheet CSV with Image_name / Sample_name columns",
    )


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("inference options")
    group.add_argument("--image-path", type=str, default="wsis", help="Folder with the whole-slide images")
    group.add_argument("--preset", type=str, default=None, help="Segmentation preset CSV (default: examples/tcga.csv)")
    group.add_argument("--uni-local-path", type=str, default=None, help="Local UNI2-h folder (required for high-res features)")
    group.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for downloading UNI2-h")
    group.add_argument("--batch-size", type=int, default=128, help="Feature-extraction batch size")
    group.add_argument("--num-workers", type=int, default=4, help="Data-loader / KD-tree workers")
    group.add_argument("--stride", type=int, default=112, help="Stride for the high-resolution 224 px patches")
    group.add_argument("--pixel-size", type=float, default=0.5, help="Target pixel size in micrometres")
    group.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', 'cuda' or 'cuda:0'")
    group.add_argument(
        "--sample",
        action="extend",
        nargs="+",
        default=[],
        metavar="SAMPLE",
        help="Restrict the run to these samples, e.g. --sample A B (repeatable)",
    )
    group.add_argument("--plot", action="store_true", help="Write the intermediate check figures")
    group.add_argument("--compile-mode", type=str, default="off", help="torch.compile mode for the high-res extractor")
    group.add_argument("--bulk", type=str, default=None, help="Bulk expression matrix for the low-resolution model")
    group.add_argument("--prediction-batch-size", type=int, default=100, help="Batch size of the prediction stages")
    group.add_argument(
        "--uint8-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Quantise predictions to uint8 and write the compact HDF5 layout "
            "(default: on, matching command.sh). --no-uint8-output keeps floats."
        ),
    )
    group.add_argument(
        "--only-224",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep only the finest (224 px) low-resolution level (default: on, matching "
            "command.sh --only_224). --no-only-224 keeps every level."
        ),
    )
    group.add_argument(
        "--h5-layout",
        type=str,
        default="spatial_gene_delta",
        choices=["spatial_gene_delta", "patch_major"],
        help=(
            "High-resolution HDF5 layout (default: spatial_gene_delta, which sorts rows "
            "spatially and delta-encodes them for much better compression)"
        ),
    )
    group.add_argument(
        "--h5-compression",
        type=str,
        default="zstd",
        choices=["zstd", "gzip", "none"],
        help="HDF5 filter for the high-resolution output (default: zstd; needs hdf5plugin)",
    )
    group.add_argument(
        "--h5-compression-level",
        type=int,
        default=9,
        help="Compression level for --h5-compression (default: 9)",
    )
    group.add_argument(
        "--output-format",
        type=str,
        default="modern",
        choices=["modern", "legacy"],
        help="Prediction pickle layout; unused when --uint8-output writes HDF5",
    )
    group.add_argument("--compress-output", action="store_true", help="Write high-res predictions as .npz.zst")
    group.add_argument("--gene-list", type=str, default=None, help="Override the gene list CSV")
    group.add_argument("--cell-types", type=str, default=None, help="Override the cell-type list CSV")
    group.add_argument("--norm-factors", type=str, default=None, help="Override the normalisation-factor CSV")
    group.add_argument(
        "--genetype-file",
        type=str,
        default=None,
        help=(
            "Gene-type table for prepare_st (default: examples/TCGA_genetype.csv); "
            "needs a 'gene_type' column"
        ),
    )


def _add_optional_stage_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("optional stages")
    # ``action="extend"`` with ``nargs="+"`` accepts both spellings, which is
    # what the README and run.sh show:
    #   --gene IL7R KRT8          --gene IL7R --gene KRT8
    # A plain ``action="append"`` only accepted the second, so the documented
    # form failed with "unrecognized arguments: KRT8".
    group.add_argument(
        "--gene",
        action="extend",
        nargs="+",
        default=[],
        metavar="GENE",
        help="Gene to plot; one or more, e.g. --gene IL7R KRT8 (repeatable)",
    )
    group.add_argument(
        "--celltype",
        action="extend",
        nargs="+",
        default=[],
        metavar="CELLTYPE",
        help="Cell type to plot; one or more, e.g. --celltype Malignant T_cell (repeatable)",
    )
    group.add_argument("--plot-resolution", type=int, default=224, help="Patch size used for the plots")
    group.add_argument(
        "--plot-downscale",
        type=int,
        default=7,
        help=(
            "Figure pixels = image pixels / this. 7 (the default) gives a figure "
            "1/7 of the rescaled slide; it must stay <= 7 to satisfy the minimum "
            "size, and smaller values give larger figures"
        ),
    )
    group.add_argument(
        "--plot-predictions",
        type=str,
        default="lowres",
        choices=["lowres", "highres"],
        help="Plot the low-resolution or the high-resolution predictions (default: lowres)",
    )
    group.add_argument("--histosweep-path", type=str, default=None, help="Path to the HistoSweep checkout")
    group.add_argument(
        "--histosweep-patch-size",
        type=int,
        default=14,
        help=(
            "Super-pixel size used to align high-resolution predictions with the "
            "HistoSweep tissue mask (default 14, as in code/predict_highres_TCGA.py)"
        ),
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training options")
    group.add_argument("--processed-data", type=str, default="processed_data", help="HEST-1K formatted training data")
    group.add_argument("--raw-data", type=str, default="raw_data", help="Raw Visium data for the preprocess stage")
    group.add_argument("--train-dir", type=str, default="checkpoints", help="Training output directory inside the run")
    group.add_argument("--smooth-counts", action="store_true", help="Train on the smoothed expression targets")
    group.add_argument("--finetune-gene-list", type=str, default=None, help="CSV of extra genes for the finetune stage")
    group.add_argument(
        "--param",
        action="extend",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a training parameter without editing any JSON, e.g. "
            "--param num_epochs=100 lr=0.0005 pred_celltype=false (repeatable)"
        ),
    )
    group.add_argument(
        "--pred-celltype",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Train the cell-type proportion head. On by default; "
            "--no-pred-celltype trains expression only"
        ),
    )
    group.add_argument(
        "--refresh-params",
        action="store_true",
        help="Rewrite config/params_lowres.json and checkpoints/params_highres.json from the checkpoint",
    )
    group.add_argument("--num-epochs", type=int, default=None, help="Override the number of fine-tuning epochs")
    group.add_argument("--cancer-type", type=str, default="", help="Cancer type stored in the combined checkpoint")
    group.add_argument("--model-name", type=str, default="AURORA", help="Model name stored in the combined checkpoint")
    group.add_argument("--scrna-reference", type=str, default="reference.rds", help="Seurat RDS reference for RCTD")


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("execution")
    group.add_argument(
        "--stage",
        action="extend",
        nargs="+",
        default=[],
        metavar="STAGE",
        help="Stage(s) to run, e.g. --stage predict_lowres predict_highres (repeatable)",
    )
    group.add_argument("--all", action="store_true", help="Run the whole default pipeline for the mode")
    group.add_argument(
        "--auto-prerequisites",
        action="store_true",
        help="Run the missing prerequisite stages automatically instead of failing",
    )
    group.add_argument("--force", action="store_true", help="Re-run stages even when they are already complete")
    group.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    group.add_argument("--keep-going", action="store_true", help="Continue with the next stage after a failure")
    group.add_argument("--quiet", action="store_true", help="Do not stream stage output to the console")


# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #


def discover_model(project: Path, explicit: Optional[str]) -> Path:
    """Resolve the combined checkpoint, searching the usual places."""
    if explicit:
        return Path(explicit)
    candidates: List[Path] = []
    for directory in (project / "trained_model", repo_root() / "trained_model", project):
        if directory.is_dir():
            candidates.extend(sorted(directory.glob("*.pth")))
    combined = [p for p in candidates if bundle_mod.is_combined_checkpoint(p)]
    if len(combined) == 1:
        print(f"Using the only combined checkpoint found: {combined[0]}")
        return combined[0]
    if len(combined) > 1:
        listing = "\n".join(f"  - {p}" for p in combined)
        raise ConfigError(
            f"{len(combined)} combined checkpoints were found; please choose one with --model",
            hint=listing,
        )
    searched = "\n".join(f"  - {d}" for d in {p.parent for p in candidates} or {"(none)"})
    raise ConfigError(
        "no combined AURORA checkpoint was found",
        hint=(
            "Pass --model explicitly, or build one from the legacy low/high-resolution pair:\n"
            "  python tools/merge_aurora_checkpoints.py --lowres <lowres.pth> --highres <highres.pth> "
            "--out trained_model/AURORA_<CANCER>_combined.pth\n"
            f"Searched:\n{searched}"
        ),
    )


def discover_samples(
    project: Path, run_name: str, run_root: Optional[Path], explicit: Optional[str]
) -> Optional[Path]:
    """Resolve the sample sheet, reusing the one the run already recorded.

    Resuming a run -- ``--stage plot`` on an existing run, say -- should not
    require repeating ``--samples``.  The prepared sheet is already inside the run
    at ``config/samples.csv``, so reuse it and say so, exactly as
    :func:`discover_model` does for the checkpoint.
    """
    if explicit:
        return Path(explicit)
    from .paths import RunLayout
    from .prepare import SAMPLES_FILENAME

    layout = RunLayout.create(project, run_name, run_root)
    recorded = layout.config_dir / SAMPLES_FILENAME
    if recorded.is_file():
        print(f"Using the sample sheet recorded by the run: {recorded}")
        return recorded
    return None


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def _training_param_overrides(args: argparse.Namespace) -> dict:
    """Collect ``--param`` plus the convenience flags into one override dict.

    ``--pred-celltype`` / ``--no-pred-celltype`` is the question people ask most
    often ("do I train the cell-type model?"), so it gets a flag of its own; the
    rest go through ``--param KEY=VALUE``.
    """
    from .prepare import parse_param_overrides

    overrides = parse_param_overrides(args.param or [])
    if getattr(args, "pred_celltype", None) is not None:
        overrides["pred_celltype"] = bool(args.pred_celltype)
    if getattr(args, "num_epochs", None) is not None:
        overrides["num_epochs"] = int(args.num_epochs)
    return overrides


def _package_location() -> str:
    """Where this ``aurora`` package is being imported from.

    Printed in the header because "I edited the checkout but the run behaves
    differently" is indistinguishable from a bug otherwise: an editable install, a
    ``PYTHONPATH`` entry and a copied site-packages install can all be in play.
    """
    try:
        from . import __file__ as package_file

        return str(Path(package_file).parent)
    except Exception:  # pragma: no cover - never break a run over a banner
        return "?"


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline, default_stages

    mode = args.mode
    stages = list(args.stage)
    if args.all or not stages:
        stages = list(default_stages(mode))
    for name in stages:
        get_stage(name)  # validates the name

    project = Path(args.project).expanduser().resolve()
    # A checkpoint is needed only by stages that read it.  `preprocess` reads none of
    # it, and a from-scratch training run only takes *configuration* from it -- the
    # parameter template and the three assets -- so supplying those with
    # --gene-list/--cell-types/--norm-factors makes it optional there too.  No
    # training stage loads its weights.
    from .spec import requires_model

    needs_model = requires_model(stages)
    assets_supplied = all([args.gene_list, args.cell_types, args.norm_factors])
    if args.model or (needs_model and not assets_supplied):
        try:
            model = discover_model(project, args.model)
        except ConfigError as exc:
            if needs_model and not assets_supplied:
                # Two ways out, and only one of them is a checkpoint.
                raise ConfigError(
                    exc.message,
                    hint=(exc.hint or "")
                    + (
                        "\n\nOr train from scratch without one, by supplying the three "
                        "things a checkpoint would have provided -- which is all it "
                        "contributes to a training run, since no training stage loads its "
                        "weights:\n"
                        "  --gene-list genes.csv --cell-types cell_types.csv "
                        "--norm-factors norm_factors.csv\n"
                        "The model parameters then come from aurora/defaults.py, and "
                        "--param KEY=VALUE overrides any of them."
                    ),
                ) from exc
            raise
    else:
        model = None
        if needs_model:
            print(
                "No checkpoint needed: these stages only take configuration from it, "
                "and --gene-list/--cell-types/--norm-factors supply that.\n"
                "Parameters fall back to aurora/defaults.py; override any of them with "
                "--param KEY=VALUE."
            )
        else:
            print(
                "No checkpoint needed: "
                + ", ".join(stages)
                + " does not read the model; nothing to do with a .pth."
            )
    samples = discover_samples(
        project, args.run, Path(args.run_root) if args.run_root else None, args.samples
    )

    config = RunConfig(
        project_path=project,
        model_path=model,
        samples_path=samples,
        run_name=args.run,
        run_root=Path(args.run_root) if args.run_root else None,
        mode=mode,
        device=args.device,
        dry_run=args.dry_run,
        force=args.force,
        keep_going=args.keep_going,
        auto_prerequisites=args.auto_prerequisites,
        log_to_console=not args.quiet,
        image_path=args.image_path,
        preset=args.preset,
        uni_local_path=args.uni_local_path,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stride=args.stride,
        target_pixel_size=args.pixel_size,
        plot=args.plot,
        compile_mode=args.compile_mode,
        samples_subset=list(args.sample),
        prediction_batch_size=args.prediction_batch_size,
        uint8_output=args.uint8_output,
        only_224=args.only_224,
        h5_layout=args.h5_layout,
        h5_compression=args.h5_compression,
        h5_compression_level=args.h5_compression_level,
        output_format=args.output_format,
        prediction_compression=args.compress_output,
        gene_list_path=Path(args.gene_list) if args.gene_list else None,
        cell_type_path=Path(args.cell_types) if args.cell_types else None,
        norm_factor_path=Path(args.norm_factors) if args.norm_factors else None,
        genetype_path=Path(args.genetype_file) if args.genetype_file else None,
        bulk_path=Path(args.bulk) if args.bulk else None,
        processed_data_path=args.processed_data,
        raw_data_path=args.raw_data,
        train_dir=args.train_dir,
        smooth_counts=args.smooth_counts,
        finetune_gene_list=Path(args.finetune_gene_list) if args.finetune_gene_list else None,
        num_epochs=args.num_epochs,
        param_overrides=_training_param_overrides(args),
        refresh_params=bool(args.refresh_params),
        gene_names=list(args.gene),
        cell_types=list(args.celltype),
        plot_resolution=args.plot_resolution,
        plot_downscale=args.plot_downscale,
        plot_predictions=args.plot_predictions,
        histosweep_path=args.histosweep_path,
        histosweep_patch_size=args.histosweep_patch_size,
        extra={
            "cancer_type": args.cancer_type,
            "model_name": args.model_name,
            "scrna_reference": args.scrna_reference,
        },
    )
    # No explicit check on args.samples here: Pipeline.create resolves it, which
    # also accepts the sheet an existing run has recorded.  A second check on the
    # raw argument used to live here and rejected exactly that resume case.
    # Resolution errors surface from Pipeline.create, as they do for --model.

    print(f"AURORA {__version__}  ({mode} mode)  [{_package_location()}]")
    print(config.describe())
    print()

    pipeline = Pipeline.create(config, require_model=needs_model and not assets_supplied)
    result = pipeline.run(stages, auto_prerequisites=config.auto_prerequisites, force=config.force)
    print()
    print("Stage summary:")
    print(result.describe())
    print()
    print(_next_steps(mode, config.run_name, result))
    return EXIT_OK if result.ok else EXIT_ERROR


def _next_steps(mode: str, run_name: str, result: Any) -> str:
    from .pipeline import OUTCOME_COMPLETE, OUTCOME_REUSED

    done = [o.stage for o in result.outcomes if o.outcome in (OUTCOME_COMPLETE, OUTCOME_REUSED)]
    lines = [f"Outputs are under aurora_runs/{run_name}/"]
    if mode == "inference":
        if "predict_lowres" in done or "predict_highres" in done:
            lines.append(
                f"  predictions : aurora_runs/{run_name}/03_predictions/"
                "{lowres,highres}/pred/"
            )
        lines.append(f"  logs        : aurora_runs/{run_name}/logs/")
        lines.append(f"  status      : python -m aurora status --run {run_name}")
    else:
        if "combine_checkpoints" in done:
            lines.append(
                f"  combined checkpoint : aurora_runs/{run_name}/checkpoints/aurora_combined.pth"
            )
        lines.append(f"  checkpoints : aurora_runs/{run_name}/checkpoints/")
        lines.append(f"  tensorboard : aurora_runs/{run_name}/checkpoints/train_log/")
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    from .state import RunState, format_stage_summary

    project = Path(args.project).expanduser().resolve()
    layout = RunLayout.create(project, args.run, Path(args.run_root) if args.run_root else None)
    if not layout.run_json.is_file():
        raise ConfigError(
            f"no run named '{args.run}' under {layout.root.parent}",
            hint="Start one with:  python -m aurora run --run <name> --samples <samples.csv> --model <combined.pth>",
        )
    state = RunState.load(layout)
    print(state.describe())
    print()
    print("Stage status:")
    print(format_stage_summary(state))
    if args.json:
        print()
        print(state.to_json())
    return EXIT_OK


def cmd_stages(args: argparse.Namespace) -> int:
    print(f"AURORA {__version__} pipeline stages\n")
    print(describe_stages(args.mode if args.mode else None, include_optional=not args.core_only))
    print()
    print("Default inference chain: " + " -> ".join(INFERENCE_PIPELINE))
    print("Default training chain : " + " -> ".join(TRAINING_PIPELINE))
    if args.packages:
        print()
        print("Third-party packages per stage:")
        for name in STAGE_ORDER:
            packages = stage_packages(name)
            print(f"  {name:<24} {', '.join(packages) if packages else '(standard library only)'}")
    return EXIT_OK


def cmd_inspect(args: argparse.Namespace) -> int:
    project = Path(args.project).expanduser().resolve()
    model = discover_model(project, args.model)
    kind = bundle_mod.classify_checkpoint(model)
    if not kind.is_combined:
        from .bundle import _legacy_error

        raise _legacy_error(Path(model), kind.reason or kind.kind)
    checkpoint = bundle_mod.CombinedCheckpoint.open(model)
    print(checkpoint.describe())
    if args.keys:
        from .checkpoint import load_stage_model, summarise_state_dict

        stage = args.stage or checkpoint.stages()[0]
        stage_model = load_stage_model(checkpoint, stage, device="cpu")
        print()
        print(f"state_dict of stage '{stage}':")
        print("\n".join(summarise_state_dict(stage_model.state_dict)))
    if args.assets:
        print()
        print("assets:")
        for key, entry in sorted(checkpoint.manifest.assets.items()):
            print(f"  {key:<20} {entry}")
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    project = Path(args.project).expanduser().resolve()
    model = discover_model(project, args.model)
    result = bundle_mod.validate_bundle(model, verify_payloads=not args.fast)
    print(result.describe())
    return EXIT_OK if result.ok else EXIT_ERROR


def cmd_clean(args: argparse.Namespace) -> int:
    """Report, and optionally remove, a run's intermediate files."""
    from .cleanup import apply_cleanup, find_runs, human_bytes, plan_cleanup

    project = Path(args.project).expanduser().resolve()
    if args.all_runs:
        roots = find_runs(project, Path(args.run_root) if args.run_root else None)
        if not roots:
            print(f"No runs found under {project / 'aurora_runs'}")
            return EXIT_OK
    else:
        layout = RunLayout.create(
            project, args.run, Path(args.run_root) if args.run_root else None
        )
        if not layout.root.is_dir():
            raise ConfigError(
                f"run directory {layout.root} does not exist",
                hint=(
                    "Check --run, or pass --all-runs to clean every run in "
                    f"{project / 'aurora_runs'}."
                ),
            )
        roots = [layout.root]

    total_freed = 0
    failures: List[str] = []
    sweep = bool(args.all_runs)
    for root in roots:
        layout = RunLayout.from_root(root, project)
        plan = plan_cleanup(layout, deep=args.deep)
        print(plan.describe())
        print()
        if not args.apply:
            print("  (report only; pass --apply to remove these)")
            print()
            continue
        try:
            freed = apply_cleanup(plan, force=args.force)
        except ConfigError as exc:
            # Sweeping a project legitimately meets runs that are still in
            # progress; that is worth reporting but is not a failure of the
            # command.  Naming one specific run is different.
            if sweep:
                print(f"  skipped: {exc.message}")
                print()
                continue
            failures.append(exc.format())
            print(f"  skipped: {exc.message}")
            print()
            continue
        total_freed += freed
        print(f"  removed {len(plan.items)} item(s), freed {human_bytes(freed)}")
        print()

    if args.apply and total_freed:
        print(f"Total freed: {human_bytes(total_freed)}")
    if failures:
        for message in failures:
            print(message, file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


def cmd_hest_setup(args: argparse.Namespace) -> int:
    """Point TRIDENT's checkpoint registry at HEST's segmentation weights.

    HEST segments tissue through TRIDENT, and TRIDENT refuses to start without the
    weights registered for offline use. Its error names
    ``trident/segmentation_models/local_ckpts.json`` (relative to the installed
    package) and then prints the *placeholder value* where the path belongs.
    """
    from . import hest

    for line in hest.diagnose():
        print(line)
    print()

    if args.check or not args.ckpt:
        ok, message = hest.check_setup()
        print(("OK    " if ok else "NEEDS SETUP: ") + message)
        if ok:
            return EXIT_OK
        print()
        print(f"Download {hest.SEGMENTATION_CKPT} from:\n  {hest.SEGMENTATION_CKPT_URL}")
        print(
            "then record it (this edits TRIDENT's registry in place):\n"
            f"  python -m aurora hest-setup --ckpt /path/to/{hest.SEGMENTATION_CKPT}"
        )
        return EXIT_ERROR if args.check else EXIT_OK

    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_file():
        raise ConfigError(
            f"{ckpt} does not exist",
            hint=(
                f"Download {hest.SEGMENTATION_CKPT} from:\n  {hest.SEGMENTATION_CKPT_URL}\n"
                "and pass its path with --ckpt."
            ),
        )
    target = hest.default_registry_path(args.config)
    data, key = hest.record_checkpoint(target, ckpt)
    print(f"wrote {target}")
    print(f"  repaired entry {key!r} -> {data[key]}")
    for name in sorted(data):
        if name != key:
            print(f"  kept          {name!r} -> {data[name]}")
    print()
    ok, message = hest.check_setup()
    print(("OK    " if ok else "STILL BROKEN: ") + message)
    if not ok:
        return EXIT_ERROR
    print()
    print(
        "If preprocess still cannot find the weights, TRIDENT is reading a different "
        "registry.\nSend the output above: it lists every "
        f"{hest.CKPT_JSON_NAME} found and the entries it holds."
    )
    return EXIT_OK


def cmd_params(args: argparse.Namespace) -> int:
    """Show a run's effective training parameters and how to change them."""
    import json as _json

    from .prepare import PARAMS_ASSET_KEYS, PARAMS_TYPES

    project = Path(args.project).expanduser().resolve()
    layout = RunLayout.create(project, args.run, Path(args.run_root) if args.run_root else None)
    files = {
        "lowres": layout.config_dir / "params_lowres.json",
        "highres": layout.checkpoints_dir / "params_highres.json",
    }
    stages = [args.stage] if args.stage else ["lowres", "highres"]
    print(f"Training parameters for run '{args.run}'  ({layout.root})")
    print()
    shown = 0
    for stage in stages:
        path = files[stage]
        print(f"  {stage}")
        print(f"    file : {path}")
        if not path.is_file():
            print(
                "    (not written yet; it is created on the first 'aurora run --mode training' "
                "for this run, from the checkpoint's model parameters)"
            )
            print()
            continue
        try:
            data = _json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"    cannot read it: {exc}")
            print()
            continue
        width = max(len(k) for k in data)
        for key in sorted(data):
            marker = "*" if key in PARAMS_TYPES else " "
            value = data[key]
            if isinstance(value, list):
                value = " ".join(str(v) for v in value)
            print(f"    {marker} {key:<{width}} = {value}")
        shown += 1
        print()

    asset_keys = ", ".join(sorted(PARAMS_ASSET_KEYS.values()))
    print("  '*' marks a parameter you can override on the command line.")
    print("  The asset paths (" + asset_keys + ") are kept in sync automatically.")
    print()
    print("  To change one:      --param num_epochs=100 lr=0.0005")
    print("  Cell-type head:     --pred-celltype / --no-pred-celltype")
    print("  Or edit the file:   the value is kept on the next run")
    print("  Back to defaults:   --refresh-params")
    if not shown:
        print()
        print(f"  Overridable now: {', '.join(sorted(PARAMS_TYPES))}")
    return EXIT_OK


def cmd_migrate_layout(args: argparse.Namespace) -> int:
    import shutil

    project = Path(args.project).expanduser().resolve()
    report = detect_legacy_layout(project)
    print(report.describe())
    if not report.any_found:
        return EXIT_OK
    layout = RunLayout.create(project, args.run, Path(args.run_root) if args.run_root else None)
    plan = plan_legacy_migration(project, args.run)
    if not plan:
        print("Nothing to migrate automatically; use --copy to move a specific directory.")
        return EXIT_OK
    layout.ensure()
    print()
    print(f"Migration plan (target run: {layout.root}):")
    for source, destination in plan:
        print(f"  {source}\n      -> {destination}")
    if args.dry_run:
        print("\n--dry-run: nothing was copied.")
        return EXIT_OK
    print()
    for source, destination in plan:
        if destination.exists() and any(destination.iterdir()):
            print(f"  skipping {destination} (already populated)")
            continue
        shutil.copytree(source, destination, dirs_exist_ok=True)
        print(f"  copied {source} -> {destination}")
    print()
    print("The original directories were left untouched. Re-run 'aurora status' to see the new layout.")
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"AURORA {__version__}")
    print()
    print("Environment")
    print(describe_environment())
    print()
    print("Dependency groups")
    groups = [args.group] if args.group else ["core", "inference", "training", "hest", "histosweep"]
    for group in groups:
        check = check_group(group)
        marker = "OK  " if check.ok else "MISS"
        requirement = GROUP_REQUIREMENTS_FILE.get(group) or "(manual install)"
        print(f"  [{marker}] {group:<12} {GROUP_PURPOSE.get(group, '')}")
        print(f"           requirements: {requirement}")
        if check.missing:
            print(f"           missing: {', '.join(sorted(set(check.missing)))}")
    if args.group:
        check = check_group(args.group)
        if not check.ok:
            print()
            print(format_missing(check))
            return EXIT_ERROR
    print()
    print("Inference/training split")
    offenders = []
    for name in STAGE_ORDER:
        spec = get_stage(name)
        groups = sorted({group_of(pkg) for pkg in stage_packages(name)})
        training_only = [g for g in groups if g == "training"]
        if spec.mode == MODE_INFERENCE and training_only:
            offenders.append(name)
    if offenders:
        print(f"  [FAIL] inference stages needing training-only packages: {', '.join(offenders)}")
    else:
        print("  [OK  ] no inference stage needs a training-only package")
    print()
    print("Run 'python -m aurora stages --packages' for the per-stage package list.")
    return EXIT_OK if not offenders else EXIT_ERROR


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="AURORA unified code system: combined checkpoints, gated stages, tidy outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m aurora run --run luad_infer --samples examples/Inference_samples.csv \\\n"
            "      --model trained_model/AURORA_LUAD_combined.pth\n"
            "  python -m aurora status --run luad_infer\n"
            "  python -m aurora stages --packages\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"aurora {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # run ------------------------------------------------------------------ #
    run = subparsers.add_parser("run", help="execute pipeline stages")
    _add_common_arguments(run)
    _add_input_arguments(run)
    run.add_argument("--mode", choices=[MODE_INFERENCE, MODE_TRAINING], default=MODE_INFERENCE)
    _add_inference_arguments(run)
    _add_optional_stage_arguments(run)
    _add_training_arguments(run)
    _add_execution_arguments(run)
    run.set_defaults(func=cmd_run)

    # status --------------------------------------------------------------- #
    status = subparsers.add_parser("status", help="show what a run has completed")
    _add_common_arguments(status)
    status.add_argument("--model", type=str, default=None)
    status.add_argument("--mode", choices=[MODE_INFERENCE, MODE_TRAINING], default=MODE_INFERENCE)
    status.add_argument("--json", action="store_true", help="Also print the raw run.json")
    status.set_defaults(func=cmd_status)

    # stages --------------------------------------------------------------- #
    stages = subparsers.add_parser("stages", help="list stages, prerequisites and packages")
    stages.add_argument("--mode", choices=[MODE_INFERENCE, MODE_TRAINING], default=None)
    stages.add_argument("--core-only", action="store_true", help="Hide the optional stages")
    stages.add_argument("--packages", action="store_true", help="Also list the packages per stage")
    stages.set_defaults(func=cmd_stages)

    # inspect -------------------------------------------------------------- #
    inspect = subparsers.add_parser("inspect", help="describe a combined checkpoint")
    _add_common_arguments(inspect)
    inspect.add_argument("--model", type=str, default=None)
    inspect.add_argument("--keys", action="store_true", help="Also list the state-dict tensors (needs torch)")
    inspect.add_argument("--stage", type=str, default=None, help="Stage used for --keys")
    inspect.add_argument("--assets", action="store_true", help="Also list the bundled assets")
    inspect.set_defaults(func=cmd_inspect)

    # validate ------------------------------------------------------------- #
    validate = subparsers.add_parser("validate", help="validate a combined checkpoint")
    _add_common_arguments(validate)
    validate.add_argument("--model", type=str, default=None)
    validate.add_argument("--fast", action="store_true", help="Skip re-hashing the payloads")
    validate.set_defaults(func=cmd_validate)

    # migrate-layout ------------------------------------------------------- #
    migrate = subparsers.add_parser(
        "migrate-layout", help="copy an old flat output tree into the new run layout"
    )
    _add_common_arguments(migrate)
    migrate.add_argument("--dry-run", action="store_true", help="Only print the plan")
    migrate.set_defaults(func=cmd_migrate_layout)

    # hest-setup ----------------------------------------------------------- #
    hest_cmd = subparsers.add_parser(
        "hest-setup",
        help="record HEST's tissue-segmentation checkpoint in its local_ckpts.json",
        description=(
            "HEST refuses to segment until deeplabv3_seg_v4.ckpt is listed in a file "
            "called local_ckpts.json. This finds that file (or creates it) and records "
            "where you downloaded the weights."
        ),
    )
    hest_cmd.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to the downloaded deeplabv3_seg_v4.ckpt",
    )
    hest_cmd.add_argument(
        "--config",
        type=str,
        default=None,
        help="Write this local_ckpts.json instead of the one next to HEST",
    )
    hest_cmd.add_argument(
        "--check",
        action="store_true",
        help="Only report whether HEST is ready, do not write anything",
    )
    hest_cmd.set_defaults(func=cmd_hest_setup)

    # params --------------------------------------------------------------- #
    params_cmd = subparsers.add_parser(
        "params", help="show a run's training parameters and how to change them"
    )
    _add_common_arguments(params_cmd)
    params_cmd.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["lowres", "highres"],
        help="Only show this stage's parameters",
    )
    params_cmd.set_defaults(func=cmd_params)

    # clean ---------------------------------------------------------------- #
    clean = subparsers.add_parser(
        "clean",
        help="remove a finished run's intermediate files, keeping its results",
        description=(
            "Reports what a run's intermediates are worth, and on --apply removes "
            "them. Predictions, figures, masks, logs, config and run state are never "
            "candidates. Without --deep only scratch files are removed."
        ),
    )
    _add_common_arguments(clean)
    clean.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without it the command only reports.",
    )
    clean.add_argument(
        "--deep",
        action="store_true",
        help=(
            "Also remove the rescaled slides and the multi-scale embeddings. "
            "Plotting or re-predicting afterwards means re-running the earlier stages."
        ),
    )
    clean.add_argument(
        "--all-runs",
        action="store_true",
        help="Clean every run under <project>/aurora_runs, not just --run",
    )
    clean.add_argument(
        "--force",
        action="store_true",
        help="Clean even a run that has produced no prediction yet",
    )
    clean.set_defaults(func=cmd_clean)

    # doctor --------------------------------------------------------------- #
    doctor = subparsers.add_parser("doctor", help="environment and dependency report")
    doctor.add_argument("--group", type=str, default=None, help="Only check this dependency group")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK
    try:
        return int(args.func(args))
    except AuroraError as exc:
        print(f"\nerror: {exc.format()}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
