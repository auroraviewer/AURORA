"""Translating a :class:`aurora.config.RunConfig` into concrete stage commands.

Each legacy script has its own idea of where things live.  The strategy is
always the same:

* the **run directory** is passed as the script's ``main_dir`` (``-p``),
* every path the script joins onto ``main_dir`` is passed *relative to the run
  directory* (paths outside the run directory become ``../..``-style relative
  paths, which ``os.path.join`` resolves correctly),
* so the whole pipeline writes into the tidy layout of :mod:`aurora.paths`
  without a single change to the legacy scripts themselves.

Standard library only.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

from ._legacy import code_dir, repo_root
from .config import RunConfig
from .errors import ConfigError
from .naming import missing_image_hint, resolve_image
from .paths import CHECKPOINTS_DIRNAME
from .runner import Command, legacy_stage_command, native_stage_command
from .spec import DEFAULT_COMBINED_NAME, StageContext

#: Figures must be at least this fraction of the rescaled slide, i.e.
#: ``figure_pixels = image_pixels / downscale`` with ``downscale <= 7``.  Small
#: patch sizes such as 56 and 14 collapse to one or two pixels at 1/10, which is
#: why the default is 7 rather than the historical 10.
MIN_PLOT_FRACTION_DENOMINATOR = 7

#: Sample sheets written into ``<run>/config`` (see :mod:`aurora.prepare`).
#: ``Image_name`` exactly as the user wrote it; used by everything that indexes
#: by ``Sample_name`` (the predictors, ``plot``, ``histosweep``).
SAMPLES_CANONICAL = "config/samples.csv"
#: ``Image_name`` replaced by the file name that exists on disk; used by
#: ``rescale`` and the feature extractors, because those scripts derive both the
#: slide path *and* the tissue-mask path from that column.
SAMPLES_RESOLVED = "config/samples_resolved.csv"
#: The original Visium sheet, whose *index* is the sample name.
SAMPLES_TRAINING = "config/train_samples.csv"


def relative_to_run(ctx: StageContext, path: os.PathLike | str) -> str:
    """Path as the legacy scripts need it: relative to the run directory."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    else:
        target = target.resolve()
    root = ctx.layout.root
    try:
        return Path(os.path.relpath(target, root)).as_posix()
    except ValueError:
        # different Windows drive: hand over the absolute path, os.path.join
        # discards the first component when the second is absolute.
        return target.as_posix()


def _samples_arg(ctx: StageContext, kind: str = "canonical") -> str:
    """The sample sheet to hand to a legacy script.

    ``canonical``
        ``Image_name`` exactly as the user wrote it.
    ``resolved``
        ``Image_name`` replaced by the file name that exists on disk.
    ``training``
        the original Visium sheet, whose index is the sample name.
    """
    if kind == "training":
        return SAMPLES_TRAINING
    if kind == "resolved":
        return SAMPLES_RESOLVED
    return SAMPLES_CANONICAL


def _is_training(ctx: StageContext) -> bool:
    return str(ctx.option("mode", "inference")) == "training"


def _require_slides(ctx: StageContext) -> None:
    """Pre-flight: every slide the sheet names must exist before rescaling.

    ``code/rescale.py`` builds the slide path *and* the tissue-mask path from the
    ``Image_name`` column, so an unresolvable slide produces a confusing
    OpenSlide error halfway through the run.  Checking here turns it into one
    precise message listing every path that was tried.

    Also checks the pixel size, because ``rescale.py`` reads it from a CSV field
    for anything that is not a ``.svs`` and a blank field becomes ``NaN``, which
    would silently rescale at the wrong magnification.
    """
    cfg: RunConfig = ctx.config
    image_root = cfg.resolved_image_root()
    resolutions = [resolve_image(image_root, sample) for sample in ctx.samples]
    missing = [r for r in resolutions if not r.found]
    if missing:
        raise ConfigError(
            f"{len(missing)} of {len(resolutions)} slide(s) from the sample sheet are not on disk",
            hint=missing_image_hint(image_root, resolutions),
        )
    no_pixel_size = [
        sample.sample_name
        for sample in ctx.samples
        if not sample.image_name.lower().endswith(".svs") and sample.pixel_size_um is None
    ]
    if no_pixel_size:
        shown = ", ".join(no_pixel_size[:5])
        more = f" (+{len(no_pixel_size) - 5} more)" if len(no_pixel_size) > 5 else ""
        raise ConfigError(
            f"{len(no_pixel_size)} non-.svs slide(s) have no Pixel_size_um value: {shown}{more}",
            hint=(
                "Pixel size is read from the file header only for .svs images. For any other "
                "format, add a 'Pixel_size_um' column with the micrometres-per-pixel of each "
                "image, otherwise the rescaling factor is undefined."
            ),
        )


def _uni_args(ctx: StageContext) -> List[str]:
    uni = ctx.option("uni_local_path")
    token = ctx.option("hf_token")
    if uni:
        return ["--UNI_local_path", str(uni)]
    if token:
        return ["--token", str(token)]
    return []


def _plot_arg(ctx: StageContext) -> List[str]:
    return ["--plot"] if ctx.option("plot") else []


def _common_extract_args(ctx: StageContext, training: bool) -> List[str]:
    return [
        "-p",
        ctx.layout.root.as_posix(),
        "-o",
        ctx.layout.relativize(ctx.layout.features_dir),
        "-m",
        _samples_arg(ctx, "training" if training else "resolved"),
        "-b",
        str(int(ctx.option("batch_size", 128))),
        "--num_workers",
        str(int(ctx.option("num_workers", 4))),
    ]


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #


def _segment_commands(ctx: StageContext) -> List[Command]:
    cfg: RunConfig = ctx.config
    preset = cfg.resolved_preset()
    image_root = cfg.resolved_image_root()
    save_dir = ctx.layout.segmentation_dir
    folders = sorted({sample.image_folder for sample in ctx.samples if sample.image_folder})
    sources: List[Path] = []
    if folders:
        sources.extend(image_root / folder for folder in folders)
    else:
        sources.append(image_root)

    commands: List[Command] = []
    for source in sources:
        if not source.is_dir():
            raise ConfigError(
                f"slide directory {source} does not exist",
                hint=(
                    "Check --image-path and the Image_folder column of the sample sheet "
                    "(the TCGA layout stores one folder per slide)."
                ),
            )
        argv = [
            "--source",
            str(source),
            "--save_dir",
            str(save_dir),
            "--seg",
            "--preset",
            str(preset),
        ]
        if ctx.option("plot") is False:
            pass
        commands.append(
            Command(
                argv=legacy_stage_command("create_patches_fp.py", argv, label=f"segment:{source.name}").argv,
                label=f"segment:{source.name}",
                cwd=ctx.layout.root,
            )
        )
    return commands


# --------------------------------------------------------------------------- #
# rescale
# --------------------------------------------------------------------------- #


def _rescale_commands(ctx: StageContext) -> List[Command]:
    cfg: RunConfig = ctx.config
    if ctx.option("mode") == "inference":
        # Fail before running anything with a precise list of the slides that
        # could not be found, instead of halfway through inside OpenSlide.
        _require_slides(ctx)
    argv = [
        "-p",
        ctx.layout.root.as_posix(),
        "-o",
        ctx.layout.relativize(ctx.layout.features_dir),
        "-m",
        _samples_arg(ctx, "training" if _is_training(ctx) else "resolved"),
        "--pixel_size",
        str(float(ctx.option("target_pixel_size", 0.5))),
    ]
    if ctx.option("mode") == "inference":
        argv += [
            "--inference",
            "--image_path",
            relative_to_run(ctx, cfg.resolved_image_root()),
            "--seg_path",
            ctx.layout.relativize(ctx.layout.segmentation_dir),
        ]
    else:
        argv += ["-d", str(ctx.option("processed_data_path", "processed_data"))]
    argv += _plot_arg(ctx)
    return [legacy_stage_command("rescale.py", argv, label="rescale")]


# --------------------------------------------------------------------------- #
# feature extraction
# --------------------------------------------------------------------------- #


#: Datasets ``code/image_utils.py::get_HEST1k_patch`` requires in a patch dump.
#: ``barcodes`` is what HEST's ``dump_patches`` writes; ``barcode`` is the older
#: spelling the rest of the code base used, and the reader now accepts either.
HEST_PATCH_DATASETS = ("barcodes", "img", "coords")

#: Attributes it reads off the ``img`` dataset.
HEST_PATCH_ATTRS = ("downsample", "patch_size_src", "patch_size_target", "pixel_size")


def _patch_dump_stem(image_name: str) -> str:
    """The name ``code/extract_features.py`` prefers for a patch dump.

    In training it computes ``image_name = image_file.split('.')[0]`` from the
    sheet's ``Image_name`` column, so the dump is keyed by the *image* stem.
    """
    return image_name.split(".")[0]


def _resolve_patch_dump(patches: Path, sample_name: str, image_name: str) -> Optional[Path]:
    """The dump for one sample, trying both naming conventions.

    ``code/HEST_preprocess.py`` writes ``<Sample>.h5``; ``code/extract_features.py``
    first tries ``<Image_name stem>.h5`` and then the sheet's sample name.  Nothing
    here creates a second file: the reader resolves the name itself.
    """
    stem = _patch_dump_stem(image_name)
    for candidate in (stem, sample_name):
        path = patches / f"{candidate}.h5"
        if path.is_file():
            return path
    return None


def _require_hest_patches(ctx: StageContext) -> None:
    """Check the HEST patch dumps that training feature extraction reads.

    It opens the dump and indexes ``barcodes``/``img``/``coords`` plus four
    attributes on ``img``.  A mismatch there used to be a bare
    ``KeyError: object 'barcode' doesn't exist`` from h5py; checking here names the
    file, the datasets it does contain and the directory it sits in.
    """
    if ctx.option("dry_run"):
        return
    try:
        import h5py
    except ImportError:  # the dependency check reports h5py itself
        return

    processed = ctx.layout.root / str(ctx.option("processed_data_path", "processed_data"))
    patches = processed / "patches"
    samples = ctx.samples.samples
    if not samples:
        return

    problems: List[str] = []
    warnings: List[str] = []
    if not patches.is_dir():
        problems.append(f"  {patches} does not exist")
    else:
        present = sorted(p.name for p in patches.glob("*.h5"))
        for sample in samples:
            path = _resolve_patch_dump(patches, sample.sample_name, sample.image_name)
            if path is None:
                problems.append(
                    f"  missing: {sample.sample_name}.h5 "
                    f"(or {_patch_dump_stem(sample.image_name)}.h5)"
                )
                continue
            try:
                with h5py.File(path, "r") as handle:
                    keys = sorted(handle.keys())
                    absent = [
                        key
                        for key in HEST_PATCH_DATASETS
                        if key not in keys and not (key == "barcodes" and "barcode" in keys)
                    ]
                    attrs = sorted(handle["img"].attrs.keys()) if "img" in keys else []
                    file_attrs = sorted(handle.attrs.keys())
                    missing_attrs = [a for a in HEST_PATCH_ATTRS if a not in attrs]
            except OSError as exc:
                problems.append(f"  {path.name}: cannot be read ({exc})")
                continue
            if absent:
                problems.append(
                    f"  {path.name}: has no {', '.join(absent)} "
                    f"(datasets present: {', '.join(keys) or 'none'})"
                )
            elif missing_attrs:
                # Not fatal: get_HEST1k_patch falls back to file attrs, the
                # metadata/<name>.json sidecar, then derived values.  Say so once.
                where = f"file attrs: {', '.join(file_attrs)}" if file_attrs else "no file attrs"
                warnings.append(
                    f"  {path.name}: 'img' has no {', '.join(missing_attrs)} "
                    f"({where}); the reader will derive or default them"
                )
        if problems:
            listing = ", ".join(present[:12]) + (" …" if len(present) > 12 else "")
            problems.append(f"  files actually in {patches}: {listing or 'none'}")

    if not problems:
        for warning in warnings[:2]:
            print(f"    note:{warning}")
        if len(warnings) > 2:
            print(f"    note: ({len(warnings) - 2} more dumps without patch metadata)")
        return
    raise ConfigError(
        "the HEST patch dumps that training feature extraction reads are not usable:\n"
        + "\n".join(problems),
        hint=(
            f"These come from the preprocess stage, which writes <Sample>.h5 with "
            f"{', '.join(HEST_PATCH_DATASETS)} plus {', '.join(HEST_PATCH_ATTRS)} on "
            "'img'. The <Sample> / <Image_name stem> difference is handled by the "
            "reader, so a 'missing' entry means preprocess produced no dump for that "
            "sample.\n"
            "Check that:\n"
            "  - the preprocess stage completed for every sample in the sheet (drop "
            "any sample whose slide failed);\n"
            "  - the directory is not left over from an earlier attempt with a "
            "different preprocessor.\n"
            f"Re-run it with:\n  python -m aurora run --mode training --run {ctx.config.run_name} "
            f"--stage preprocess --force"
        ),
    )


def _features_lowres_commands(ctx: StageContext) -> List[Command]:
    cfg: RunConfig = ctx.config
    if _is_training(ctx):
        _require_hest_patches(ctx)
    argv = _common_extract_args(ctx, _is_training(ctx))
    argv += ["--patch_sizes", *[str(int(p)) for p in ctx.lowres_patch_sizes]]
    if ctx.option("mode") == "inference":
        argv += [
            "--inference",
            "--image_path",
            relative_to_run(ctx, cfg.resolved_image_root()),
            "--seg_path",
            ctx.layout.relativize(ctx.layout.segmentation_dir),
        ]
    else:
        argv += ["-d", str(ctx.option("processed_data_path", "processed_data"))]
    argv += _uni_args(ctx)
    argv += _plot_arg(ctx)
    return [legacy_stage_command("extract_features.py", argv, label="features_lowres")]


def _features_highres_commands(ctx: StageContext) -> List[Command]:
    uni = ctx.option("uni_local_path")
    if not uni:
        raise ConfigError(
            "high-resolution feature extraction requires a local UNI2-h checkpoint",
            hint=(
                "The high-resolution extractor loads UNI2-h from a local folder and refuses to "
                "download it. Clone https://huggingface.co/MahmoodLab/UNI2-h and pass\n"
                "  --uni-local-path /path/to/UNI2-h\n"
                "(In the original command.sh this was --UNI_local_path ./uni2h.)"
            ),
        )
    argv = _common_extract_args(ctx, _is_training(ctx))
    argv += ["--patch_sizes", *[str(int(p)) for p in ctx.highres_patch_sizes]]
    argv += [
        "-c",
        str(code_dir()),
        "--sample_name_col",
        "1",
        "--stride",
        str(int(ctx.option("stride", 112))),
        "--UNI_local_path",
        str(uni),
        "--compile-mode",
        str(ctx.option("compile_mode", "off")),
    ]
    if ctx.option("mode") == "inference":
        argv += ["--inference"]
    argv += _plot_arg(ctx)
    return [legacy_stage_command("extract_features_highres.py", argv, label="features_highres")]


# --------------------------------------------------------------------------- #
# native predictors
# --------------------------------------------------------------------------- #


def _predict_lowres_arguments(ctx: StageContext) -> List[str]:
    cfg: RunConfig = ctx.config
    argv = [
        "--run-root",
        ctx.layout.root.as_posix(),
        "--model",
        str(cfg.resolved_model()),
        "--samples",
        _samples_arg(ctx, "canonical"),
        "--pred-dir",
        ctx.layout.relativize(ctx.layout.prediction_dir("lowres")),
        "--interim-dir",
        ctx.layout.relativize(ctx.layout.features_dir),
        "--device",
        str(ctx.option("device", "cpu")),
        "--batch-size",
        str(int(ctx.option("prediction_batch_size", 100))),
        "--output-format",
        str(ctx.option("output_format", "modern")),
    ]
    # Both spellings are always passed explicitly: the defaults are on now, so an
    # omitted flag would silently win over the user's --no-... choice.
    argv.append("--uint8-output" if ctx.option("uint8_output") else "--no-uint8-output")
    argv.append("--only-224" if ctx.option("only_224") else "--no-only-224")
    return argv


def _predict_highres_arguments(ctx: StageContext) -> List[str]:
    cfg: RunConfig = ctx.config
    argv = [
        "--run-root",
        ctx.layout.root.as_posix(),
        "--model",
        str(cfg.resolved_model()),
        "--samples",
        _samples_arg(ctx, "canonical"),
        "--pred-dir",
        ctx.layout.relativize(ctx.layout.prediction_dir("highres")),
        "--interim-dir",
        ctx.layout.relativize(ctx.layout.features_dir),
        "--lowres-dir",
        ctx.layout.relativize(ctx.layout.prediction_dir("lowres")),
        "--lowres-state",
        ctx.layout.relativize(ctx.layout.run_json),
        "--rescaled-dir",
        ctx.layout.relativize(ctx.layout.rescaled_dir),
        "--histosweep-patch-size",
        str(int(ctx.option("histosweep_patch_size", 14))),
        "--device",
        str(ctx.option("device", "cpu")),
        "--batch-size",
        str(int(ctx.option("prediction_batch_size", 100))),
        "--h5-layout",
        str(ctx.option("h5_layout", "spatial_gene_delta")),
        "--h5-compression",
        str(ctx.option("h5_compression", "zstd")),
        "--h5-compression-level",
        str(int(ctx.option("h5_compression_level", 9))),
    ]
    argv.append("--uint8-output" if ctx.option("uint8_output") else "--no-uint8-output")
    if ctx.option("compress_output"):
        argv.append("--compress-output")
    return argv


# --------------------------------------------------------------------------- #
# combine checkpoints (native, training)
# --------------------------------------------------------------------------- #


def _combine_commands(ctx: StageContext) -> List[Command]:
    cfg: RunConfig = ctx.config
    checkpoints = _train_path(ctx)
    # The merge tool runs with the run directory as its working directory, so a path the
    # user typed relative to their shell (``--gene-list examples/BRCA_gene.csv``) has to
    # be re-expressed relative to the run, exactly like the other stages' inputs.
    gene = (
        relative_to_run(ctx, cfg.gene_list_path)
        if cfg.gene_list_path
        else str(checkpoints / "gene_list.csv")
    )
    cells = (
        relative_to_run(ctx, cfg.cell_type_path)
        if cfg.cell_type_path
        else str(checkpoints / "cell_types.csv")
    )
    norm = (
        relative_to_run(ctx, cfg.norm_factor_path)
        if cfg.norm_factor_path
        else str(checkpoints / "gene_normalize_factor.csv")
    )
    lowres = checkpoints / _lowres_weights_name(ctx)
    highres = checkpoints / "AURORA_highres_model_weights.pth"
    out = checkpoints / str(ctx.option("combined_name", DEFAULT_COMBINED_NAME))
    argv = [
        "--lowres",
        str(lowres),
        "--highres",
        str(highres),
        "--out",
        str(out),
        "--force",
        "--gene-list",
        str(gene),
        "--cell-types",
        str(cells),
        "--norm-factors",
        str(norm),
        "--lowres-params",
        str(ctx.layout.config_dir / "params_lowres.json"),
        "--highres-params",
        str(checkpoints / "params_highres.json"),
    ]
    model_name = str(cfg.extra.get("model_name", "") or "")
    if model_name:
        argv += ["--model-name", model_name]
    cancer_type = str(cfg.extra.get("cancer_type", "") or "")
    if cancer_type:
        argv += ["--cancer-type", cancer_type]
    return [
        Command(
            argv=[_python(), str(repo_root() / "tools" / "merge_aurora_checkpoints.py"), *argv],
            label="combine_checkpoints",
            cwd=ctx.layout.root,
        )
    ]


def _python() -> str:
    from ._legacy import python_executable

    return python_executable()


# --------------------------------------------------------------------------- #
# training stages
# --------------------------------------------------------------------------- #


def _train_path(ctx: StageContext) -> Path:
    """Training output directory (`-t`): weights, assets and caches."""
    return ctx.layout.root / str(ctx.option("train_dir", CHECKPOINTS_DIRNAME))


def _training_common(ctx: StageContext) -> List[str]:
    return [
        "-p",
        ctx.layout.root.as_posix(),
        "-i",
        ctx.layout.relativize(ctx.layout.features_dir),
        "-t",
        _train_path(ctx).name,
        "-d",
        str(ctx.option("processed_data_path", "processed_data")),
    ]


def _prepare_st_commands(ctx: StageContext) -> List[Command]:
    cfg: RunConfig = ctx.config
    argv = [
        "-p",
        ctx.layout.root.as_posix(),
        "-d",
        str(ctx.option("processed_data_path", "processed_data")),
        "-m",
        _samples_arg(ctx, "training"),
        "-o",
        ctx.layout.relativize(ctx.layout.features_dir),
    ]
    # prepare_st.py opens f"{project_path}/{genetype_file}", so this must be relative
    # to the run root. It selects the protein-coding genes to rank; without it the
    # script looks for <run>/TCGA_genetype.csv and fails. `relative_to_run` is used
    # rather than `layout.relativize` because the table normally lives outside the
    # run (examples/), and relativize returns an absolute path in that case -- which
    # the script would then append to the run root a second time.
    genetype = cfg.resolved_genetype()
    if genetype is None:
        raise ConfigError(
            "prepare_st needs a gene-type table and none was found",
            hint=(
                "It should be a CSV with a 'gene_type' column, such as the "
                "examples/TCGA_genetype.csv shipped with the repository.\n"
                "Pass one explicitly with --genetype-file /path/to/TCGA_genetype.csv."
            ),
        )
    argv += ["--genetype_file", relative_to_run(ctx, genetype)]
    return [legacy_stage_command("prepare_st.py", argv, label="prepare_st")]


def _deconvolution_commands(ctx: StageContext) -> List[Command]:
    cfg: RunConfig = ctx.config
    argv = [
        "-p",
        ctx.layout.root.as_posix(),
        "-d",
        str(ctx.option("processed_data_path", "processed_data")),
        "-m",
        _samples_arg(ctx, "training"),
        "-o",
        ctx.layout.relativize(ctx.layout.features_dir),
        "-r",
        str(cfg.extra.get("scrna_reference", "reference.rds")),
        "--python_path",
        _python(),
    ]
    return [
        Command(
            argv=["Rscript", str(code_dir() / "get_deconvolution.R"), *argv],
            label="deconvolution",
            cwd=ctx.layout.root,
        )
    ]


def _smooth_cnts_commands(ctx: StageContext) -> List[Command]:
    argv = [
        "-p",
        ctx.layout.root.as_posix(),
        "-d",
        str(ctx.option("processed_data_path", "processed_data")),
        "-m",
        _samples_arg(ctx, "training"),
        "-o",
        ctx.layout.relativize(ctx.layout.features_dir),
    ]
    return [legacy_stage_command("smooth_cnt.py", argv, label="smooth_cnts")]


def _train_lowres_commands(ctx: StageContext) -> List[Command]:
    argv = _training_common(ctx)
    argv += [
        "-j",
        "config/params_lowres.json",
        "--train_samples",
        _samples_arg(ctx, "training"),
        "--train_sample_col",
        str(int(ctx.option("train_sample_col", 0))),
        "--log",
    ]
    if ctx.option("smooth_counts"):
        argv.append("--smooth_cnts")
    argv += _plot_arg(ctx)
    return [legacy_stage_command("train.py", argv, label="train_lowres")]


def _lowres_weights_name(ctx: StageContext) -> str:
    """Name of the low-resolution state dict produced by ``train.py``.

    ``train.py`` writes ``AURORA_model_weights_smooth.pth`` with ``--smooth_cnts``
    and ``AURORA_model_weights.pth`` without it, so every consumer has to agree.
    """
    return (
        "AURORA_model_weights_smooth.pth"
        if ctx.option("smooth_counts")
        else "AURORA_model_weights.pth"
    )


def _prepare_highres_training_commands(ctx: StageContext) -> List[Command]:
    checkpoints = ctx.layout.relativize(_train_path(ctx))
    argv = [
        "-p",
        ctx.layout.root.as_posix(),
        "-t",
        checkpoints,
        "-d",
        str(ctx.option("processed_data_path", "processed_data")),
        "-j",
        # This stage builds the low-resolution model and loads the low-res weights,
        # so its parameters must describe that architecture (see prepare.py).
        "params_highres_training.json",
        "--model_pth",
        _lowres_weights_name(ctx),
        "--sample_sheet",
        _samples_arg(ctx, "training"),
    ]
    return [
        legacy_stage_command("prepare_highres_training.py", argv, label="prepare_highres_training")
    ]


def _train_highres_commands(ctx: StageContext) -> List[Command]:
    argv = _training_common(ctx)
    argv += [
        "-j",
        "params_highres.json",
        "--train_samples",
        _samples_arg(ctx, "training"),
        "--train_sample_col",
        str(int(ctx.option("train_sample_col", 0))),
        "--log",
    ]
    argv += _plot_arg(ctx)
    return [legacy_stage_command("train_highres.py", argv, label="train_highres")]


def _finetune_commands(ctx: StageContext) -> List[Command]:
    argv = _training_common(ctx)
    argv += [
        "-j",
        "config/params_lowres.json",
        "--train_samples",
        _samples_arg(ctx, "training"),
        "--train_sample_col",
        str(int(ctx.option("train_sample_col", 0))),
        "--model_pth",
        f"{_train_path(ctx).name}/AURORA_model_weights.pth",
        "--log",
    ]
    gene_list = ctx.option("finetune_gene_list")
    if gene_list:
        argv += ["--finetune_gene_list", str(gene_list)]
    num_epochs = ctx.option("num_epochs")
    if num_epochs:
        argv += ["--num_epochs", str(int(num_epochs))]
    argv += _plot_arg(ctx)
    return [legacy_stage_command("finetune.py", argv, label="finetune")]


def _project_relative(target: Path, project: Path) -> str:
    """``target`` as a path relative to ``project``.

    ``code/HEST_preprocess.py`` builds every path by f-string concatenation --
    ``f"{args.project_path}/{args.raw_data_path}"`` and the same for ``-m`` and
    ``-o`` -- so each of those must be *relative to* ``-p``.  An absolute value
    produces the path twice over
    (``<run>/abs/path/raw_data``), and so does ``os.path.join``'s
    absolute-wins behaviour, which is why the option values are made relative
    here rather than the script changed.
    """
    try:
        relative = os.path.relpath(target, project)
    except ValueError as exc:  # Windows, target on another drive
        raise ConfigError(
            f"{target} cannot be reached by a path relative to {project}",
            hint=(
                "The preprocessing script concatenates its paths onto --project, so the "
                "raw data and output must live on the same drive. Move the project, or "
                "the data, so they share one."
            ),
        ) from exc
    return Path(relative).as_posix()


def _preprocess_commands(ctx: StageContext) -> List[Command]:
    cfg: RunConfig = ctx.config
    project = cfg.resolved_project()
    # `HEST_preprocess.py` concatenates *everything* onto -p:
    #   data_dir  = f"{project_path}/{raw_data_path}"
    #   metadata  = f"{project_path}/{metadata_path}"
    #   outdir    = f"{project_path}/{output_path}"
    # So -p is the project, and -m/-o are made project-relative paths that still
    # land inside the run -- keeping the run self-contained without patching the
    # legacy script.
    raw_data = cfg.resolved_raw_data()
    if not raw_data.is_dir():
        candidates = "\n".join(f"  - {c}" for c in cfg.raw_data_candidates())
        if ctx.option("dry_run"):
            print(f"    note: raw Visium data is not at {raw_data} (would fail at run time)")
        else:
            raise ConfigError(
                f"raw Visium data directory {raw_data} does not exist",
                hint=(
                    "Point --raw-data at the folder that holds one sub-directory per sample "
                    "(each with the H&E .tif, filtered_feature_bc_matrix.h5 and spatial/).\n"
                    "It is resolved against the project directory, so with "
                    f"--project {project} the default looks under {project / cfg.raw_data_path}."
                    f"\nSearched:\n{candidates}"
                ),
            )
    # HEST refuses to segment without its own checkpoint catalogue, and its error
    # prints the catalogue path where the checkpoint path belongs.  Check first so
    # the message names the command that fixes it.
    from .hest import check_setup, require_setup

    if ctx.option("dry_run"):
        ok, detail = check_setup()
        if not ok:
            print(f"    note: HEST is not ready ({detail.splitlines()[0]})")
            print("    note: run 'python -m aurora hest-setup --check' for the fix")
    else:
        require_setup()

    processed = layout_processed_dir(ctx)
    sheet = layout_training_sheet(ctx)
    argv = [
        "-p",
        project.as_posix(),
        "-r",
        _project_relative(raw_data, project),
        "-m",
        _project_relative(sheet, project),
        "-o",
        _project_relative(processed, project),
    ]
    return [legacy_stage_command("HEST_preprocess.py", argv, label="preprocess")]


def layout_processed_dir(ctx: StageContext) -> Path:
    """Where ``preprocess`` writes: ``<run>/processed_data``."""
    return ctx.layout.root / str(ctx.option("processed_data_path", "processed_data"))


def layout_training_sheet(ctx: StageContext) -> Path:
    """The training sample sheet the preprocess stage reads."""
    from .prepare import TRAIN_SAMPLES_FILENAME

    return ctx.layout.config_dir / TRAIN_SAMPLES_FILENAME


# --------------------------------------------------------------------------- #
# optional stages
# --------------------------------------------------------------------------- #


def _histosweep_commands(ctx: StageContext) -> List[Command]:
    # -i is the interim directory that holds UNI_multiscale_patches/wsis_rescaled;
    # -t only receives HistoSweep's native outputs, which it deletes on success.
    # The mask it leaves in wsis_rescaled/ is what predict_highres applies.
    from .naming import HISTOSWEEP_MASK_SUFFIX

    argv = [
        "-p",
        ctx.layout.root.as_posix(),
        "-i",
        ctx.layout.relativize(ctx.layout.features_dir),
        "-t",
        ctx.layout.relativize(ctx.layout.prediction_dir("lowres")),
        "--sample_sheet",
        _samples_arg(ctx, "canonical"),
        # Tie the mask name to the one the high-resolution stage looks for.
        "--mask-suffix",
        HISTOSWEEP_MASK_SUFFIX,
    ]
    histosweep = ctx.option("histosweep_path")
    if histosweep:
        argv += ["--histosweep_path", str(histosweep)]
    return [legacy_stage_command("run_HistoSweep.py", argv, label="histosweep")]


def _plot_commands(ctx: StageContext) -> List[Command]:
    genes = list(ctx.option("gene_names") or [])
    cell_types = list(ctx.option("cell_types") or [])
    if not genes and not cell_types:
        # `code/plot.py` raises this itself, but only after it has already failed
        # to read its arguments, so the stage checks first and says what to pass.
        raise ConfigError(
            "the plot stage needs at least one --gene or --celltype",
            hint="For example: --gene IL7R KRT8, or --celltype Malignant T_cell.",
        )
    resolution = int(ctx.option("plot_resolution", 224))
    kind = str(ctx.option("plot_predictions", "lowres"))
    # The figure is exactly 1/downscale of the rescaled slide, so the "no smaller
    # than 1/7" requirement is a bound on the divisor -- and a patch must still
    # occupy at least one figure pixel, or the drawing loop writes nothing.
    downscale = int(ctx.option("plot_downscale", MIN_PLOT_FRACTION_DENOMINATOR))
    if downscale < 1:
        raise ConfigError(
            f"--plot-downscale must be >= 1, got {downscale}",
            hint="1 gives a full-size figure; the default is 7.",
        )
    if downscale > MIN_PLOT_FRACTION_DENOMINATOR:
        raise ConfigError(
            f"--plot-downscale {downscale} would make the figure 1/{downscale} of the "
            f"rescaled slide, smaller than the required 1/{MIN_PLOT_FRACTION_DENOMINATOR}",
            hint=(
                f"Use --plot-downscale {MIN_PLOT_FRACTION_DENOMINATOR} (the default) or less. "
                "Smaller values give larger figures."
            ),
        )
    patch_pixels = resolution // downscale
    if patch_pixels < 1:
        raise ConfigError(
            f"a {resolution} px patch would be {patch_pixels} figure pixels at "
            f"--plot-downscale {downscale}",
            hint=(
                f"Lower --plot-downscale below {resolution} so each patch covers at least "
                "one pixel."
            ),
        )

    # code/plot.py reads the legacy 3-D array layout, while predict_* writes the
    # modern dict layout by default.  Build a derived view instead of asking the
    # user to re-run the prediction with --output-format legacy.  The same view
    # mechanism serves the high-resolution predictions, whose file is named
    # '<sample>-highres-pred.pickle' but which plot.py opens as
    # '<sample>-pred.pickle'.
    from .plotting import build_view, validate_kind

    kind = validate_kind(kind)
    prediction_root, notes = build_view(
        ctx.layout,
        ctx.layout.prediction_dir(kind),
        ctx.sample_names,
        resolution=resolution,
        force=bool(ctx.option("force_plots", False)),
        dry_run=bool(ctx.option("dry_run", False)),
        kind=kind,
    )
    for note in notes:
        print(f"    {note}")

    # Keep the two kinds apart on disk as well as in tmp/.
    plot_dir = ctx.layout.plots_dir if kind == "lowres" else ctx.layout.plots_dir / "highres"

    argv = [
        "-p",
        ctx.layout.root.as_posix(),
        "-i",
        ctx.layout.relativize(ctx.layout.features_dir),
        "-t",
        ctx.layout.relativize(prediction_root),
        "--plot_dir",
        ctx.layout.relativize(plot_dir),
        "--resolution",
        str(resolution),
        "--downscale",
        str(downscale),
        "--samples",
        *ctx.sample_names,
    ]
    # Only pass a flag that has values: `--genes` with nothing after it would let
    # argparse consume the *next* option as its value.
    if genes:
        argv += ["--genes", *genes]
    if cell_types:
        argv += ["--celltypes", *cell_types]
    return [legacy_stage_command("plot.py", argv, label="plot")]


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

COMMAND_BUILDERS = {
    "segment": _segment_commands,
    "rescale": _rescale_commands,
    "features_lowres": _features_lowres_commands,
    "features_highres": _features_highres_commands,
    "predict_lowres": lambda ctx: [
        native_stage_command("aurora.inference.predict_lowres", _predict_lowres_arguments(ctx))
    ],
    "predict_highres": lambda ctx: [
        native_stage_command("aurora.inference.predict_highres", _predict_highres_arguments(ctx))
    ],
    "histosweep": _histosweep_commands,
    "plot": _plot_commands,
    "preprocess": _preprocess_commands,
    "prepare_st": _prepare_st_commands,
    "deconvolution": _deconvolution_commands,
    "smooth_cnts": _smooth_cnts_commands,
    "train_lowres": _train_lowres_commands,
    "finetune": _finetune_commands,
    "prepare_highres_training": _prepare_highres_training_commands,
    "train_highres": _train_highres_commands,
    "combine_checkpoints": _combine_commands,
}


def build_commands(stage: str, ctx: StageContext) -> List[Command]:
    """The commands that execute ``stage`` for this run."""
    try:
        builder = COMMAND_BUILDERS[stage]
    except KeyError as exc:
        raise ConfigError(
            f"no command builder is registered for stage {stage!r}",
            hint="This is a bug in aurora/commands.py; please report it.",
        ) from exc
    return list(builder(ctx))
