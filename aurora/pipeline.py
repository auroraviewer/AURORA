"""The pipeline executor: gate, run, verify, record.

Responsibilities
----------------
* resolve the run (combined checkpoint + sample sheet + layout),
* materialise the inputs the legacy scripts expect,
* refuse to start a stage whose prerequisites are not complete (requirement #2),
* execute each stage's commands, capture their logs, and
* only mark a stage complete when its declared artefacts exist on disk.

Standard library only (the heavy work happens in the stage subprocesses).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .bundle import CombinedCheckpoint
from .checkpoint import CheckpointAssets, load_assets
from .commands import build_commands
from .config import RunConfig, select_samples
from .deps import DEPENDENCIES, resolve_import_name
from .errors import (
    CheckpointFormatError,
    ConfigError,
    MissingDependencyError,
    StageFailedError,
    StagePrerequisiteError,
)
from .paths import RunLayout, write_layout_stamp
from .prepare import load_run_params, prepare_run
from .runner import build_stage_environment, run_command
from .samples import SampleSheet, read_sample_sheet
from .spec import (
    INFERENCE_PIPELINE,
    MODE_TRAINING,
    STAGE_ORDER,
    TRAINING_PIPELINE,
    StageContext,
    check_outputs_exist,
    get_stage,
    run_post_hook,
    stage_packages,
)
from .state import (
    RunState,
    evaluate_gate,
    format_stage_summary,
)

#: Outcome values of :class:`StageOutcome`.
OUTCOME_COMPLETE = "complete"
OUTCOME_REUSED = "reused"
OUTCOME_FAILED = "failed"
OUTCOME_PLANNED = "planned"
OUTCOME_SKIPPED = "skipped"


@dataclass
class StageOutcome:
    """What happened to one stage."""

    stage: str
    outcome: str
    message: str = ""
    duration_s: float = 0.0
    log_path: Optional[Path] = None
    commands: List[str] = field(default_factory=list)

    def describe(self) -> str:
        text = f"{self.stage:<24} {self.outcome:<9}"
        if self.duration_s:
            text += f" {self.duration_s:7.1f}s"
        if self.message:
            text += f"  {self.message}"
        return text


@dataclass
class PipelineResult:
    """Result of running a list of stages."""

    outcomes: List[StageOutcome] = field(default_factory=list)
    run_root: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return not any(o.outcome == OUTCOME_FAILED for o in self.outcomes)

    @property
    def failed(self) -> List[StageOutcome]:
        return [o for o in self.outcomes if o.outcome == OUTCOME_FAILED]

    def describe(self) -> str:
        if not self.outcomes:
            return "(no stages were requested)"
        return "\n".join("  " + outcome.describe() for outcome in self.outcomes)


def build_context(
    layout: RunLayout,
    sheet: SampleSheet,
    config: RunConfig,
    checkpoint: Optional[CombinedCheckpoint] = None,
) -> StageContext:
    """The stage context a run executes with.

    Parameters come from the files preparation wrote -- from the checkpoint, from
    ``aurora/defaults.py``, or from ``--param`` -- so a run without a checkpoint (the
    ``--gene-list``/``--cell-types``/``--norm-factors`` route) still knows its patch
    sizes.  Anything the checkpoint manifest records and the files do not is filled in
    afterwards.
    """
    lowres_params, highres_params = load_run_params(layout, config)
    context = StageContext(
        layout=layout,
        samples=sheet,
        lowres_params=lowres_params,
        highres_params=highres_params,
        options=config.stage_options(),
        config=config,
    )
    if checkpoint is not None:
        for stage, params in (
            ("lowres", context.lowres_params),
            ("highres", context.highres_params),
        ):
            if stage not in checkpoint.manifest.stages:
                continue
            entry = checkpoint.manifest.stages[stage]
            if entry.patch_sizes and not params.get("patch_size"):
                params["patch_size"] = list(entry.patch_sizes)
            if entry.img_emb_dim is not None and not params.get("img_emb_dim"):
                params["img_emb_dim"] = int(entry.img_emb_dim)
    return context


class Pipeline:
    """The run context plus the stage executor."""

    def __init__(
        self,
        config: RunConfig,
        layout: RunLayout,
        state: RunState,
        context: StageContext,
        checkpoint: CombinedCheckpoint,
        assets: CheckpointAssets,
    ) -> None:
        self.config = config
        self.layout = layout
        self.state = state
        self.context = context
        self.checkpoint = checkpoint
        self.assets = assets

    # ------------------------------------------------------------- factories
    @classmethod
    def create(cls, config: RunConfig, require_model: bool = True) -> "Pipeline":
        """Resolve the run described by ``config``.

        ``require_model`` is ``False`` for a stage set that never reads the
        checkpoint (``preprocess`` today), so such a run does not have to name a
        ``.pth``.  Everything that does read it -- parameters, bundled assets,
        weights -- keeps the requirement.
        """
        project = config.resolved_project()
        layout = RunLayout.create(project, config.run_name, config.run_root)

        checkpoint = None
        assets = None
        model_path = None
        if config.model_path is None and not require_model:
            pass  # nothing downstream of this run needs a checkpoint
        else:
            if config.model_path is None:
                config.resolved_model()  # raises the friendly error
            model_path = config.resolved_model()
            checkpoint = CombinedCheckpoint.open(model_path)
            # Fail before any stage runs if a payload could never be loaded.  This is
            # a cheap zip listing, needs no PyTorch, and turns a late
            # "file in archive is not in a subdirectory" into an actionable error.
            broken: List[str] = []
            for stage in checkpoint.stages():
                ok, _root, problem = checkpoint.payload_layout(stage)
                if not ok:
                    broken.append(f"  - {stage}: {problem}")
            if broken:
                raise CheckpointFormatError(
                    f"{model_path} contains payload(s) that torch.load cannot read:\n"
                    + "\n".join(broken),
                    hint=checkpoint.rebuild_hint(),
                )

        samples_path = config.resolved_samples()
        sheet = read_sample_sheet(samples_path, kind="inference")
        sheet = select_samples(sheet, config.samples_subset)

        layout.ensure(config.mode)
        write_layout_stamp(layout)

        state = RunState.load(layout, create=True)
        state.set_mode(config.mode)
        if checkpoint is not None:
            state.set_model(
                path=model_path.as_posix(),
                sha256=None,
                model_name=checkpoint.model_id,
                cancer_type=checkpoint.cancer_type,
                stages=checkpoint.stages(),
                requires={k: list(v) for k, v in checkpoint.manifest.requires.items()},
                format=checkpoint.manifest.format,
                format_version=checkpoint.manifest.format_version,
            )
        state.set_samples(
            path=samples_path.as_posix(),
            count=len(sheet),
            names=sheet.sample_names,
            fingerprint=sheet.fingerprint(),
        )
        state.save()

        if checkpoint is not None:
            assets = load_assets(
                checkpoint,
                search_dirs=[project, model_path.parent, layout.config_dir, layout.root],
            )
        # Without a checkpoint there are no bundled assets to materialise; the
        # stage that got us here does not read them.
        report = prepare_run(
            layout, config, sheet, checkpoint, assets, require_checkpoint=checkpoint is not None
        )
        state.data["inputs"] = {
            "samples": report.get("samples", {}),
            "bulk": report.get("bulk"),
            "checkpoint_assets": report.get("checkpoint_assets", {}),
        }
        for note in report.get("notes", []):
            print(f"note: {note}")
        state.save()

        context = build_context(layout, sheet, config, checkpoint)
        return cls(config, layout, state, context, checkpoint, assets)

    # ------------------------------------------------------------ dependency
    def check_stage_dependencies(self, stage: str) -> None:
        """Raise when a package the stage needs is missing."""
        missing: List[str] = []
        for import_name in stage_packages(stage):
            if resolve_import_name_by_name(import_name) is None:
                missing.append(import_name)
        if not missing:
            return
        from .deps import (
            _dependency_for,
            group_of,
            install_hint,
            requirements_file_for,
        )

        groups = sorted({group_of(name) for name in missing})
        hints = []
        source_only = []
        for import_name in missing:
            group, hint = install_hint(import_name)
            first_line = hint.splitlines()[0] if hint else ""
            hints.append(f"  - {import_name:<16} [{group}] {first_line}")
            dep = _dependency_for(import_name)
            if dep is not None and not dep.installable:
                source_only.append(dep)
        # A requirements file cannot install a source-only dependency, so say what
        # actually works rather than pointing at a command that will not help.
        installable_groups = [
            g
            for g in groups
            if any(
                d.group == g and d.installable for d in (_dependency_for(n) for n in missing)
                if d is not None
            )
        ]
        requirement_files = [requirements_file_for(g) for g in installable_groups]
        install_lines = [
            f"  pip install -r {req}" for req in requirement_files if req
        ]
        for dep in source_only:
            install_lines.append(f"  {dep.source_command}   # {dep.distribution}")
            install_lines.append(
                f"     ('{dep.distribution}' has no usable PyPI release, so "
                f"'pip install .[{dep.group}]' cannot provide it)"
            )
        raise MissingDependencyError(
            f"stage '{stage}' needs {len(missing)} package(s) that are not installed: "
            + ", ".join(missing),
            hint=(
                "Missing packages:\n"
                + "\n".join(hints)
                + "\n\nInstall them with:\n"
                + ("\n".join(install_lines) if install_lines else "  (see the README)")
                + "\n\nRun 'python -m aurora doctor' for a full report, or pass --dry-run to "
                "print the commands without needing the packages."
            ),
        )

    # ------------------------------------------------------------------ run
    def run(
        self,
        stages: Sequence[str],
        auto_prerequisites: Optional[bool] = None,
        force: bool = False,
    ) -> PipelineResult:
        """Execute ``stages`` in dependency order."""
        auto = self.config.auto_prerequisites if auto_prerequisites is None else auto_prerequisites
        result = PipelineResult(run_root=self.layout.root)
        print(f"Run directory: {self.layout.root}")
        if self.checkpoint is None:
            if self.config.mode == "training":
                print(
                    "Configuration: none -- parameters from aurora/defaults.py, assets "
                    "from --gene-list/--cell-types/--norm-factors"
                )
            else:
                print("Checkpoint: none needed by this stage set")
        elif self.config.mode == "training":
            # A training run starts from scratch: no stage loads these weights, so
            # printing the full checkpoint banner reads as "resuming from a model".
            manifest = self.checkpoint.manifest
            print(f"Configuration: {manifest.model_name or 'combined checkpoint'}")
            print(
                "  supplies the model parameters and the gene/cell-type/normalisation "
                "assets."
            )
            print(
                "  Training starts from scratch; these weights are not loaded. "
                "The result is written to checkpoints/aurora.pth."
            )
        else:
            print(self.checkpoint.describe())
        print()

        requested = list(stages)
        missing = self._collect_missing_prerequisites(requested)
        if missing:
            if auto:
                print(
                    "Adding missing prerequisite stage(s) automatically: "
                    + ", ".join(missing)
                )
                requested = missing + requested
            # Without --auto-prerequisites the gate below reports them.

        ordered = _order_stages(requested)
        for stage in ordered:
            outcome = self._run_one(stage, force=force)
            result.outcomes.append(outcome)
            if outcome.outcome == OUTCOME_FAILED and not self.config.keep_going:
                print(f"\nStopping after the failure of '{stage}'.")
                break
        return result

    def _collect_missing_prerequisites(self, stages: Sequence[str]) -> List[str]:
        missing: List[str] = []
        for stage in stages:
            for name in evaluate_gate(self.state, self.context, stage).missing:
                if name not in missing:
                    missing.append(name)
        return [n for n in STAGE_ORDER if n in missing]

    def _artefacts_for(self, spec: Any) -> List[str]:
        """The files a just-finished stage wrote, for stages with a fingerprint."""
        found: List[str] = []
        for resolved in spec.outputs(self.context):
            path = Path(resolved)
            if path.is_dir():
                found.extend(sorted(p.as_posix() for p in path.rglob("*") if p.is_file()))
            elif path.is_file():
                found.append(path.as_posix())
        return found

    def _remove_previous_artefacts(self, stage: str) -> int:
        """Delete the files the last run of ``stage`` wrote, before re-running it.

        Only used when a stage is re-run because its arguments changed, and only for
        the exact files it recorded.  Otherwise asking for a new gene list would leave
        the old figures next to the new ones, which reads as "the change did nothing".
        """
        record = self.state.record(stage)
        files = list(((record.extra if record else {}) or {}).get("artefacts") or [])
        removed = 0
        for name in files:
            path = Path(str(name))
            try:
                if path.is_file():
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        return removed

    def _run_one(self, stage: str, force: bool = False) -> StageOutcome:
        spec = get_stage(stage)
        outputs = spec.outputs(self.context)
        present, missing_outputs = check_outputs_exist(self.context, stage)
        recorded = self.state.is_recorded_complete(stage)

        # A stage whose output depends on the command line is only reusable when that
        # command line has not changed: asking `plot` for different genes must not
        # hand back the previous figures as "already complete".
        stale_args = "" if force else recorded_args_mismatch(spec, self.context, self.state)
        if stale_args:
            recorded = False

        if recorded and present and not force:
            self.state.skip_stage(stage, "already complete; artefacts present")
            return StageOutcome(
                stage=stage,
                outcome=OUTCOME_REUSED,
                message="already complete",
            )
        if stale_args:
            print(f"    re-running: {stale_args}")
            removed = self._remove_previous_artefacts(stage)
            if removed:
                print(f"    removed {removed} file(s) the previous arguments produced")

        # ---- the gate ----------------------------------------------------- #
        gate = evaluate_gate(self.state, self.context, stage)
        if not gate.ok:
            if self.config.dry_run:
                print(
                    f"    note: would first need to complete: {', '.join(gate.missing)}"
                )
            else:
                raise StagePrerequisiteError(
                    stage=stage,
                    missing=gate.missing,
                    reason=(
                        f"refusing to run stage '{stage}': "
                        + "; ".join(f"{n} ({gate.reasons.get(n, 'not ready')})" for n in gate.missing)
                    ),
                    hint=_prerequisite_hint(stage, gate.missing, self.layout.run_name),
                )

        if self.config.dry_run:
            # A dry run prints the plan; missing packages are reported but do not
            # stop it, so the plan can be inspected before installing anything.
            absent = [
                name
                for name in stage_packages(stage)
                if resolve_import_name_by_name(name) is None
            ]
            for name in absent:
                print(f"    note: '{name}' is not installed in this environment")
        else:
            self.check_stage_dependencies(stage)

        commands = build_commands(stage, self.context)
        out_dir = self.layout.root / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"{stage}.log"
        env = build_stage_environment()

        description = "; ".join(c.rendered() for c in commands)
        print(f"\n=== stage {stage} " + "=" * max(0, 60 - len(stage)))
        print(f"    {spec.summary}")

        if self.config.dry_run:
            for command in commands:
                print(f"    would run: {command.rendered()}")
            return StageOutcome(
                stage=stage,
                outcome=OUTCOME_PLANNED,
                commands=[c.rendered() for c in commands],
            )

        self.state.begin_stage(stage, command=description, samples=self.context.sample_names)
        started = time.time()
        failures: List[str] = []
        for command in commands:
            result = run_command(
                command,
                log_path=log_path,
                env=env,
                echo=True,
                dry_run=False,
                stream=self.config.log_to_console,
            )
            if not result.ok:
                failures.append(
                    f"{command.label or command.rendered()} exited with {result.returncode}"
                )
                if not self.config.keep_going:
                    break

        duration = round(time.time() - started, 2)
        if failures:
            message = "; ".join(failures)
            self.state.fail_stage(stage, message, returncode=1)
            print(f"    FAILED: {message}\n    log: {log_path}")
            if not self.config.keep_going:
                raise StageFailedError(
                    stage=stage,
                    returncode=1,
                    log_path=str(log_path),
                    hint=(
                        "Open the log above for the full output, fix the problem, then re-run the "
                        "same command. Stages that already finished are reused, so nothing "
                        "computed so far is lost."
                    ),
                )
            return StageOutcome(
                stage=stage,
                outcome=OUTCOME_FAILED,
                message=message,
                duration_s=duration,
                log_path=log_path,
                commands=[c.rendered() for c in commands],
            )

        # Normalise whatever the commands produced before verifying it, so a
        # single naming difference cannot fail an otherwise successful stage.
        notes = run_post_hook(self.context, stage)
        for note in notes:
            print(f"    {note}")

        present, missing_outputs = check_outputs_exist(self.context, stage)
        if not present:
            shown = "\n".join(f"      - {p}" for p in missing_outputs[:8])
            more = (
                f"\n      (+{len(missing_outputs) - 8} more)" if len(missing_outputs) > 8 else ""
            )
            message = (
                f"the command(s) succeeded but {len(missing_outputs)} declared output file(s) "
                f"are missing:\n{shown}{more}"
            )
            self.state.fail_stage(stage, message, returncode=0, missing_outputs=missing_outputs)
            print(f"    FAILED: {message}")
            raise StageFailedError(
                stage=stage,
                returncode=0,
                log_path=str(log_path),
                hint=_missing_output_hint(stage, self.context, log_path),
            )

        self.state.complete_stage(
            stage,
            outputs=[str(p) for p in outputs],
            returncode=0,
            message=f"{len(commands)} command(s)",
            extra={
                "log": log_path.as_posix(),
                "mode": self.config.mode,
                # Remember what the stage was asked to do, and which files that
                # produced, so a later run with different arguments is neither
                # mistaken for already complete nor left beside stale output.
                **(
                    {
                        "args_fingerprint": spec.args_fingerprint(self.context),
                        "artefacts": self._artefacts_for(spec),
                    }
                    if spec.args_fingerprint is not None
                    else {}
                ),
            },
        )
        print(f"    done in {duration:.1f}s  (log: {log_path})")
        return StageOutcome(
            stage=stage,
            outcome=OUTCOME_COMPLETE,
            duration_s=duration,
            log_path=log_path,
            commands=[c.rendered() for c in commands],
        )

    # ------------------------------------------------------------ reporting
    def status(self) -> str:
        lines = [
            self.state.describe(),
            "",
            "Stage status:",
            format_stage_summary(self.state, self.context),
        ]
        return "\n".join(lines)


def _missing_output_hint(stage: str, ctx: StageContext, log_path: Path) -> str:
    """Explain a stage that exited 0 without producing its declared outputs."""
    from .naming import rescue_hint

    if stage == "segment":
        return (
            "code/create_patches_fp.py exited cleanly but produced no tissue mask, which means "
            "it found no slide to process.\n\n"
            + rescue_hint(ctx.layout, ctx.samples)
            + f"\n\nWrote no output for:\n    {ctx.layout.masks_dir}"
        )
    lines = [
        "The stage exited cleanly but did not produce what the pipeline expects.",
        "Check that the sample names in the sheet match the files the stage writes, then read "
        f"{log_path.name} in the run's logs/ directory.",
    ]
    if stage in ("predict_lowres", "predict_highres"):
        lines.append(
            "For predictions, confirm that the feature files under "
            "02_features/UNI_multiscale_patches/ exist for every sample in the sheet."
        )
        lines.append(
            "The high-resolution filename depends on --uint8-output and --compress-output; "
            "the stage expects the one those flags select, and the writer uses the same rule "
            "(aurora/naming.py)."
        )
    return "\n".join(lines)


def recorded_args_mismatch(spec: Any, context: StageContext, state: RunState) -> str:
    """Why ``spec`` must be re-run although it is marked complete, or ``""``.

    Stages such as ``plot`` produce whatever the command line asked for, so being
    "complete" says nothing about a *different* request.  The fingerprint it recorded
    is compared with the current one.
    """
    if spec.args_fingerprint is None:
        return ""
    wanted = spec.args_fingerprint(context)
    record = state.record(spec.name)
    had = str(((record.extra if record else {}) or {}).get("args_fingerprint", ""))
    if had == wanted:
        return ""
    if not had:
        return "the previous run did not record which arguments it used"
    return "the arguments changed since it last ran"


def resolve_import_name_by_name(import_name: str) -> Optional[str]:
    """Look up a dependency by import name and return the importable spelling."""
    from .deps import resolve_import_name

    for dependency in DEPENDENCIES:
        if import_name in dependency.import_names:
            return resolve_import_name(dependency)
    return None


def _order_stages(stages: Sequence[str]) -> List[str]:
    """Dependency-first order, preserving the caller's relative order otherwise."""
    from .spec import resolve_execution_order

    return resolve_execution_order(list(stages), include_prerequisites=False)


def _prerequisite_hint(stage: str, missing: Sequence[str], run_name: str) -> str:
    lines: List[str] = []
    if stage == "predict_highres" and "predict_lowres" in missing:
        lines.append(
            "The high-resolution model refines the low-resolution prediction, so AURORA will not "
            "run it on its own. Run the low-resolution stage first."
        )
        lines.append("")
    lines.append("Run the missing stage(s) first:")
    lines.append(f"  python -m aurora run --run {run_name} --stage " + " --stage ".join(missing))
    lines.append("")
    lines.append("Or let the CLI add them for you:")
    lines.append(
        f"  python -m aurora run --run {run_name} --stage {stage} --auto-prerequisites"
    )
    lines.append("")
    lines.append(f"See what already finished with:  python -m aurora status --run {run_name}")
    return "\n".join(lines)


def default_stages(mode: str) -> Tuple[str, ...]:
    """The default stage chain for a mode."""
    if mode == MODE_TRAINING:
        return TRAINING_PIPELINE
    return INFERENCE_PIPELINE
