"""Per-run state and the stage gate.

Two responsibilities:

1. **Record** what each stage did (``run.json``): command, timestamps, return
   code, produced files, and a fingerprint of the sample set it ran on.
2. **Gate** the pipeline: :func:`assert_prerequisites` refuses to start a stage
   whose prerequisites are not complete *and* whose declared artefacts are
   genuinely present on disk.

Requirement #2 of the reorganisation -- "prevent the users from only running the
high-resolution model without running the low-resolution ones first" -- is
enforced here.  It is deliberately fail-closed:

* the state file alone is not trusted; the artefacts are re-checked every time,
* a stage recorded against a *different* sample set is treated as stale,
* if the combined checkpoint advertises a prerequisite graph that disagrees with
  :mod:`aurora.spec`, the stricter union of the two is used.

Pure standard library.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .errors import LayoutError, StagePrerequisiteError
from .paths import RunLayout
from .spec import (
    STAGE_ORDER,
    StageContext,
    check_outputs_exist,
    prerequisites_of,
    resolve_execution_order,
)
from .version import RUN_FORMAT, RUN_FORMAT_VERSION, __version__

#: Values of:attr:`StageRecord.status`.
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class StageRecord:
    """One entry of ``run.json['stages']``."""

    name: str
    status: str = STATUS_RUNNING
    started_utc: str = ""
    finished_utc: str = ""
    duration_s: float = 0.0
    command: str = ""
    returncode: Optional[int] = None
    outputs: List[str] = field(default_factory=list)
    missing_outputs: List[str] = field(default_factory=list)
    samples_fingerprint: str = ""
    samples: List[str] = field(default_factory=list)
    message: str = ""
    reused: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.status == STATUS_COMPLETE

    @classmethod
    def from_json(cls, name: str, data: Mapping[str, Any]) -> "StageRecord":
        known = {f for f in cls.__dataclass_fields__ if f != "name"}
        return cls(
            name=name,
            **{k: v for k, v in data.items() if k in known},
            extra={k: v for k, v in data.items() if k not in known},
        )

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra") or {}
        data.update(extra)
        return data

    def describe(self) -> str:
        stamp = self.finished_utc or self.started_utc or "?"
        line = f"{self.name:<24} {self.status:<9} {stamp}"
        if self.returncode not in (None, 0):
            line += f"  exit={self.returncode}"
        if self.message:
            line += f"  ({self.message})"
        return line


@dataclass
class RunState:
    """The ``run.json`` document of one run directory."""

    layout: RunLayout
    data: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def load(cls, layout: RunLayout, create: bool = False) -> "RunState":
        """Read ``run.json``; with ``create=True`` initialise a fresh document."""
        if layout.run_json.is_file():
            with open(layout.run_json, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("format") != RUN_FORMAT:
                raise LayoutError(
                    f"{layout.run_json} is not an AURORA run file "
                    f"(format marker {data.get('format')!r})",
                    hint="Pick a different --run name, or delete the directory if it is a leftover.",
                )
            return cls(layout, data)
        if not create:
            raise LayoutError(
                f"no run state found at {layout.run_json}",
                hint=(
                    "Create the run first with e.g.\n"
                    "  python -m aurora run --run <name> --samples samples.csv --model <combined.pth>"
                ),
            )
        return cls.new(layout)

    @classmethod
    def new(cls, layout: RunLayout) -> "RunState":
        state = cls(
            layout,
            {
                "format": RUN_FORMAT,
                "format_version": RUN_FORMAT_VERSION,
                "aurora_version": __version__,
                "run_name": layout.run_name,
                "project_path": layout.project_path.as_posix(),
                "run_root": layout.root.as_posix(),
                "created_utc": utc_now_iso(),
                "updated_utc": utc_now_iso(),
                "mode": "",
                "model": {},
                "samples": {},
                "layout": layout.describe(),
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "hostname": socket.gethostname(),
                },
                "stages": {},
                "history": [],
            },
        )
        return state

    def save(self) -> Path:
        """Atomically write ``run.json``."""
        self.data["updated_utc"] = utc_now_iso()
        self.data["layout"] = self.layout.describe()
        self.layout.root.mkdir(parents=True, exist_ok=True)
        tmp = self.layout.run_json.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2, sort_keys=False)
            handle.write("\n")
        os.replace(tmp, self.layout.run_json)
        return self.layout.run_json

    # -------------------------------------------------------------- metadata
    def set_model(self, **info: Any) -> None:
        self.data.setdefault("model", {}).update(info)

    def set_samples(self, **info: Any) -> None:
        self.data.setdefault("samples", {}).update(info)

    def set_mode(self, mode: str) -> None:
        self.data["mode"] = mode

    @property
    def mode(self) -> str:
        """``inference`` or ``training``, as recorded for this run."""
        return str(self.data.get("mode") or "inference")

    @property
    def model_info(self) -> Dict[str, Any]:
        return dict(self.data.get("model", {}))

    @property
    def samples_info(self) -> Dict[str, Any]:
        return dict(self.data.get("samples", {}))

    @property
    def samples_fingerprint(self) -> str:
        return str(self.samples_info.get("fingerprint", ""))

    # ----------------------------------------------------------- stage access
    def stages(self) -> Dict[str, StageRecord]:
        raw = self.data.get("stages", {}) or {}
        return {name: StageRecord.from_json(name, payload) for name, payload in raw.items()}

    def record(self, stage: str) -> Optional[StageRecord]:
        raw = (self.data.get("stages", {}) or {}).get(stage)
        return StageRecord.from_json(stage, raw) if raw else None

    def is_recorded_complete(self, stage: str) -> bool:
        record = self.record(stage)
        return bool(record and record.is_complete)

    def _store(self, record: StageRecord) -> None:
        self.data.setdefault("stages", {})[record.name] = record.to_json()

    def _log_history(self, stage: str, action: str, detail: str = "") -> None:
        self.data.setdefault("history", []).append(
            {"utc": utc_now_iso(), "stage": stage, "action": action, "detail": detail}
        )

    def begin_stage(
        self,
        stage: str,
        command: str = "",
        samples: Optional[Sequence[str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> StageRecord:
        """Mark ``stage`` as running."""
        record = StageRecord(
            name=stage,
            status=STATUS_RUNNING,
            started_utc=utc_now_iso(),
            command=command,
            samples_fingerprint=self.samples_fingerprint,
            samples=list(samples or []),
            extra=dict(extra or {}),
        )
        self._store(record)
        self._log_history(stage, "begin", command)
        self.save()
        return record

    def complete_stage(
        self,
        stage: str,
        outputs: Optional[Sequence[os.PathLike | str]] = None,
        returncode: int = 0,
        message: str = "",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> StageRecord:
        """Mark ``stage`` as complete, storing the artefacts it produced."""
        record = self.record(stage) or StageRecord(name=stage)
        record.status = STATUS_COMPLETE
        record.finished_utc = utc_now_iso()
        record.returncode = returncode
        record.message = message
        record.outputs = [Path(p).as_posix() for p in (outputs or [])]
        record.missing_outputs = []
        if extra:
            record.extra.update(extra)
        started = _parse_iso(record.started_utc)
        if started is not None:
            record.duration_s = round((datetime.now(timezone.utc) - started).total_seconds(), 3)
        self._store(record)
        self._log_history(stage, "complete", message)
        self.save()
        return record

    def fail_stage(
        self,
        stage: str,
        message: str,
        returncode: Optional[int] = None,
        missing_outputs: Optional[Sequence[os.PathLike | str]] = None,
    ) -> StageRecord:
        """Mark ``stage`` as failed (its outputs are *not* trusted afterwards)."""
        record = self.record(stage) or StageRecord(name=stage)
        record.status = STATUS_FAILED
        record.finished_utc = utc_now_iso()
        record.returncode = returncode
        record.message = message
        record.missing_outputs = [Path(p).as_posix() for p in (missing_outputs or [])]
        started = _parse_iso(record.started_utc)
        if started is not None:
            record.duration_s = round((datetime.now(timezone.utc) - started).total_seconds(), 3)
        self._store(record)
        self._log_history(stage, "fail", message)
        self.save()
        return record

    def skip_stage(self, stage: str, message: str) -> StageRecord:
        """Record that an already-complete stage was reused instead of re-run."""
        record = self.record(stage) or StageRecord(name=stage)
        record.message = message
        record.reused = True
        self._store(record)
        self._log_history(stage, "reuse", message)
        self.save()
        return record

    def invalidate(self, stage: str, cascade: bool = True, reason: str = "") -> List[str]:
        """Forget ``stage`` (and, by default, everything downstream of it)."""
        from .spec import dependents_of

        affected = [stage]
        if cascade:
            pending = list(dependents_of(stage))
            while pending:
                name = pending.pop(0)
                if name in affected:
                    continue
                affected.append(name)
                pending.extend(dependents_of(name))
        stages = self.data.setdefault("stages", {})
        for name in affected:
            if name in stages:
                stages[name]["status"] = STATUS_SKIPPED
                stages[name]["message"] = reason or f"invalidated by re-running {stage}"
                stages[name]["reused"] = False
        self._log_history(stage, "invalidate", reason or f"cascade: {', '.join(affected)}")
        self.save()
        return affected

    # ----------------------------------------------------------- reporting
    def describe(self) -> str:
        info = self.samples_info
        lines = [
            f"Run '{self.layout.run_name}'  ({self.layout.root})",
            f"  mode      : {self.data.get('mode') or '(unset)'}",
        ]
        model = self.model_info
        if model:
            lines.append(f"  model     : {model.get('model_name') or model.get('path', '?')}")
        if info:
            lines.append(f"  samples   : {info.get('count', '?')} from {info.get('path', '?')}")
        lines.append("  stages:")
        records = self.stages()
        if not records:
            lines.append("    (nothing has run yet)")
        for name in STAGE_ORDER:
            record = records.get(name)
            if record is not None:
                lines.append("    " + record.describe())
        for name, record in records.items():
            if name not in STAGE_ORDER:
                lines.append("    " + record.describe())
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.data, indent=2, sort_keys=False)


def _parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


@dataclass
class GateResult:
    """Outcome of :func:`evaluate_gate`."""

    stage: str
    ok: bool
    missing: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        if self.ok:
            return f"stage '{self.stage}': prerequisites satisfied"
        lines = [f"stage '{self.stage}': prerequisites NOT satisfied"]
        for name in self.missing:
            lines.append(f"  - {name}: {self.reasons.get(name, 'not complete')}")
        return "\n".join(lines)


def _checkpoint_requires(state: RunState, stage: str) -> List[str]:
    """Prerequisites the combined checkpoint itself advertises for ``stage``."""
    raw = state.model_info.get("requires")
    if isinstance(raw, Mapping):
        value = raw.get(stage)
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
    return []


def effective_prerequisites(state: RunState, stage: str) -> List[str]:
    """Union of the code's prerequisites and the checkpoint's, in pipeline order.

    Taking the union means a checkpoint cannot be used to *weaken* the gate: if
    either source says the low-resolution model must run first, it must.

    Mode matters: ``rescale`` needs ``segment`` only for inference, so a training
    run must not be gated on a stage it never uses.
    """
    mode = state.mode
    declared = set(prerequisites_of(stage, mode=mode))
    declared.update(_checkpoint_requires(state, stage))
    if not declared:
        return []
    # expand transitively so the error message lists the whole chain
    expanded: List[str] = []
    for name in resolve_execution_order(
        sorted(declared, key=lambda n: STAGE_ORDER.index(n)), mode=mode
    ):
        if name not in expanded:
            expanded.append(name)
    expanded = [n for n in expanded if n != stage]
    return [n for n in STAGE_ORDER if n in expanded] + [n for n in expanded if n not in STAGE_ORDER]


def evaluate_gate(
    state: RunState,
    ctx: StageContext,
    stage: str,
    verify_artifacts: bool = True,
) -> GateResult:
    """Decide whether ``stage`` may run, without raising."""
    result = GateResult(stage=stage, ok=True)
    current_fingerprint = ctx.samples.fingerprint()
    for prerequisite in effective_prerequisites(state, stage):
        record = state.record(prerequisite)
        if record is None or not record.is_complete:
            result.ok = False
            result.missing.append(prerequisite)
            status = record.status if record else "never run"
            result.reasons[prerequisite] = f"status is '{status}'"
            continue
        if record.samples_fingerprint and record.samples_fingerprint != current_fingerprint:
            result.ok = False
            result.missing.append(prerequisite)
            result.reasons[prerequisite] = (
                "it ran on a different sample set than the one requested now "
                "(re-run it, or use the same --samples)"
            )
            continue
        if verify_artifacts:
            present, missing_paths = check_outputs_exist(ctx, prerequisite)
            if not present:
                result.ok = False
                result.missing.append(prerequisite)
                shown = ", ".join(p.name for p in missing_paths[:3])
                more = f" (+{len(missing_paths) - 3} more)" if len(missing_paths) > 3 else ""
                result.reasons[prerequisite] = (
                    f"it is marked complete but its output files are missing: {shown}{more}"
                )
    # de-duplicate while preserving order
    seen: List[str] = []
    for name in result.missing:
        if name not in seen:
            seen.append(name)
    result.missing = seen
    return result


def _gate_hint(missing: Sequence[str], stage: str, state: RunState, ctx: StageContext) -> str:
    run_name = state.layout.run_name
    lines: List[str] = []
    if stage == "predict_highres" and "predict_lowres" in missing:
        lines.append(
            "The high-resolution model refines the low-resolution prediction, so it is not "
            "allowed to run on its own. Run the low-resolution stage first."
        )
        lines.append("")
    lines.append("Run the missing stage(s) first, in this order:")
    lines.append(f"  python -m aurora run --run {run_name} --stage " + " --stage ".join(missing))
    lines.append("")
    lines.append(
        "Or let the CLI add them automatically:\n"
        f"  python -m aurora run --run {run_name} --stage {stage} --auto-prerequisites"
    )
    lines.append("")
    lines.append(
        "Inspect what has already finished with:\n"
        f"  python -m aurora status --run {run_name}"
    )
    return "\n".join(lines)


def assert_prerequisites(
    state: RunState,
    ctx: StageContext,
    stage: str,
    verify_artifacts: bool = True,
) -> None:
    """Raise :class:`StagePrerequisiteError` unless ``stage`` may run.

    This is the single choke point every stage execution goes through.
    """
    result = evaluate_gate(state, ctx, stage, verify_artifacts=verify_artifacts)
    if result.ok:
        return
    details = "\n".join(
        f"  - {name}: {result.reasons.get(name, 'not complete')}" for name in result.missing
    )
    raise StagePrerequisiteError(
        stage=stage,
        missing=result.missing,
        reason=(
            f"refusing to run stage '{stage}': {len(result.missing)} prerequisite stage(s) "
            f"are not ready\n{details}"
        ),
        hint=_gate_hint(result.missing, stage, state, ctx),
    )


def missing_prerequisites(state: RunState, ctx: StageContext, stage: str) -> List[str]:
    """The prerequisite stages that still need to run, in execution order."""
    return evaluate_gate(state, ctx, stage).missing


def completed_stages(state: RunState, ctx: Optional[StageContext] = None) -> List[str]:
    """Stages recorded as complete (artefacts verified when ``ctx`` is given)."""
    done: List[str] = []
    for name in STAGE_ORDER:
        if not state.is_recorded_complete(name):
            continue
        if ctx is not None:
            present, _ = check_outputs_exist(ctx, name)
            if not present:
                continue
        done.append(name)
    return done


def format_stage_summary(state: RunState, ctx: Optional[StageContext] = None) -> str:
    """Compact ``stage -> status`` table for the CLI."""
    records = state.stages()
    width = max((len(n) for n in STAGE_ORDER), default=10)
    lines: List[str] = []
    for name in STAGE_ORDER:
        record = records.get(name)
        if record is None:
            status = "-"
            note = ""
        else:
            status = record.status
            note = record.message or ""
            if ctx is not None and record.is_complete:
                present, missing = check_outputs_exist(ctx, name)
                if not present:
                    status = "stale"
                    note = f"{len(missing)} output file(s) missing"
        lines.append(f"  {name:<{width}}  {status:<9} {note}".rstrip())
    return "\n".join(lines)
