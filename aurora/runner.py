"""Running a stage: subprocess plumbing, logging and environment setup.

Every stage runs either

* as a **legacy script** in a subprocess (so a crash cannot take the
  orchestrator down and the full output is captured in ``logs/<stage>.log``), or
* as a **native** module of the ``aurora`` package, also in a subprocess so the
  behaviour and the logging are identical from the user's point of view.

Standard library only.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from ._legacy import code_dir, python_executable, repo_root
from .errors import StageFailedError

#: Environment variables exported to every stage subprocess.
STAGE_ENV_PASSTHROUGH: Sequence[str] = (
    "CUDA_VISIBLE_DEVICES",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TMPDIR",
    "TEMP",
    "TMP",
)


@dataclass
class Command:
    """One external process a stage wants to run."""

    argv: Sequence[str]
    label: str = ""
    cwd: Optional[Path] = None
    #: Extra environment variables for this command only.
    env: Dict[str, str] = field(default_factory=dict)
    #: When true, the pipeline may skip this command if the stage outputs exist.
    skippable: bool = True

    def rendered(self) -> str:
        return " ".join(_quote(part) for part in self.argv)


def _quote(part: str) -> str:
    text = str(part)
    if not text or any(ch in text for ch in ' \t"\'()&|<>;'):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def build_stage_environment(
    extra: Optional[Mapping[str, str]] = None,
    code_override: Optional[Path] = None,
) -> Dict[str, str]:
    """Environment for a stage subprocess.

    Guarantees that the stage can ``import`` the legacy modules and that the
    variables the original scripts expect (``CODE_PATH``, ``ISTAR_PATH``) are
    set, without clobbering anything the user exported.
    """
    env = dict(os.environ)
    code = Path(code_override) if code_override is not None else code_dir()
    env.setdefault("CODE_PATH", str(code))
    env["AURORA_CODE_PATH"] = str(code)
    env.setdefault("AURORA_REPO_ROOT", str(repo_root()))
    env.setdefault("ISTAR_PATH", str(code / "istar"))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(code), str(repo_root())] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


@dataclass
class CommandResult:
    """Outcome of one stage command."""

    command: Command
    returncode: int
    duration_s: float
    log_path: Optional[Path] = None
    tail: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def tail_text(self, lines: int = 25) -> str:
        return "\n".join(self.tail[-lines:])


def run_command(
    command: Command,
    log_path: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    echo: bool = True,
    dry_run: bool = False,
    stream: bool = True,
) -> CommandResult:
    """Execute ``command``, tee-ing its output into ``log_path``.

    Returns a :class:`CommandResult`; it does **not** raise on a non-zero exit,
    because the caller decides whether that is fatal (a stage may run several
    commands).
    """
    started = time.time()
    argv = [str(part) for part in command.argv]
    if echo:
        print(f"$ {command.rendered()}", flush=True)
    if dry_run:
        return CommandResult(command=command, returncode=0, duration_s=0.0, log_path=log_path)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        argv,
        cwd=str(command.cwd) if command.cwd else None,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )
    tail: List[str] = []
    handle = open(log_path, "a", encoding="utf-8") if log_path is not None else None
    try:
        if handle is not None:
            handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {command.rendered()}\n")
            handle.flush()
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            tail.append(line)
            if len(tail) > 200:
                del tail[:100]
            if handle is not None:
                handle.write(raw_line)
                handle.flush()
            if stream:
                print(line, flush=True)
        returncode = process.wait()
    finally:
        if handle is not None:
            handle.close()
    return CommandResult(
        command=command,
        returncode=returncode,
        duration_s=round(time.time() - started, 3),
        log_path=log_path,
        tail=tail,
    )


def run_command_or_raise(
    command: Command,
    log_path: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    stage: str = "",
    dry_run: bool = False,
) -> CommandResult:
    """Like :func:`run_command` but raises :class:`StageFailedError` on failure."""
    result = run_command(command, log_path=log_path, env=env, dry_run=dry_run)
    if not result.ok:
        raise StageFailedError(
            stage=stage or command.label or "stage",
            returncode=result.returncode,
            log_path=str(log_path) if log_path else None,
            hint=(
                "The last lines of the log were:\n"
                + result.tail_text(25)
                + "\n\nFix the reported problem and re-run the same command; finished stages "
                "are reused, so nothing already computed is thrown away."
            ),
        )
    return result


def native_stage_command(module: str, arguments: Sequence[str]) -> Command:
    """A command that runs one of this package's own modules."""
    return Command(
        argv=[python_executable(), "-m", module, *[str(a) for a in arguments]],
        label=module,
    )


def legacy_stage_command(script: str, arguments: Sequence[str], label: str = "") -> Command:
    """A command that runs an original script from ``code/``."""
    return Command(
        argv=[python_executable(), str(code_dir() / script), *[str(a) for a in arguments]],
        label=label or script,
    )


def python_version() -> str:
    return sys.version.split()[0]
