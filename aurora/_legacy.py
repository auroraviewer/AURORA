"""Bridge to the legacy ``code/`` directory.

The reorganisation keeps every original script exactly as it was; the new
package drives them instead of replacing them.  This module is the only place
that knows where the legacy sources live and how to import them.

Discovery order for the legacy code directory:

1. the ``AURORA_CODE_PATH`` environment variable,
2. ``<repository root>/code`` computed from this file's location,
3. ``<current working directory>/code``.

Nothing here imports a heavy third-party package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

from .errors import AuroraError

#: Environment variable that overrides the legacy code location.
CODE_PATH_ENV = "AURORA_CODE_PATH"

#: Environment variable that overrides the repository root.
REPO_ROOT_ENV = "AURORA_REPO_ROOT"

#: Environment variable that overrides the Python interpreter used for
#: subprocesses (useful when AURORA lives in a conda env).
PYTHON_ENV = "AURORA_PYTHON"

_cached_code_dir: Optional[Path] = None
_imported: Dict[str, ModuleType] = {}


def repo_root() -> Path:
    """Absolute path of the repository that contains this package."""
    override = os.environ.get(REPO_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    # aurora/_legacy.py -> aurora/ -> <repo>/
    return Path(__file__).resolve().parent.parent


def _looks_like_code_dir(path: Path) -> bool:
    """A directory holding the original scripts."""
    return (path / "model.py").is_file() and (path / "predict.py").is_file()


def code_search_paths() -> List[Path]:
    """Every place the original ``code/`` directory is looked for, in order.

    The package is importable from anywhere once installed, but the scripts it
    drives live in the checkout, so the search has to cover both layouts:

    * ``pip install -e .`` -- ``repo_root()`` is the checkout, so
      ``<checkout>/code`` is found directly;
    * ``pip install .`` -- ``repo_root()`` is site-packages, so the checkout is
      located from the working directory, or named with ``AURORA_CODE_PATH``.
    """
    candidates: List[Path] = []
    override = os.environ.get(CODE_PATH_ENV)
    if override:
        expanded = Path(override).expanduser()
        # Accept either the code directory itself or the repository root.
        candidates.extend([expanded, expanded / "code"])
    candidates.append(repo_root() / "code")

    cwd = Path.cwd()
    if _looks_like_code_dir(cwd):
        candidates.append(cwd)
    candidates.append(cwd / "code")
    # Running from a subdirectory of the checkout (a project directory, say):
    # walk up looking for a sibling `code/`.
    for parent in list(cwd.parents)[:6]:
        if _looks_like_code_dir(parent):
            candidates.append(parent)
        candidates.append(parent / "code")
    return candidates


def find_code_dir(required: bool = True) -> Optional[Path]:
    """Locate the legacy ``code/`` directory."""
    global _cached_code_dir
    if _cached_code_dir is not None and _cached_code_dir.is_dir():
        return _cached_code_dir

    candidates = code_search_paths()
    for candidate in candidates:
        resolved = candidate.resolve()
        if _looks_like_code_dir(resolved):
            _cached_code_dir = resolved
            return resolved
    if required:
        searched = "\n".join(f"  - {c}" for c in candidates)
        raise AuroraError(
            "could not find the AURORA legacy code directory (expected a folder "
            "containing model.py and predict.py)",
            hint=(
                "AURORA drives the original scripts in `code/`, which are not part of the "
                "installed package. Point at your checkout:\n"
                "  export AURORA_CODE_PATH=/path/to/AURORA/code      # bash\n"
                "  $env:AURORA_CODE_PATH = 'C:\\path\\to\\AURORA\\code'  # PowerShell\n"
                "Installing from the checkout with `pip install -e .` avoids this entirely, "
                "because the code directory is then found next to the package.\n"
                f"Searched:\n{searched}"
            ),
        )
    return None


def code_dir() -> Path:
    """The legacy code directory; raises when it cannot be found."""
    found = find_code_dir(required=True)
    assert found is not None
    return found


def ensure_code_on_path() -> Path:
    """Put the legacy code directory on ``sys.path`` (idempotent)."""
    directory = code_dir()
    entry = str(directory)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return directory


def legacy_path(*parts: str) -> Path:
    """Path inside the legacy code directory."""
    return code_dir().joinpath(*parts)


def legacy_script(name: str, required: bool = True) -> Optional[Path]:
    """Absolute path of a legacy script, e.g. ``'predict.py'``."""
    path = legacy_path(name)
    if path.is_file():
        return path
    if required:
        raise AuroraError(
            f"legacy script '{name}' was not found under {code_dir()}",
            hint=(
                "The new pipeline drives the original scripts rather than deleting them. "
                "Make sure the repository is complete, or set AURORA_CODE_PATH."
            ),
        )
    return None


def import_legacy(module_name: str) -> ModuleType:
    """Import a module from the legacy ``code/`` directory (cached)."""
    if module_name in _imported:
        return _imported[module_name]
    ensure_code_on_path()
    import importlib

    module = importlib.import_module(module_name)
    _imported[module_name] = module
    return module


def python_executable() -> str:
    """Interpreter used for subprocess stages.

    Defaults to the interpreter running this process so that a conda/virtual
    environment stays consistent between the orchestrator and the stages.
    """
    override = os.environ.get(PYTHON_ENV)
    if override:
        return override
    return sys.executable or "python"


def describe_environment() -> str:
    """Short report used by ``aurora doctor``."""
    found = find_code_dir(required=False)
    lines = [
        f"repository root : {repo_root()}",
        f"legacy code dir : {found if found else '(NOT FOUND)'}",
        f"python          : {python_executable()}",
    ]
    if found is not None:
        for script in (
            "create_patches_fp.py",
            "rescale.py",
            "extract_features.py",
            "extract_features_highres.py",
            "train.py",
            "train_highres.py",
            "plot.py",
        ):
            marker = "ok " if (found / script).is_file() else "MISSING"
            lines.append(f"  [{marker}] {script}")
    return "\n".join(lines)
