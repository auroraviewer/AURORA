#!/usr/bin/env python
"""Verify that the mandatory histosweep stage's imports are all installable.

HistoSweep is vendored at ``code/HistoSweep`` and is now a mandatory part of the
default inference chain, so every third-party module it imports has to be in the
``histosweep`` stage's dependency group -- which is the ``inference`` group,
because ``pip install .[inference]`` must be enough to run the whole chain.

This was written after `numba` was found missing at run time
(``ModuleNotFoundError: No module named 'numba'``), with `scikit-learn` waiting
behind it: Python reports only the first missing import, so the only reliable way
to find the rest is to read them out of the source.

Imports that genuinely are not needed are listed in :data:`EXEMPT` with the
reason, so the check stays meaningful instead of being relaxed.

    python tests/test_histosweep_deps.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aurora.deps import dependencies_for_group  # noqa: E402
from aurora.spec import get_stage  # noqa: E402

#: Import name -> distribution name, for the ones that differ.
IMPORT_TO_DISTRIBUTION: Dict[str, str] = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
}

#: Imports that are present in the source but cannot be reached by this stage.
#: Each entry needs a reason; anything not here must be declared.
EXEMPT: Dict[str, str] = {
    "pyvips": (
        "code/HistoSweep/utils.py imports it at module load, but run_HistoSweep.py "
        "installs a stub before importing HistoSweep, and the real thing is only used "
        "by its rescale/preprocess helpers, which the stage does not call."
    ),
    "brokenaxes": (
        "only additionalPlots.py imports it, and run_HistoSweep.py imports "
        "computeMetrics, densityFiltering, textureAnalysis, ratioFiltering, "
        "generateMask and saveParameters -- not additionalPlots."
    ),
    "psutil": (
        "computeMetrics.get_memory_usage imports it inside a try/except ImportError "
        "and is never called by anything in HistoSweep."
    ),
}

#: Modules of the vendored package itself, which are not third-party.
LOCAL_MODULES: Set[str] = {
    path.stem for path in (REPO_ROOT / "code" / "HistoSweep").glob("*.py")
} | {"utils", "saveParameters"}

failures: List[str] = []


def expect(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def scan(paths: List[Path]) -> Dict[str, List[Tuple[Path, int]]]:
    """Third-party top-level imports, with where each was seen."""
    found: Dict[str, List[Tuple[Path, int]]] = {}
    stdlib = set(sys.stdlib_module_names)
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: List[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in stdlib or name in LOCAL_MODULES:
                    continue
                found.setdefault(name, []).append((path, getattr(node, "lineno", 0)))
    return found


def main() -> int:
    histogram = REPO_ROOT / "code" / "HistoSweep"
    expect(histogram.is_dir(), f"HistoSweep is vendored at {histogram.relative_to(REPO_ROOT)}")
    if not histogram.is_dir():
        return 1

    files = sorted(p for p in histogram.rglob("*.py") if "__pycache__" not in str(p))
    files.append(REPO_ROOT / "code" / "run_HistoSweep.py")
    imports = scan(files)
    expect(bool(imports), f"scanned {len(files)} file(s) and found third-party imports")

    stage = get_stage("histosweep")
    declared = {dep.distribution for dep in dependencies_for_group(stage.requires_group)}
    print(f"  histosweep requires the '{stage.requires_group}' group: {len(declared)} distributions")
    print()

    missing: List[str] = []
    for module in sorted(imports):
        distribution = IMPORT_TO_DISTRIBUTION.get(module, module)
        where = imports[module][0]
        location = f"{where[0].relative_to(REPO_ROOT).as_posix()}:{where[1]}"
        if distribution in declared:
            expect(True, f"{module:<14} -> {distribution:<16} declared  ({location})")
        elif module in EXEMPT:
            expect(True, f"{module:<14} -> excused: {EXEMPT[module].split('.')[0]}.")
        else:
            missing.append(
                f"{module} (imported at {location}) maps to distribution "
                f"{distribution!r}, which .[{stage.requires_group}] does not install"
            )
            expect(False, f"{module:<14} -> {distribution:<16} MISSING   ({location})")

    expect(
        not missing,
        "every reachable HistoSweep import is installable"
        if not missing
        else "missing: " + "; ".join(missing),
    )

    # The exemptions must not rot: if one becomes declared, drop the entry.
    for module, reason in EXEMPT.items():
        distribution = IMPORT_TO_DISTRIBUTION.get(module, module)
        expect(
            distribution not in declared,
            f"'{module}' is declared now, so its exemption ({reason.split('.')[0]}.) is stale",
        )
    print()
    if failures:
        print(f"{len(failures)} FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"histosweep dependencies verified ({len(imports)} imports, {len(EXEMPT)} exemptions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
