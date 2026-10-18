#!/usr/bin/env python
"""Prove that the inference path never reaches a training-only package.

The audit walks the real import graph:

* it starts from the inference entry points of the new ``aurora`` package and
  from the legacy scripts the pipeline drives in inference mode,
* it follows every *local* import (a module that resolves to a ``.py`` file or a
  package inside the repository, including namespace packages),
* it records the third-party top-level module of every import that executes when
  the module is imported,
* imports inside a function body are reported separately as *deferred*, because
  they only run on the branch that needs them and therefore do not force an
  install on an inference-only user.

The verdict fails if any eagerly-executed import belongs to a group that
inference-only users must not need (``training``, ``hest``, ``deconvolution``).

    python tools/audit_inference_deps.py
    python tools/audit_inference_deps.py --json          # machine readable
    python tools/audit_inference_deps.py --show-lazy     # list deferred imports
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aurora.deps import group_of  # noqa: E402

CODE_DIR = REPO_ROOT / "code"
AURORA_DIR = REPO_ROOT / "aurora"

#: Import names that are never third-party.
STDLIB = set(getattr(sys, "stdlib_module_names", ()))
STDLIB.update({"__future__", "typing_extensions", "typing"})

#: Groups an inference-only installation may use.
INFERENCE_GROUPS = {"core", "inference", "histosweep", "catching", "unknown"}

#: Groups that must NOT be reachable from the inference path.
TRAINING_GROUPS = {"training", "hest", "deconvolution", "dev"}

#: Entry points of the new package that must be safe for inference-only users.
NATIVE_ENTRY_POINTS: Tuple[str, ...] = (
    "aurora/__main__.py",
    "aurora/cli.py",
    "aurora/pipeline.py",
    "aurora/commands.py",
    "aurora/prepare.py",
    "aurora/state.py",
    "aurora/spec.py",
    "aurora/paths.py",
    "aurora/samples.py",
    "aurora/assets.py",
    "aurora/bundle.py",
    "aurora/checkpoint.py",
    "aurora/config.py",
    "aurora/runner.py",
    "aurora/_legacy.py",
    "aurora/inference/predict_lowres.py",
    "aurora/inference/predict_highres.py",
    "aurora/inference/common.py",
)

#: Legacy scripts the new pipeline drives for an inference run.
LEGACY_ENTRY_POINTS: Tuple[str, ...] = (
    "code/create_patches_fp.py",
    "code/rescale.py",
    "code/extract_features.py",
    "code/extract_features_highres.py",
    "code/plot.py",
    "code/run_HistoSweep.py",
)

#: Why each deferred import is allowed to be deferred.
EXPECTED_LAZY: Dict[str, str] = {
    "sklearn": "KMeans / MiniBatchKMeans only draw a --plot check figure",
    "matplotlib": "figures are only drawn by --plot",
    "scanpy": "only the training branch of rescale.py reads .h5ad files",
    "geopandas": "replaced by a standard-library GeoJSON reader in rescale.py",
    "torch_geometric": "only the GCN / GraphLSTM model variants need it",
    "pyvips": "only the CATCH / HEST preprocessing branches need it",
    "hest": "only HEST_preprocess.py needs it",
    "tensorboard": "only the training stages log to TensorBoard",
    "zstandard": "only --compress-output writes .npz.zst",
    "pybiomart": "only gene annotation during training",
}

#: Directories never worth scanning.
SKIP_DIRS = {"__pycache__", ".git", ".github", "code_dev", "docs", "_selftest"}


# --------------------------------------------------------------------------- #
# Repository module index
# --------------------------------------------------------------------------- #


def build_index() -> Dict[str, Path]:
    """Map every importable top-level name in the repository to its source file."""
    index: Dict[str, Path] = {}
    for root in (CODE_DIR, AURORA_DIR, REPO_ROOT):
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if len(relative.parts) == 1:
                index.setdefault(path.stem, path)
                continue
            top = relative.parts[0]
            if path.name == "__init__.py":
                index.setdefault(top, path)
            else:
                index.setdefault(top, path)  # namespace package: use the first module
    # HistoSweep modules are imported after a runtime sys.path insert.
    for path in (CODE_DIR / "HistoSweep").glob("*.py"):
        index.setdefault(path.stem, path)
    for path in (CODE_DIR / "wsi_core").glob("*.py"):
        index.setdefault(path.stem, path)
    return index


_INDEX: Dict[str, Path] = {}


def resolve_local(name: str, index: Dict[str, Path]) -> Optional[Path]:
    """Return the file a local import refers to, or ``None`` when it is external."""
    top = name.split(".")[0]
    return index.get(top)


# --------------------------------------------------------------------------- #
# AST collection
# --------------------------------------------------------------------------- #


class ImportCollector(ast.NodeVisitor):
    """Collect eager and deferred imports, honouring scopes and TYPE_CHECKING."""

    def __init__(self) -> None:
        self.eager: Set[str] = set()
        self.deferred: Set[str] = set()
        self._in_function = 0
        self._type_checking = 0

    # -- scopes ---------------------------------------------------------- #
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._in_function += 1
        self.generic_visit(node)
        self._in_function -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self._in_function += 1
        self.generic_visit(node)
        self._in_function -= 1

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        guarded = _is_type_checking(node.test)
        self._type_checking += int(guarded)
        self.generic_visit(node)
        self._type_checking -= int(guarded)

    # -- imports --------------------------------------------------------- #
    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        if self._type_checking:
            return
        for alias in node.names:
            self._add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if self._type_checking or not node.module:
            return
        self._add(node.module)

    def _add(self, dotted: str) -> None:
        target = self.deferred if self._in_function else self.eager
        target.add(dotted if self._in_function else dotted.split(".")[0])


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


@dataclass
class ModuleImports:
    """Imports collected from one source file."""

    path: Path
    eager: Set[str] = field(default_factory=set)
    deferred: Set[str] = field(default_factory=set)


def analyse_file(path: Path) -> ModuleImports:
    collector = ImportCollector()
    collector.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return ModuleImports(path=path, eager=collector.eager, deferred=collector.deferred)


# --------------------------------------------------------------------------- #
# Graph walk
# --------------------------------------------------------------------------- #


def walk_graph(
    entry_points: Sequence[Path], index: Dict[str, Path]
) -> Tuple[Dict[Path, ModuleImports], Set[str], Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Follow local imports from ``entry_points``.

    Returns ``(per_file, eager, deferred, who_eager, who_deferred)``.
    """
    seen: Dict[Path, ModuleImports] = {}
    eager: Set[str] = set()
    deferred: Set[str] = set()
    who: Dict[str, Set[str]] = {}
    who_deferred: Dict[str, Set[str]] = {}
    queue: List[Path] = [p for p in entry_points if p.is_file()]
    while queue:
        path = queue.pop(0).resolve()
        if path in seen:
            continue
        info = analyse_file(path)
        seen[path] = info
        for name in sorted(info.eager):
            if name in STDLIB:
                continue
            local = resolve_local(name, index)
            if local is not None:
                if local not in seen and local not in queue:
                    queue.append(local)
                continue
            eager.add(name)
            who.setdefault(name, set()).add(_rel(path))
        for dotted in sorted(info.deferred):
            top = dotted.split(".")[0]
            if top in STDLIB:
                continue
            if resolve_local(top, index) is not None:
                continue
            deferred.add(top)
            who_deferred.setdefault(top, set()).add(_rel(path))
    return seen, eager, deferred, who, who_deferred


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    parser.add_argument("--show-lazy", action="store_true", help="Also list the deferred imports")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail when a package is missing from the dependency policy (default: only warn)",
    )
    args = parser.parse_args(argv)

    index = build_index()
    entries = [REPO_ROOT / p for p in NATIVE_ENTRY_POINTS + LEGACY_ENTRY_POINTS]
    missing_files = [p for p in entries if not p.is_file()]
    per_file, eager, deferred, who, who_deferred = walk_graph(entries, index)

    offenders = [
        {"module": module, "group": group_of(module), "imported_by": sorted(who.get(module, ()))}
        for module in sorted(eager)
        if group_of(module) in TRAINING_GROUPS
    ]
    unknown = sorted(m for m in eager if group_of(m) == "unknown")
    undeclared_deferred = sorted(
        m for m in deferred if group_of(m) == "unknown" and m not in EXPECTED_LAZY
    )

    verdict = "PASS"
    if offenders or missing_files:
        verdict = "FAIL"
    if args.strict and (unknown or undeclared_deferred):
        verdict = "FAIL"

    report = {
        "entry_points": [_rel(p) for p in entries],
        "missing_entry_points": [_rel(p) for p in missing_files],
        "files_analysed": len(per_file),
        "eager_third_party": sorted(eager),
        "eager_groups": {m: group_of(m) for m in sorted(eager)},
        "deferred_third_party": sorted(deferred),
        "training_only_offenders": offenders,
        "unknown_eager_packages": unknown,
        "unknown_deferred_packages": undeclared_deferred,
        "verdict": verdict,
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if verdict == "PASS" else 1

    width = max((len(m) for m in eager | deferred), default=12)
    print("AURORA inference dependency audit")
    print("=" * 72)
    print(f"entry points analysed : {len(entries)}")
    print(f"source files reached  : {len(per_file)}")
    print()
    print("Third-party modules imported eagerly (all must be installed for inference):")
    for module in sorted(eager):
        group = group_of(module)
        marker = "TRAINING-ONLY" if group in TRAINING_GROUPS else group
        print(f"  {module:<{width}}  [{marker}]")
    print()
    print("Deferred imports (only executed on the branch that needs them):")
    for module in sorted(deferred):
        group = group_of(module)
        reason = EXPECTED_LAZY.get(module, "")
        suffix = f"  -- {reason}" if reason else ""
        print(f"  {module:<{width}}  [{group}]{suffix}")
    print()

    if missing_files:
        print("Missing entry points:")
        for path in missing_files:
            print(f"  - {_rel(path)}")
    if offenders:
        print("FAIL: the inference path reaches training-only packages")
        for entry in offenders:
            print(f"  - {entry['module']} [{entry['group']}] imported by: {', '.join(entry['imported_by'])}")
        print()
    if unknown:
        print("Packages not present in the dependency policy (add them to aurora/deps.py):")
        for module in unknown:
            print(f"  - {module} (imported by {', '.join(sorted(who.get(module, ())))})")
        print()
    if undeclared_deferred:
        print("Deferred packages with no documented reason (add them to EXPECTED_LAZY):")
        for module in undeclared_deferred:
            print(f"  - {module}")
        print()

    if verdict == "PASS":
        print("PASS: no inference entry point reaches a training-only package.")
        print()
        print("Install for inference only with:")
        print("  pip install -r requirements/inference.txt")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
