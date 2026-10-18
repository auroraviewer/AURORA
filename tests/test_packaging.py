#!/usr/bin/env python
"""Checks the packaging metadata without needing a build backend.

``pyproject.toml`` is easy to get subtly wrong -- a version that disagrees with
``aurora/version.py``, an entry point that does not resolve, a ``packages.find``
rule that would sweep the legacy ``code/`` directory into the wheel.  None of
that shows up until someone installs the package, so it is checked here.

The build backend itself is not invoked: ``setuptools`` is not a runtime
dependency and may not be installed.

    python tests/test_packaging.py
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

failures: list[str] = []


def expect(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def matches_any(path: str, patterns: list[str]) -> bool:
    """Minimal stand-in for setuptools' include/exclude glob matching."""
    import fnmatch

    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path.split(".")[0], pattern):
            return True
    return False


def main() -> int:
    pyproject = REPO_ROOT / "pyproject.toml"
    expect(pyproject.is_file(), "pyproject.toml exists")
    if not pyproject.is_file():
        return 1

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    build = data.get("build-system", {})

    print("1. metadata")
    expect(bool(project.get("name")), f"name = {project.get('name')!r}")
    from aurora.version import __version__

    expect(
        project.get("version") == __version__,
        f"version matches aurora/version.py ({project.get('version')!r} vs {__version__!r})",
    )
    expect(bool(project.get("description")), "description present")
    expect(
        project.get("readme") == "README.md",
        f"readme is README.md, got {project.get('readme')!r}",
    )
    expect("setuptools" in str(build.get("requires", [])), f"build requires: {build.get('requires')}")
    expect(bool(build.get("build-backend")), f"backend = {build.get('build-backend')!r}")

    print()
    print("2. the runtime package stays dependency-free")
    # `import aurora` must work before anything is installed: that is what lets
    # `--help`, `validate`, `inspect` and `migrate-layout` run anywhere.
    expect(
        project.get("dependencies") == [],
        f"core dependencies must be empty, got {project.get('dependencies')}",
    )

    print()
    print("3. the console entry point resolves")
    scripts = project.get("scripts", {})
    expect("aurora" in scripts, f"scripts = {scripts}")
    target = scripts.get("aurora", "")
    expect(":" in target, f"entry point is 'module:function', got {target!r}")
    module_name, _, function_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        expect(False, f"{module_name} is not importable ({exc})")
    else:
        function = getattr(module, function_name, None)
        expect(callable(function), f"{target} resolves to a callable")
        # console_scripts use the return value as the process exit code.
        import inspect

        signature = inspect.signature(function)
        expect(
            len(signature.parameters) <= 1,
            f"{function_name} takes at most one optional argument: {signature}",
        )

    print()
    print("4. package discovery selects aurora/ and nothing else")
    find = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
    include = find.get("include", [])
    exclude = find.get("exclude", [])
    expect(include, f"packages.find.include = {include}")
    expect(exclude, f"packages.find.exclude = {exclude}")
    expect(matches_any("aurora", include), "aurora is included")
    expect(matches_any("aurora.cli", include), "aurora.cli is included")
    for unwanted in ("code", "tools", "tests", "docs", "examples"):
        expect(
            not matches_any(unwanted, include) or matches_any(unwanted, exclude),
            f"{unwanted}/ is not collected as a package",
        )
    # `code/` has no __init__.py and is driven as subprocesses, never imported.
    expect(
        not (REPO_ROOT / "code" / "__init__.py").exists(),
        "code/ is not a package (no __init__.py), so find() cannot pick it up",
    )

    print()
    print("5. extras mirror aurora/deps.py")
    from aurora.deps import GROUP_PARENT, dependencies_for_group

    extras = project.get("optional-dependencies", {})
    expect(extras, "extras are declared")
    for group in GROUP_PARENT:
        if group == "deconvolution":
            continue  # conda + R, no pip extra
        expect(group in extras, f"extra '{group}' is declared")
    for group, declared in extras.items():
        # Source-only dependencies have no usable PyPI name, so they must not be in
        # an extra at all -- `pip install .[hest]` would fetch the wrong project.
        expected = {
            dep.requirement_line for dep in dependencies_for_group(group) if dep.installable
        }
        declared_set = {line.strip() for line in declared}
        expect(
            declared_set == expected,
            f"extra '{group}' matches deps.py "
            f"({len(declared_set)} declared, {len(expected)} expected"
            + (
                ""
                if declared_set == expected
                else f"; diff {sorted(declared_set ^ expected)}"
            )
            + ")",
        )

    print()
    print("6. the legacy scripts are reachable without installing them")
    from aurora._legacy import code_search_paths, find_code_dir

    expect(bool(code_search_paths()), "there is a search path for code/")
    found = find_code_dir(required=False)
    expect(found is not None and found.is_dir(), f"code/ resolved to {found}")

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("packaging test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
