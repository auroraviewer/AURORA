#!/usr/bin/env python
"""Generate ``requirements/*.txt`` from the dependency policy in ``aurora/deps.py``.

The policy in :mod:`aurora.deps` is what the code actually checks at run time
(``aurora doctor``, the per-stage check, and the static audit).  Generating the
requirements files from it guarantees the two can never disagree.

    python tools/sync_requirements.py            # write the files
    python tools/sync_requirements.py --check    # fail when a file is out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aurora.deps import (  # noqa: E402
    DEPENDENCIES,
    GROUP_PARENT,
    GROUP_PURPOSE,
    GROUP_REQUIREMENTS_FILE,
)

REQUIREMENTS_DIR = REPO_ROOT / "requirements"

#: Groups that get a generated file, in dependency order.
ORDERED_GROUPS: List[str] = ["core", "inference", "training", "hest", "histosweep", "catching", "compression", "dev"]

#: Extra lines appended to a group's file (things pip needs that are not a
#: plain PyPI package, or that need a special index / a manual step).
GROUP_TRAILER: Dict[str, List[str]] = {
    "core": [
        "",
        "# PyTorch: install the build that matches your CUDA version first, e.g.",
        "#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130",
        "# then install this file; pip will keep the build you already have.",
    ],
    "inference": [
        "",
        "# 'openslide-bin' ships the native OpenSlide libraries, so 'openslide-python'",
        "# works without a system-wide install. Drop it if you already have OpenSlide.",
    ],
    "hest": [
        "",
        "# HEST is not on PyPI. Install it from source:",
        "#   git clone https://github.com/mahmoodlab/HEST.git",
        "#   pip install -e HEST",
        "# 'pyvips' is only a binding: it also needs the native libvips library.",
        "#   conda install -c conda-forge pyvips   (installs the library too)",
        "#   apt-get install libvips-dev           (Debian/Ubuntu)",
        "#   brew install vips                     (macOS)",
    ],
    "histosweep": [
        "",
        "# HistoSweep ships inside this repository at code/HistoSweep, so there is",
        "# nothing to clone; pass --histosweep-path only to use a different checkout.",
        "# The stage sweeps the rescaled H&E image and writes a tissue mask that",
        "# predict_highres applies, so it is part of the default inference chain.",
    ],
    "dev": [
        "",
        "# The self-test and the audits run with the standard library plus the",
        "# training group; nothing else is required.",
    ],
    "compression": [
        "",
        "# Optional: these only make the high-resolution HDF5 smaller.",
        "# Without them AURORA writes gzip, which h5py supports natively, so a run",
        "# never depends on this file. 'hdf5plugin' also has to be installed on the",
        "# machine that *reads* the resulting file.",
        "# If pip cannot reach your index, these are the two ways that avoid a mirror:",
        "#   pip install --prefer-binary --retries 5 --timeout 60 hdf5plugin zstandard",
        "#   conda install -c conda-forge hdf5plugin zstandard",
    ],
}

HEADER = """\
# GENERATED FILE -- do not edit by hand.
# Source of truth: aurora/deps.py  (regenerate with: python tools/sync_requirements.py)
#
# Group: {group}
# Purpose: {purpose}
#
# Install with:
#   pip install -r {relative}
"""


def lines_for_group(group: str) -> List[str]:
    """Render one requirements file."""
    body: List[str] = []
    parent = GROUP_PARENT.get(group)
    if parent and parent != group:
        parent_file = GROUP_REQUIREMENTS_FILE.get(parent)
        if parent_file:
            body.append(f"-r {Path(parent_file).name}")
            body.append("")
    own = [d for d in DEPENDENCIES if d.group == group]
    own.sort(key=lambda d: d.distribution.lower())
    for dep in own:
        if not dep.installable:
            # A bare name here would make pip fetch an unrelated project under the
            # same name, or fail.  The group trailer explains the source install.
            continue
        comment = f"  # {dep.note}" if dep.note else ""
        body.append(f"{dep.requirement_line}{comment}")
    body.extend(GROUP_TRAILER.get(group, []))
    return body


def render(group: str) -> str:
    relative = GROUP_REQUIREMENTS_FILE.get(group)
    assert relative is not None
    header = HEADER.format(
        group=group,
        purpose=GROUP_PURPOSE.get(group, ""),
        relative=Path(relative).as_posix(),
    )
    return header + "\n".join(lines_for_group(group)) + "\n"


def check_pyproject_extras() -> List[str]:
    """Verify pyproject's ``[project.optional-dependencies]`` match deps.py.

    The extras are the same policy as ``requirements/*.txt`` reached by a
    different route, so they are checked against the same source of truth rather
    than maintained by hand.
    """
    import tomllib

    from aurora.deps import GROUP_PARENT as parents
    from aurora.deps import dependencies_for_group

    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return ["pyproject.toml is missing"]
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data.get("project", {}).get("optional-dependencies", {})
    problems: List[str] = []
    for group, lines in extras.items():
        # A duplicate is invisible to the set comparison below but is a real
        # mistake, and pip would install the entry twice.
        duplicates = sorted({line for line in lines if lines.count(line) > 1})
        if duplicates:
            problems.append(f"pyproject extra '{group}' lists duplicates: {duplicates}")
    for group in ORDERED_GROUPS:
        if group not in extras:
            continue
        declared = sorted({line.strip() for line in extras[group] if line.strip()})
        # Only the installable ones: a source-only dependency has no usable PyPI
        # name, so it must not appear in an extra (see Dependency.installable).
        expected = sorted(
            {dep.requirement_line for dep in dependencies_for_group(group) if dep.installable}
        )
        # `hest`, `pyvips` etc. are not on PyPI; requirements/*.txt notes that in
        # a comment, and pyproject cannot express a comment, so allow the extras
        # to name exactly the same distributions.
        if declared != expected:
            missing = [name for name in expected if name not in declared]
            extra = [name for name in declared if name not in expected]
            detail = []
            if missing:
                detail.append(f"missing {missing}")
            if extra:
                detail.append(f"unexpected {extra}")
            problems.append(f"pyproject extra '{group}': " + "; ".join(detail))
    unknown = sorted(set(extras) - set(parents))
    if unknown:
        problems.append(f"pyproject declares unknown extra(s): {unknown}")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Only verify that the files are up to date")
    args = parser.parse_args(argv)

    REQUIREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    stale: List[str] = []
    written: List[str] = []
    for group in ORDERED_GROUPS:
        relative = GROUP_REQUIREMENTS_FILE.get(group)
        if not relative:
            continue
        path = REPO_ROOT / relative
        content = render(group)
        existing = path.read_text(encoding="utf-8") if path.is_file() else None
        if existing == content:
            continue
        if args.check:
            stale.append(relative)
            continue
        path.write_text(content, encoding="utf-8", newline="\n")
        written.append(relative)

    if args.check:
        failed = False
        if stale:
            print("Out-of-date requirements files:")
            for name in stale:
                print(f"  - {name}")
            failed = True
        extra_problems = check_pyproject_extras()
        if extra_problems:
            print("pyproject.toml extras do not match aurora/deps.py:")
            for problem in extra_problems:
                print(f"  - {problem}")
            failed = True
        if failed:
            print("\nRun: python tools/sync_requirements.py")
            return 1
        print("requirements/*.txt and pyproject.toml match aurora/deps.py")
        return 0

    if written:
        print("Wrote:")
        for name in written:
            print(f"  - {name}")
    else:
        print("requirements/*.txt were already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
