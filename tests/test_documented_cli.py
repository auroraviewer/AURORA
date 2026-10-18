#!/usr/bin/env python
"""Check that every documented ``aurora`` invocation actually parses.

The ``--gene IL7R KRT8`` failure happened because the README, ``run.sh`` and the
parser disagreed: the docs showed a space-separated list, the parser accepted one
value per flag.  Documentation drift like that is only visible when someone
copies the command, so this test copies them all.

It extracts every ``python -m aurora …`` command from

* fenced code blocks in ``README.md``,
* ``run.sh``,

joins shell line continuations, tokenises with ``shlex``, and feeds the arguments
to the real parser.  Nothing is executed -- only argument parsing is checked, so
it runs with no dependencies and no model.

    python tests/test_documented_cli.py
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aurora.cli import build_parser  # noqa: E402

failures: List[str] = []


def expect(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


SHELL_LANGUAGES = {"bash", "sh", "shell", "console", "zsh"}


def fenced_blocks(text: str) -> List[Tuple[str, str]]:
    """``(language, body)`` for every ```-fenced block.

    Fences inside a blockquote -- the README wraps its examples in a
    ``> [!TIP]`` callout so they are visually marked -- are recognised too, with the
    ``> `` prefix stripped, so those commands are still checked.
    """
    blocks: List[Tuple[str, str]] = []
    current: List[str] = []
    language = ""
    inside = False
    for raw in text.splitlines():
        line = raw
        stripped = line.lstrip()
        # Strip one level of blockquote so "> ```bash" and "> cmd" are seen.
        quoted = stripped.startswith(">")
        if quoted:
            line = stripped[1:]
            if line.startswith(" "):
                line = line[1:]
            stripped = line.lstrip()
        if stripped.startswith("```"):
            if inside:
                blocks.append((language, "\n".join(current)))
                current = []
                language = ""
            else:
                language = stripped[3:].strip().lower()
            inside = not inside
            continue
        if inside:
            current.append(line)
    if current:
        blocks.append((language, "\n".join(current)))
    return blocks


def aurora_commands(text: str, *, blocks_only: bool = False) -> List[Tuple[int, str]]:
    """Return ``(line number, command)`` for every ``python [-u] -m aurora …``.

    With ``blocks_only`` only fenced blocks declared as a shell language are
    scanned, so a documentation *listing* (``​```text``) is not mistaken for a
    runnable command.
    """
    if blocks_only:
        sources = [body for language, body in fenced_blocks(text) if language in SHELL_LANGUAGES]
    else:
        sources = [text]
    found: List[Tuple[int, str]] = []
    for source in sources:
        lines = source.splitlines()
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped.startswith(("python -m aurora", "python -u -m aurora", "python3 -m aurora")):
                command = stripped
                start = index + 1
                # join shell line continuations
                while command.rstrip().endswith("\\") and index + 1 < len(lines):
                    command = command.rstrip()[:-1] + " " + lines[index + 1].strip()
                    index += 1
                found.append((start, command))
            index += 1
    return found


def check_file(path: Path, *, blocks_only: bool) -> int:
    text = path.read_text(encoding="utf-8")
    commands = aurora_commands(text, blocks_only=blocks_only)
    print(f"{path.relative_to(REPO_ROOT)}: {len(commands)} documented command(s)")
    parser = build_parser()
    checked = 0
    for line_number, command in commands:
        try:
            # comments=True drops a trailing '# ...' the way a shell would.
            tokens = shlex.split(command, comments=True)
        except ValueError as exc:
            expect(False, f"{path.name}:{line_number} cannot be tokenised ({exc}): {command}")
            continue
        # Drop the interpreter and the module selector.
        while tokens and tokens[0] in ("python", "python3", "-u", "-B"):
            tokens.pop(0)
        if tokens[:2] != ["-m", "aurora"]:
            continue
        argv = tokens[2:]
        if not argv:
            continue
        checked += 1
        try:
            parser.parse_args(argv)
        except SystemExit:
            expect(
                False,
                f"{path.name}:{line_number} does not parse: {command}\n"
                f"        argv = {argv}",
            )
        else:
            expect(True, f"{path.name}:{line_number}  {command[:96]}")
    return checked


def main() -> int:
    total = 0
    total += check_file(REPO_ROOT / "README.md", blocks_only=True)
    run_sh = REPO_ROOT / "run.sh"
    if run_sh.is_file():
        total += check_file(run_sh, blocks_only=False)

    # The exact command from the bug report, and the two spellings of each
    # repeatable option, must all parse.
    print()
    print("explicit spellings")
    parser = build_parser()
    cases = [
        ["run", "--run", "R", "--stage", "plot", "--gene", "IL7R", "KRT8",
         "--celltype", "Malignant", "T_cell"],
        ["run", "--run", "R", "--stage", "plot", "--gene", "IL7R", "--gene", "KRT8"],
        ["run", "--run", "R", "--stage", "plot", "--celltype", "Malignant"],
        ["run", "--run", "R", "--stage", "segment", "rescale", "predict_lowres"],
        ["run", "--run", "R", "--stage", "segment", "--stage", "rescale"],
        ["run", "--run", "R", "--sample", "A", "B"],
        ["run", "--run", "R", "--sample", "A", "--sample", "B"],
        ["run", "--run", "R"],
    ]
    for argv in cases:
        try:
            parser.parse_args(argv)
        except SystemExit:
            expect(False, f"{argv} does not parse")
        else:
            expect(True, " ".join(argv))
    total += len(cases)

    # Parsing is only half the story: the values must reach the stage command.
    print()
    print("parsed values reach the stage command")
    from aurora.commands import build_commands
    from aurora.config import RunConfig
    from aurora.paths import RunLayout
    from aurora.samples import Sample, SampleSheet
    from aurora.spec import StageContext

    namespace = parser.parse_args(cases[0])
    root = REPO_ROOT / f".cli_test_{os.getpid()}"
    project = root / "project"
    (project / "wsis").mkdir(parents=True)
    (project / "wsis" / "S1.svs").write_text("x", encoding="ascii")
    try:
        config = RunConfig(
            project_path=project,
            run_name="cli",
            mode="inference",
            samples_path=None,
            gene_names=list(namespace.gene),
            cell_types=list(namespace.celltype),
            plot_resolution=namespace.plot_resolution,
            # The plot stage projects the existing predictions into a legacy view;
            # a dry run reports that without needing them to exist yet.  The real
            # path is covered by tests/test_prediction_layout.py.
            dry_run=True,
        )
        layout = RunLayout.create(project, "cli")
        layout.ensure()
        sheet = SampleSheet(
            path=None,
            samples=[Sample(sample_name="S1", image_name="S1.svs")],
            columns=["Image_name", "Sample_name"],
        )
        ctx = StageContext(
            layout=layout, samples=sheet, options=config.stage_options(), config=config
        )
        argv = build_commands("plot", ctx)[0].argv
        rendered = " ".join(str(a) for a in argv)
        expect("--genes IL7R KRT8" in rendered, f"--genes IL7R KRT8 in the plot command: {rendered}")
        expect(
            "--celltypes Malignant T_cell" in rendered,
            f"--celltypes Malignant T_cell in the plot command: {rendered}",
        )
        total += 2
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"{total} documented/spelled invocations all parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
