#!/usr/bin/env python
"""Regression test: a flattened payload must be caught before torch is involved.

The first revision of ``tools/merge_aurora_checkpoints.py`` stripped the root
directory that ``torch.save`` writes.  Checkpoints built with it failed at run
time with::

    RuntimeError: [enforce fail at inline_container.cc:174]
    . file in archive is not in a subdirectory: data.pkl

This test builds a combined checkpoint containing exactly that flat layout (no
PyTorch needed -- the container is a zip), and asserts that

* ``aurora.bundle.validate_bundle`` reports it,
* ``CombinedCheckpoint.ensure_payload_loadable`` raises a
  ``CheckpointFormatError`` whose hint contains the rebuild command,
* ``Python -m aurora validate`` exits non-zero, and
* ``aurora run --dry-run`` refuses before running any stage.

The "good" layout is checked alongside it so the test cannot pass by rejecting
everything.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aurora.bundle import (  # noqa: E402
    MANIFEST_ENTRY,
    CombinedCheckpoint,
    classify_checkpoint,
    torch_archive_layout,
    validate_bundle,
)
from aurora.errors import CheckpointFormatError  # noqa: E402

failures: list[str] = []


def expect(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def fake_torch_archive(root: str) -> bytes:
    """Minimal stand-in for torch.save output; ``root=''`` gives a flat archive."""
    prefix = f"{root}/" if root else ""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{prefix}data.pkl", b"not-a-real-pickle")
        archive.writestr(f"{prefix}data/0", os.urandom(64))
        archive.writestr(f"{prefix}version", b"3\n")
        archive.writestr(f"{prefix}byteorder", b"little")
        archive.writestr(f"{prefix}.data/serialization_id", b"0" * 40)
    return buffer.getvalue()


def build_bundle(path: Path, root: str) -> Path:
    payload = fake_torch_archive(root)
    import hashlib

    manifest = {
        "format": "aurora.combined",
        "format_version": 1,
        "payload_kind": "torch_archive",
        "model_name": "AURORA_TEST",
        "cancer_type": "TEST",
        "stages": {
            "lowres": {
                "payload": "payload/lowres.pt",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "source_file": "AURORA_TEST_lowres.pth",
                "archive_root": f"{root}/" if root else "",
                "patch_sizes": [224],
                "img_emb_dim": 8,
                "params": {"img_emb_dim": 8, "patch_size": [224]},
            }
        },
        "assets": {},
        "requires": {},
        "notes": "synthetic, for the layout regression test",
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payload/lowres.pt", payload)
        archive.writestr(MANIFEST_ENTRY, json.dumps(manifest))
    return path


def main() -> int:
    root = REPO_ROOT / f".layout_test_{os.getpid()}"
    root.mkdir()
    try:
        flat = build_bundle(root / "flat.pth", root="")
        good = build_bundle(root / "good.pth", root="AURORA_TEST_lowres")

        print("1. both synthetic bundles are recognised as combined checkpoints")
        expect(classify_checkpoint(flat).is_combined, "the flat bundle is a combined checkpoint")
        expect(classify_checkpoint(good).is_combined, "the rooted bundle is a combined checkpoint")

        print("2. the layout rule separates them")
        with zipfile.ZipFile(io.BytesIO(fake_torch_archive(""))) as archive:
            flat_names = archive.namelist()
        with zipfile.ZipFile(io.BytesIO(fake_torch_archive("AURORA_TEST_lowres"))) as archive:
            good_names = archive.namelist()
        ok, _root, problem = torch_archive_layout(flat_names)
        expect(not ok, f"flat rejected: {problem}")
        ok, detected, problem = torch_archive_layout(good_names)
        expect(ok, f"rooted accepted: {problem}")
        expect(detected == "AURORA_TEST_lowres/", f"root detected as {detected!r}")

        print("3. validate_bundle reports the flat payload")
        flat_report = validate_bundle(flat)
        expect(not flat_report.ok, "the flat bundle fails validation")
        expect(
            any("torch.load" in problem for problem in flat_report.problems),
            f"the problem names torch.load: {flat_report.problems}",
        )
        good_report = validate_bundle(good)
        expect(good_report.ok, f"the rooted bundle passes validation: {good_report.problems}")

        print("4. the loader refuses before importing torch")
        checkpoint = CombinedCheckpoint.open(flat)
        try:
            checkpoint.ensure_payload_loadable("lowres")
        except CheckpointFormatError as exc:
            expect(
                "cannot be loaded by torch.load" in exc.message,
                "CheckpointFormatError explains the problem",
            )
            expect(
                "merge_aurora_checkpoints.py" in (exc.hint or ""),
                "the hint contains the rebuild command",
            )
            expect(
                "AURORA_TEST_lowres.pth" in (exc.hint or ""),
                "the hint names the recorded source file",
            )
        else:
            expect(False, "ensure_payload_loadable should have raised")
        try:
            checkpoint.load_stage_state_dict("lowres")
        except CheckpointFormatError as exc:
            expect(True, "load_stage_state_dict raises CheckpointFormatError, not RuntimeError")
            del exc
        except Exception as exc:  # noqa: BLE001
            expect(False, f"unexpected {type(exc).__name__}: {exc}")

        good_checkpoint = CombinedCheckpoint.open(good)
        expect(
            good_checkpoint.ensure_payload_loadable("lowres") == "AURORA_TEST_lowres/",
            "the rooted payload is accepted and its root returned",
        )

        print("5. the CLI refuses the flat checkpoint")
        validate_cli = subprocess.run(
            [sys.executable, "-B", "-m", "aurora", "validate", "--model", str(flat)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        expect(validate_cli.returncode != 0, "aurora validate exits non-zero")
        expect(
            "torch.load" in (validate_cli.stdout + validate_cli.stderr),
            "aurora validate prints the reason",
        )

        run_cli = subprocess.run(
            [sys.executable, "-B", "-m", "aurora", "run", "--project", str(root),
             "--run", "r", "--samples", str(root / "s.csv"), "--model", str(flat),
             "--all", "--dry-run", "--quiet"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        combined = run_cli.stdout + run_cli.stderr
        expect(run_cli.returncode != 0, "aurora run refuses")
        expect(
            "torch.load cannot read" in combined or "cannot be loaded by torch.load" in combined,
            f"aurora run explains why: {combined.strip().splitlines()[-1][:120]}",
        )
        expect(
            "merge_aurora_checkpoints.py" in combined,
            "aurora run prints the rebuild command",
        )
        print()
        print(f"    {(run_cli.stderr or combined).strip().splitlines()[0][:150]}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("payload layout regression test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
