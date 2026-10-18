#!/usr/bin/env python
"""Tests for `aurora clean`, which reclaims a run's intermediate files.

Cleanup is destructive, so the tests are mostly about what it must NOT do:

* predictions, figures, masks, resolved inputs, logs and run state are never
  candidates -- including under ``--deep``, which has no option to reach them;
* a run that has produced no prediction is refused, because at that point the
  intermediates are the only thing that exists;
* nothing outside the run directory is touched;
* the reported size matches what is actually freed.

    python tests/test_cleanup.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aurora.cleanup import (  # noqa: E402
    TIER_RECOMPUTABLE,
    TIER_SCRATCH,
    apply_cleanup,
    human_bytes,
    plan_cleanup,
)
from aurora.errors import ConfigError  # noqa: E402
from aurora.paths import RunLayout  # noqa: E402

failures: list[str] = []
SAMPLE = "TCGA-44-5645-01B-04-BS4"


def expect(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_run(root: Path, with_results: bool = True) -> RunLayout:
    """A run tree shaped like a real one, every directory populated."""
    layout = RunLayout.from_root(root, root.parent.parent)
    layout.ensure()

    # results and things that must survive
    if with_results:
        write(layout.prediction_output_dir("lowres") / f"{SAMPLE}-pred.pickle", b"L" * 4000)
        write(layout.prediction_output_dir("highres") / f"{SAMPLE}-highres-pred.h5", b"H" * 8000)
    write(layout.plots_dir / SAMPLE / f"{SAMPLE}-IL7R-224pixels.png", b"P" * 1000)
    write(layout.masks_dir / f"{SAMPLE}.geojson", b"G" * 500)
    write(layout.masks_dir / f"{SAMPLE}.png", b"g" * 200)
    write(layout.segmentation_dir / "process_list_autogen.csv", b"csv" * 50)
    write(layout.config_dir / "samples.csv", b"Image_name,Sample_name\n" * 20)
    write(layout.logs_dir / "segment.log", b"log" * 100)
    write(layout.run_json, json.dumps({"stages": {}}).encode())
    write(layout.state_dir / "layout.json", b"{}")
    write(layout.checkpoints_dir / "gene_list.csv", b"gene\n" * 10)
    write(layout.prediction_dir("lowres") / "pred" / "keep-me.txt", b"k" * 10)

    # scratch tier
    write(layout.tmp_dir / "legacy_pred_view" / "pred" / f"{SAMPLE}-pred.pickle", b"T" * 3000)
    write(layout.prediction_cache_dir("lowres") / f"{SAMPLE}.pkl", b"C" * 5000)
    write(layout.prediction_cache_dir("highres") / f"{SAMPLE}_superpatch_0.pkl", b"D" * 6000)
    for sub in ("contours", "patches", "viz", "stitches"):
        write(layout.segmentation_dir / sub / "x.bin", b"S" * 700)
    write(layout.check_figure_dir / "check.png", b"F" * 800)
    write(layout.feature_bundle_dir / "strip_tmp" / "chunk.tif", b"Z" * 900)
    write(layout.feature_bundle_dir / "HistoSweep_tmp" / "native.png", b"N" * 1000)
    write(layout.feature_bundle_dir / "emb_56" / f"{SAMPLE}_raw.npz", b"R" * 2000)

    # recomputable tier
    write(layout.rescaled_dir / f"{SAMPLE}_rescaled.tif", b"R" * 20000)
    write(layout.rescaled_dir / f"{SAMPLE}_mask_rescaled.tif", b"M" * 10000)
    write(layout.rescaled_dir / f"{SAMPLE}_HistoSweep_mask_rescaled.tif", b"W" * 12000)
    for patch_size in (3584, 896, 224):
        write(layout.embeddings_dir(patch_size) / f"{SAMPLE}.npz", b"E" * 15000)
    write(layout.histosweep_dir / "native" / "mask.png", b"h" * 300)
    return layout


def main() -> int:
    root = REPO_ROOT / f".cleanup_{os.getpid()}"
    project = root / "debug"
    try:
        layout = build_run(project / "aurora_runs" / "luainfer")
        outside = project / "Inference_samples.csv"
        write(outside, b"outside" * 100)
        before_outside = outside.read_bytes()

        # ---------------------------------------------------------------- 1
        print("1. the report separates scratch from recomputable")
        plan = plan_cleanup(layout)
        deep_plan = plan_cleanup(layout, deep=True)
        scratch = {item.path.name for item in plan.by_tier(TIER_SCRATCH)}
        # The recomputable tier is NOT in the shallow plan; it is only listed as
        # deferred, which is what makes `--apply` safe.
        recomputable = {item.path.name for item in plan.deferred}
        expect(
            recomputable == {item.path.name for item in deep_plan.by_tier(TIER_RECOMPUTABLE)},
            "the deferred set is exactly the deep plan's recomputable tier",
        )
        expect(
            not plan.by_tier(TIER_RECOMPUTABLE),
            "the shallow plan holds no recomputable item at all",
        )
        expect("tmp" in scratch, f"tmp/ is scratch: {sorted(scratch)}")
        expect("processed_pkl" in scratch, "processed_pkl/ is scratch")
        expect("processed_pkl_test_highres" in scratch, "the high-res cache is scratch")
        expect({"contours", "patches", "viz", "stitches"} <= scratch, "segmentation scratch")
        expect("check_figure" in scratch, "check_figure/ is scratch")
        expect("strip_tmp" in scratch, "strip_tmp/ is scratch")
        expect(
            any(name.endswith("_raw.npz") for name in scratch),
            f"_raw.npz is scratch: {sorted(scratch)}",
        )
        expect("wsis_rescaled" in recomputable, "rescaled slides are recomputable, not scratch")
        expect(
            {"emb_3584", "emb_896", "emb_224"} <= recomputable,
            f"embeddings are recomputable: {sorted(recomputable)}",
        )
        shallow_total = plan.total_bytes
        expect(plan.has_results, "the run is seen as having produced results")
        expect(sorted(p.name for p in plan.results) == [
            f"{SAMPLE}-highres-pred.h5", f"{SAMPLE}-pred.pickle"
        ], f"results found: {[p.name for p in plan.results]}")

        print()
        print("2. the shallow plan excludes everything recomputable")
        expect(
            deep_plan.total_bytes > shallow_total,
            f"--deep is bigger ({human_bytes(deep_plan.total_bytes)} vs "
            f"{human_bytes(shallow_total)})",
        )
        expect(
            not any(item.tier == TIER_RECOMPUTABLE for item in plan.items),
            "the shallow plan contains no recomputable item",
        )
        expect(
            "code/plot.py" in deep_plan.describe(),
            "the report explains that plotting needs the rescaled slides",
        )

        # ---------------------------------------------------------------- 3
        print("3. nothing that is a result is ever a candidate")
        for candidate_plan in (plan, deep_plan):
            paths = {str(item.path) for item in candidate_plan.items}
            for protected in (
                layout.prediction_output_dir("lowres"),
                layout.prediction_output_dir("highres"),
                layout.plots_dir,
                layout.masks_dir,
                layout.config_dir,
                layout.logs_dir,
                layout.state_dir,
                layout.checkpoints_dir,
                layout.run_json,
            ):
                expect(
                    not any(p == str(protected) or p.startswith(str(protected) + os.sep) for p in paths),
                    f"{protected.name} is not a candidate",
                )

        # ---------------------------------------------------------------- 4
        print("4. applying the shallow plan keeps the results and the inputs")
        freed = apply_cleanup(plan)
        expect(freed == shallow_total, f"freed {human_bytes(freed)} == reported {human_bytes(shallow_total)}")
        for survivor in (
            layout.prediction_output_dir("lowres") / f"{SAMPLE}-pred.pickle",
            layout.prediction_output_dir("highres") / f"{SAMPLE}-highres-pred.h5",
            layout.prediction_dir("lowres") / "pred" / "keep-me.txt",
            layout.plots_dir / SAMPLE / f"{SAMPLE}-IL7R-224pixels.png",
            layout.masks_dir / f"{SAMPLE}.geojson",
            layout.config_dir / "samples.csv",
            layout.logs_dir / "segment.log",
            layout.run_json,
            layout.checkpoints_dir / "gene_list.csv",
        ):
            expect(survivor.exists(), f"kept {survivor.relative_to(layout.root).as_posix()}")
        for removed in (
            layout.tmp_dir,
            layout.prediction_cache_dir("lowres"),
            layout.prediction_cache_dir("highres"),
            layout.segmentation_dir / "contours",
            layout.segmentation_dir / "stitches",
            layout.check_figure_dir,
            layout.feature_bundle_dir / "strip_tmp",
            layout.feature_bundle_dir / "emb_56" / f"{SAMPLE}_raw.npz",
        ):
            expect(not removed.exists(), f"removed {removed.name}")
        expect(outside.read_bytes() == before_outside, "a file outside the run was untouched")

        # ---------------------------------------------------------------- 5
        print("5. --deep then removes the embeddings and rescaled slides")
        deep_plan = plan_cleanup(layout, deep=True)
        freed_deep = apply_cleanup(deep_plan)
        expect(freed_deep > 0, f"--deep freed {human_bytes(freed_deep)}")
        expect(not layout.rescaled_dir.exists() or not any(layout.rescaled_dir.iterdir()),
               "rescaled slides are gone")
        expect(
            not any(layout.feature_bundle_dir.glob("emb_*")) or
            all(not any(d.iterdir()) for d in layout.feature_bundle_dir.glob("emb_*")),
            "embeddings are gone",
        )
        expect(
            (layout.prediction_output_dir("lowres") / f"{SAMPLE}-pred.pickle").exists(),
            "the predictions survived the deep clean",
        )

        # ---------------------------------------------------------------- 6
        print("6. a run with no results is refused")
        bare = build_run(project / "aurora_runs" / "unfinished", with_results=False)
        bare_plan = plan_cleanup(bare, deep=True)
        expect(not bare_plan.has_results, "the unfinished run has no results")
        expect(
            "produced no prediction" in bare_plan.describe(),
            "the report says so plainly",
        )
        try:
            apply_cleanup(bare_plan)
        except ConfigError as exc:
            expect("no prediction" in str(exc), f"refused: {str(exc)[:90]}")
            expect("--force" in (exc.hint or ""), "the hint offers --force")
        else:
            expect(False, "a run with no results should be refused")
        expect(bare.rescaled_dir.is_dir(), "nothing was removed by the refusal")
        freed_forced = apply_cleanup(bare_plan, force=True)
        expect(freed_forced > 0, f"--force cleaned it anyway ({human_bytes(freed_forced)})")

        # ---------------------------------------------------------------- 7
        print("7. the CLI reports by default and deletes on --apply")
        cli_run = build_run(project / "aurora_runs" / "clirun")
        report = subprocess.run(
            [sys.executable, "-B", "-m", "aurora", "clean",
             "--project", str(project), "--run", "clirun"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        expect(report.returncode == 0, (report.stdout + report.stderr)[-800:])
        expect("report only" in report.stdout, "it says it is a report")
        expect(cli_run.tmp_dir.exists(), "nothing was deleted without --apply")

        applied = subprocess.run(
            [sys.executable, "-B", "-m", "aurora", "clean",
             "--project", str(project), "--run", "clirun", "--apply"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        expect(applied.returncode == 0, (applied.stdout + applied.stderr)[-800:])
        expect("removed" in applied.stdout and "Total freed" in applied.stdout,
               f"it reports what it freed: {applied.stdout.strip().splitlines()[-1]}")
        expect(not cli_run.tmp_dir.exists(), "tmp/ was removed by --apply")
        expect(
            (cli_run.prediction_output_dir("lowres") / f"{SAMPLE}-pred.pickle").exists(),
            "the prediction survived the CLI clean",
        )

        all_runs = subprocess.run(
            [sys.executable, "-B", "-m", "aurora", "clean",
             "--project", str(project), "--all-runs", "--apply"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        expect(all_runs.returncode == 0, (all_runs.stdout + all_runs.stderr)[-800:])
        expect("Cleanup plan for" in all_runs.stdout, "--all-runs visits every run")

        # ---------------------------------------------------------------- 8
        print("8. a missing run is a clear error, not a traceback")
        missing = subprocess.run(
            [sys.executable, "-B", "-m", "aurora", "clean",
             "--project", str(project), "--run", "nope"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        expect(missing.returncode != 0, "a missing run exits non-zero")
        combined = missing.stdout + missing.stderr
        expect("does not exist" in combined, f"and says why: {combined[-200:]}")
        expect("--all-runs" in combined, "and suggests --all-runs")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("cleanup test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
