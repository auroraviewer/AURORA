"""Focused checks for the Sample_name / Image_name naming fix and the sheet columns."""

from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aurora.naming import (  # noqa: E402
    naming_for,
    reconcile_segmentation_names,
    resolve_image,
    resolve_sheet_images,
    segmentation_output_groups,
)
from aurora.paths import RunLayout  # noqa: E402
from aurora.samples import read_sample_sheet, write_sample_sheet  # noqa: E402

# A realistic TCGA row: the GDC file name carries a UUID, the sample barcode does not.
IMAGE = "TCGA-44-5645-01B-04-BS4.76ED5128-FB3D-44C7-A679-848860CAA3C5.svs"
SAMPLE = "TCGA-44-5645-01B-04-BS4"
IMAGE_STEM = "TCGA-44-5645-01B-04-BS4.76ED5128-FB3D-44C7-A679-848860CAA3C5"

failures = []


def expect(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def build(root: Path, file_names) -> Path:
    project = root / "project"
    (project / "wsis").mkdir(parents=True)
    for name in file_names:
        (project / "wsis" / name).write_text("dummy", encoding="ascii")
    with open(project / "samples.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Image_name", "Sample_name", "Bulk_sample_name"])
        writer.writerow([IMAGE, SAMPLE, "BULK-1"])
    return project


def main() -> int:
    root = REPO_ROOT / f".naming_test_{os.getpid()}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()

    try:
        # ---------------------------------------------------------------- 1
        print("1. the two spellings are recognised as different")
        project = build(root, [IMAGE])
        sheet = read_sample_sheet(project / "samples.csv")
        naming = naming_for(sheet.samples[0])
        expect(naming.sample_name == SAMPLE, f"sample name resolved to {naming.sample_name!r}")
        expect(naming.image_stem == IMAGE_STEM, f"image stem resolved to {naming.image_stem!r}")
        expect(naming.names_differ, "the two spellings are detected as different")

        # ---------------------------------------------------------------- 2
        print("2. declared output is keyed by Sample_name, either spelling accepted")
        layout = RunLayout.create(project, "run")
        layout.ensure()
        groups = segmentation_output_groups(layout, sheet)
        expect(len(groups) == 1 and len(groups[0]) == 2, f"two acceptable paths: {groups[0]}")
        expect(groups[0][0].name == f"{SAMPLE}.geojson", f"canonical is {groups[0][0].name}")
        expect(groups[0][1].name == f"{IMAGE_STEM}.geojson", f"alias is {groups[0][1].name}")

        # ---------------------------------------------------------------- 3
        print("3. a mask written under the image name is linked to the sample name")
        (layout.masks_dir / f"{IMAGE_STEM}.geojson").write_text('{"type":"FeatureCollection"}')
        notes = reconcile_segmentation_names(layout, sheet)
        canonical = layout.masks_dir / f"{SAMPLE}.geojson"
        alias = layout.masks_dir / f"{IMAGE_STEM}.geojson"
        expect(canonical.is_file(), f"{canonical.name} was created")
        expect(alias.is_file(), f"{alias.name} still exists")
        expect(notes, "a note was reported")
        if hasattr(os, "stat") and os.name == "nt" or True:
            try:
                expect(os.stat(canonical).st_ino == os.stat(alias).st_ino, "the two names share one inode")
            except OSError:
                pass

        # ---------------------------------------------------------------- 4
        print("4. a mask written under the sample name is linked to the image name (for rescale.py)")
        fresh = root / "project2"
        (fresh / "wsis").mkdir(parents=True)
        (fresh / "wsis" / f"{SAMPLE}.svs").write_text("dummy", encoding="ascii")
        with open(fresh / "samples.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Image_name", "Sample_name", "Bulk_sample_name"])
            writer.writerow([IMAGE, SAMPLE, "BULK-1"])
        sheet2 = read_sample_sheet(fresh / "samples.csv")
        layout2 = RunLayout.create(fresh, "run")
        layout2.ensure()
        (layout2.masks_dir / f"{SAMPLE}.geojson").write_text("{}")
        reconcile_segmentation_names(layout2, sheet2)
        expect(
            (layout2.masks_dir / f"{IMAGE_STEM}.geojson").is_file(),
            "the image-named alias was created for code/rescale.py",
        )

        # ---------------------------------------------------------------- 5
        print("5. the slide is found even when Image_name is not the file on disk")
        resolution = resolve_image(fresh / "wsis", sheet2.samples[0])
        expect(resolution.found, "the slide was found")
        expect(resolution.resolved_name == f"{SAMPLE}.svs", f"resolved to {resolution.resolved_name!r}")
        resolved_sheet, resolutions, notes = resolve_sheet_images(sheet2, fresh / "wsis")
        expect(
            resolved_sheet.samples[0].image_name == f"{SAMPLE}.svs",
            f"the resolved sheet says {resolved_sheet.samples[0].image_name!r}",
        )
        expect(resolved_sheet.samples[0].sample_name == SAMPLE, "Sample_name is unchanged")
        expect(bool(notes), "the substitution is reported as a note")

        # ---------------------------------------------------------------- 6
        print("6. an unfindable slide reports the paths it tried")
        empty = root / "project3"
        (empty / "wsis").mkdir(parents=True)
        with open(empty / "samples.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Image_name", "Sample_name", "Bulk_sample_name"])
            writer.writerow([IMAGE, SAMPLE, "BULK-1"])
        sheet3 = read_sample_sheet(empty / "samples.csv")
        missing = resolve_image(empty / "wsis", sheet3.samples[0])
        expect(not missing.found, "the slide is reported as missing")
        expect(len(missing.tried) >= 2, f"{len(missing.tried)} candidate paths were tried")

        # ---------------------------------------------------------------- 7
        print("7. identical names produce a single declared output")
        plain = root / "project4"
        (plain / "wsis").mkdir(parents=True)
        (plain / "wsis" / "S1.svs").write_text("dummy", encoding="ascii")
        with open(plain / "samples.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Image_name", "Sample_name", "Bulk_sample_name"])
            writer.writerow(["S1.svs", "S1", "B1"])
        sheet4 = read_sample_sheet(plain / "samples.csv")
        layout4 = RunLayout.create(plain, "run")
        layout4.ensure()
        groups4 = segmentation_output_groups(layout4, sheet4)
        expect(len(groups4[0]) == 1, f"one acceptable path when the names agree: {groups4[0]}")
        expect(reconcile_segmentation_names(layout4, sheet4) == [], "nothing to reconcile")

        # ---------------------------------------------------------------- 8
        print("8. an all-empty optional column is not written at all")
        # code/rescale.py picks the per-slide-folder layout from the *presence* of
        # the Image_folder column, and pandas turns a blank field into NaN, so an
        # empty column used to produce <image_path>/nan/<file>.
        blank = root / "project5"
        (blank / "wsis").mkdir(parents=True)
        (blank / "wsis" / "S1.svs").write_text("dummy", encoding="ascii")
        with open(blank / "samples.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Image_name", "Sample_name", "Bulk_sample_name", "Image_folder", "Pixel_size_um"])
            writer.writerow(["S1.svs", "S1", "", "", ""])
        sheet5 = read_sample_sheet(blank / "samples.csv")
        out5 = root / "out5.csv"
        write_sample_sheet(sheet5, out5)
        header5 = out5.read_text(encoding="utf-8").splitlines()[0]
        expect(header5 == "Image_name,Sample_name", f"columns written: {header5!r}")
        expect("Image_folder" not in header5, "the empty Image_folder column was dropped")
        expect("Pixel_size_um" not in header5, "the empty Pixel_size_um column was dropped")
        expect("Bulk_sample_name" not in header5, "the empty Bulk_sample_name column was dropped")

        # ---------------------------------------------------------------- 9
        print("9. a partially filled Image_folder is completed with '.' (never 'nan')")
        mixed = root / "project6"
        (mixed / "wsis").mkdir(parents=True)
        with open(mixed / "samples.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Image_name", "Sample_name", "Image_folder"])
            writer.writerow(["A.svs", "A", "CASE-1"])
            writer.writerow(["B.svs", "B", ""])
        sheet6 = read_sample_sheet(mixed / "samples.csv")
        out6 = root / "out6.csv"
        write_sample_sheet(sheet6, out6)
        rows6 = list(csv.DictReader(out6.read_text(encoding="utf-8").splitlines()))
        expect(rows6[0]["Image_folder"] == "CASE-1", f"folder kept: {rows6[0]['Image_folder']!r}")
        expect(rows6[1]["Image_folder"] == ".", f"blank completed with '.': {rows6[1]['Image_folder']!r}")
        # Reproduce what code/rescale.py does with the value.
        import os as _os

        joined = _os.path.join("/run", f"/data/wsis/{rows6[1]['Image_folder']}/", "B.svs")
        expect("nan" not in joined, f"rescale.py would build {joined!r}")
        expect(_os.path.normpath(joined) == _os.path.normpath("/data/wsis/B.svs"),
               f"and it normalises to /data/wsis/B.svs, got {_os.path.normpath(joined)!r}")

        # --------------------------------------------------------------- 10
        print("10. a partially filled Bulk_sample_name falls back to Sample_name")
        bulkpart = root / "project7"
        (bulkpart / "wsis").mkdir(parents=True)
        with open(bulkpart / "samples.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Image_name", "Sample_name", "Bulk_sample_name"])
            writer.writerow(["A.svs", "A", "BULK-A"])
            writer.writerow(["B.svs", "B", ""])
        sheet7 = read_sample_sheet(bulkpart / "samples.csv")
        out7 = root / "out7.csv"
        write_sample_sheet(sheet7, out7)
        rows7 = list(csv.DictReader(out7.read_text(encoding="utf-8").splitlines()))
        expect(rows7[1]["Bulk_sample_name"] == "B", f"fell back to {rows7[1]['Bulk_sample_name']!r}")

        # --------------------------------------------------------------- 12
        print("12. the high-resolution filename follows both output flags")
        from aurora.naming import (
            highres_output_kind,
            highres_output_path,
            lowres_output_path,
        )

        combinations = [
            (False, False, "pickle", "-highres-pred.pickle"),
            (False, True, "pickle", "-highres-pred.pickle"),
            (True, False, "h5", "-highres-pred.h5"),
            (True, True, "zst", "-highres-pred.npz.zst"),
        ]
        for uint8, compress, kind, suffix in combinations:
            got_kind = highres_output_kind(uint8, compress)
            got_name = highres_output_path(root, "S1", uint8, compress).name
            expect(
                got_kind == kind,
                f"uint8={uint8} compress={compress} -> kind {got_kind!r}",
            )
            expect(
                got_name == f"S1{suffix}",
                f"uint8={uint8} compress={compress} -> {got_name!r}",
            )
        expect(
            lowres_output_path(root, "S1").name == "S1-pred.pickle",
            "the low-resolution name does not depend on the flags",
        )

        # The stage's declared output must be the same file the writer produces.
        from aurora.spec import StageContext, get_stage
        from aurora.config import RunConfig

        for uint8, compress, _kind, suffix in combinations:
            cfg = RunConfig(
                project_path=root,
                run_name="flags",
                uint8_output=uint8,
                prediction_compression=compress,
            )
            ctx = StageContext(
                layout=RunLayout.create(root, "flags"),
                samples=sheet4,
                options=cfg.stage_options(),
                config=cfg,
            )
            declared = get_stage("predict_highres").outputs(ctx)
            expect(
                [p.name for p in declared] == [f"S1{suffix}"],
                f"declared output for uint8={uint8} compress={compress}: "
                f"{[p.name for p in declared]}",
            )

        # --------------------------------------------------------------- 13
        print("13. the high-resolution writer does not hard-code its filenames")
        writer = (REPO_ROOT / "aurora" / "inference" / "predict_highres.py").read_text(
            encoding="utf-8"
        )
        offenders = [
            line.strip()
            for line in writer.splitlines()
            if "highres-pred" in line and not line.strip().startswith("#")
        ]
        expect(
            not offenders,
            f"predict_highres.py must build its path with highres_output_path(); found {offenders}",
        )
        expect(
            "highres_output_path(" in writer,
            "predict_highres.py uses the shared naming helper",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("all naming checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
