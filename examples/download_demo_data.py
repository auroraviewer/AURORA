#!/usr/bin/env python3
"""Download the demo data used by the runnable examples in the README.

Everything lands under ``examples/demo/`` with fixed names, so the commands in the
README work as written. Standard library only; downloads resume where possible.

    python examples/download_demo_data.py            # everything (~3.6 GB)
    python examples/download_demo_data.py --what inference
    python examples/download_demo_data.py --what training

Sources:

* the TCGA slide comes from the Genomic Data Commons (open access, no login);
* the two Visium samples are the public 10x Genomics breast-cancer datasets
  (Block A, sections 1 and 2), which are the raw data AURORA was trained on.

UNI2-h is not downloaded here because it lives behind a Hugging Face repository and
needs git; the script prints the one-line command for it at the end.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE / "demo"

#: The published LUAD weights. This file is the *low-resolution* model only: AURORA
#: inference needs a combined checkpoint, so it is merged with the high-resolution
#: weights that ship in ``trained_model/`` (see :func:`build_checkpoint`).
MODEL_URL = (
    "https://huggingface.co/datasets/AURORAData/AURORA_prediction/resolve/main/"
    "trained_models/LUAD/AURORA_LUAD_V2.1.pth"
)
MODEL_NAME = "AURORA_LUAD_V2.1.pth"
MODEL_SIZE = 13_854_757

#: Files in ``trained_model/`` used to turn it into a combined checkpoint.
HIGHRES_WEIGHTS = "AURORA_highres_model_weights.pth"
COMBINED_NAME = "aurora_combined.pth"
#: TCGA-44-5645-01B-04-BS4, a LUAD tissue slide, via the GDC file endpoint.
SLIDE_URL = "https://api.gdc.cancer.gov/data/21a90851-e349-4fc3-bdaa-cc3979529e45"
SLIDE_NAME = "TCGA-44-5645-01B-04-BS4.svs"
#: Exact sizes of the published files. They make a re-run a no-op instead of a
#: second download, which matters when the whole set is several gigabytes.
SLIDE_SIZE = 122_648_105

#: Visium samples, as published by 10x Genomics.
TENX_ROOT = "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0"
VISIUM = {
    "BRCA1": "V1_Breast_Cancer_Block_A_Section_1",
    "BRCA2": "V1_Breast_Cancer_Block_A_Section_2",
}
VISIUM_FILES = (
    ("_image.tif", 1_763_006_484),
    ("_filtered_feature_bc_matrix.h5", 28_201_028),
    ("_spatial.tar.gz", 9_960_960),
)
#: Section 2 was published with slightly different sizes.
VISIUM_FILES_BY_SAMPLE = {
    "V1_Breast_Cancer_Block_A_Section_1": VISIUM_FILES,
    "V1_Breast_Cancer_Block_A_Section_2": (
        ("_image.tif", 1_763_006_484),
        ("_filtered_feature_bc_matrix.h5", 28_898_028),
        ("_spatial.tar.gz", 10_345_453),
    ),
}


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def remote_size(url: str) -> int:
    """The size the server reports for ``url``, or 0 when it will not say.

    Sizes are read rather than hardcoded: the CDNs round them, and a wrong guess would
    either re-download a finished file or accept a truncated one.
    """
    headers = {"User-Agent": "aurora-demo/1.0"}
    try:
        request = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            if length:
                return int(length)
    except Exception:  # noqa: BLE001 - fall through to a ranged GET
        pass
    try:
        request = urllib.request.Request(
            url, headers={**headers, "Range": "bytes=0-0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content_range = response.headers.get("Content-Range") or ""
            if "/" in content_range:
                return int(content_range.rsplit("/", 1)[1])
    except Exception:  # noqa: BLE001
        pass
    return 0


def download(url: str, destination: Path, expected: int = 0) -> None:
    """Fetch ``url`` to ``destination``, resuming a partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = expected or remote_size(url)
    if destination.is_file() and size and destination.stat().st_size >= size:
        print(f"  have    {destination.relative_to(HERE)} ({human(destination.stat().st_size)})")
        return

    have = destination.stat().st_size if destination.is_file() else 0
    if have and size:
        print(f"  resume  {destination.name} at {human(have)}")
    elif have and not size:
        # Size unknown and the file exists: keep it rather than risk a re-download.
        print(f"  have    {destination.relative_to(HERE)} ({human(have)})")
        return
    elif have:
        destination.unlink()
        have = 0

    request = urllib.request.Request(url, headers={"User-Agent": "aurora-demo/1.0"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    mode = "ab" if have else "wb"
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = response.headers.get("Content-Length")
            total = have + int(total) if total else size
            print(f"  get     {destination.name}  ({human(total) if total else '?'})")
            with open(destination, mode) as handle:
                next_report = 10
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    have += len(chunk)
                    if total:
                        percent = 100 * have / total
                        if percent >= next_report:
                            print(
                                f"          {percent:5.1f}%  {human(have)} / {human(total)}",
                                flush=True,
                            )
                            next_report += 10
            print()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"  failed  {url}\n          HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"  failed  {url}\n          {exc.reason}") from exc


def download_slide() -> Path:
    target = DEMO / "wsis" / SLIDE_NAME
    print(f"\nTCGA slide → {target.relative_to(HERE)}")
    download(SLIDE_URL, target, SLIDE_SIZE)
    return target


def download_visium() -> dict:
    written = {}
    for sample, tenx_id in VISIUM.items():
        folder = DEMO / "raw_data" / sample
        print(f"\nVisium {sample} ({tenx_id}) → {folder.relative_to(HERE)}")
        for suffix, size in VISIUM_FILES_BY_SAMPLE.get(tenx_id, VISIUM_FILES):
            name = f"{tenx_id}{suffix}"
            url = f"{TENX_ROOT}/{tenx_id}/{name}"
            target = folder / name
            download(url, target, size)
            if suffix == "_spatial.tar.gz":
                print("  unpack  spatial/")
                with tarfile.open(target) as archive:
                    members = [
                        m for m in archive.getmembers() if not m.name.startswith(("/", ".."))
                    ]
                    try:
                        archive.extractall(folder, members=members, filter="data")
                    except TypeError:  # Python < 3.12 has no extraction filter
                        archive.extractall(folder, members=members)
                extracted = folder / name[: -len(".tar.gz")]
                wanted = folder / "spatial"
                if extracted.is_dir() and not wanted.exists():
                    extracted.rename(wanted)
            elif suffix == "_filtered_feature_bc_matrix.h5":
                # The preprocessor reads the plain Visium name.
                plain = folder / "filtered_feature_bc_matrix.h5"
                if target.is_file() and not plain.exists():
                    shutil.copyfile(target, plain)
                    print(f"  copy    {plain.name}")
        written[sample] = {"image": f"{tenx_id}_image.tif", "dir": sample}
    return written


def _is_combined_checkpoint(path: Path) -> "tuple[bool, str]":
    """Ask AURORA itself whether ``path`` is a checkpoint it can read."""
    result = subprocess.run(
        [sys.executable, "-m", "aurora", "inspect", "--model", str(path)],
        capture_output=True,
        text=True,
        cwd=str(HERE.parent),
    )
    lines = (result.stderr or result.stdout or "").strip().splitlines()
    return result.returncode == 0, (lines[0] if lines else "")


def download_model(model_file: Path | None = None) -> Path | None:
    """Fetch the published full LUAD model into ``examples/demo/models/``.

    The published file may already be the combined checkpoint that AURORA inference
    reads, or a legacy state_dict from the original code base.  Rather than assuming
    either, the download is checked with ``aurora inspect``: a combined file is copied
    to ``aurora_combined.pth``, and a legacy one is reported with the exact command
    that converts it.  Nothing silently produces a checkpoint inference cannot open.
    """
    models = DEMO / "models"
    published = models / MODEL_NAME
    print(f"\nLUAD model → {published.relative_to(HERE)}")
    if model_file is not None:
        source = Path(model_file).expanduser()
        if not source.is_file():
            raise SystemExit(f"  failed  {source} does not exist")
        models.mkdir(parents=True, exist_ok=True)
        if source.resolve() != published.resolve():
            shutil.copyfile(source, published)
        print(f"  copy    {source.name} → {published.name}")
    else:
        download(MODEL_URL, published, MODEL_SIZE)

    combined = models / COMBINED_NAME
    ok, detail = _is_combined_checkpoint(published)
    if ok:
        if not combined.is_file() or combined.stat().st_mtime < published.stat().st_mtime:
            shutil.copyfile(published, combined)
            print(f"  copy    {combined.name}")
        print(
            f"  ready   {combined.relative_to(HERE)} "
            f"({human(combined.stat().st_size)})"
        )
        return combined

    print(f"  note    {published.name} ({human(published.stat().st_size)}) is not a "
          "combined AURORA checkpoint")
    if detail:
        print(f"          {detail[:160]}")
    repo = HERE.parent
    highres = repo / "trained_model" / HIGHRES_WEIGHTS
    if highres.is_file():
        print("          if this file is only the low-resolution model, combine it with")
        print("          the high-resolution weights that ship in trained_model/:")
        print()
        print(f"            python tools/merge_aurora_checkpoints.py \\")
        print(f"                --lowres {published} \\")
        print(f"                --highres {highres} --out {combined} \\")
        print("                --gene-list trained_model/LUAD_gene.csv \\")
        print("                --cell-types trained_model/LUAD_cell_types.csv \\")
        print("                --norm-factors trained_model/LUAD_gene_normalize_factor.csv \\")
        print("                --lowres-params trained_model/model_parameter_LUAD.json \\")
        print("                --highres-params trained_model/model_parameter_LUAD_highres.json")
        print()
    return None


def write_sheets(visium: dict) -> None:
    inference = DEMO / "Inference_sample.csv"
    inference.write_text(
        "Sample_name,Image_name,Bulk_sample_name\n"
        "TCGA-44-5645-01B-04-BS4,"
        "TCGA-44-5645-01B-04-BS4.svs,"
        "26fdaa75-cea5-4355-b3b1-6b5c2c75e542\n",
        encoding="utf-8",
    )
    print(f"\nwrote {inference.relative_to(HERE)}")

    rows = ["Sample,Dir_name,Image_name"]
    for sample, info in visium.items():
        rows.append(f"{sample},{info['dir']},{info['image']}")
    training = DEMO / "Training_samples.csv"
    training.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {training.relative_to(HERE)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--what",
        choices=["all", "inference", "training", "model"],
        default="all",
        help="which dataset to fetch ('model' is the LUAD weights on their own)",
    )
    parser.add_argument(
        "--model-file",
        default=None,
        help="use an already-downloaded AURORA_LUAD_V2.1.pth instead of fetching it",
    )
    parser.add_argument(
        "--sheets-only",
        action="store_true",
        help="only rewrite the sample sheets; download nothing",
    )
    args = parser.parse_args()

    print(f"demo directory: {DEMO}")
    if not args.sheets_only:
        try:
            if args.what in ("all", "inference"):
                download_slide()
            if args.what == "model":
                download_model(Path(args.model_file) if args.model_file else None)
            elif args.what in ("all", "inference"):
                download_model(Path(args.model_file) if args.model_file else None)
            visium = {}
            if args.what in ("all", "training"):
                visium = download_visium()
            else:
                for sample, tenx_id in VISIUM.items():
                    visium[sample] = {"image": f"{tenx_id}_image.tif", "dir": sample}
            if args.what != "model":
                write_sheets(visium)
        except KeyboardInterrupt:
            print("\ninterrupted; run the same command again to resume")
            return 130
    else:
        write_sheets(
            {
                sample: {"image": f"{tenx_id}_image.tif", "dir": sample}
                for sample, tenx_id in VISIUM.items()
            }
        )

    print("\n" + "=" * 70)
    if not (DEMO / "uni2h" / "config.json").is_file():
        print("One thing left: UNI2-h, which both inference and training need.")
        print("Clone it once into the demo directory:")
        print()
        print("  git clone https://huggingface.co/MahmoodLab/UNI2-h examples/demo/uni2h")
        print()
    print("Then run the examples in the README; no paths need changing.")
    return 0

    print("\n" + "=" * 70)
    if not (DEMO / "uni2h" / "config.json").is_file():
        print("One thing left: UNI2-h, which both inference and training need.")
        print("Clone it once into the demo directory:")
        print()
        print("  git clone https://huggingface.co/MahmoodLab/UNI2-h examples/demo/uni2h")
        print()
    print("Then run the examples in the README; no paths need changing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
