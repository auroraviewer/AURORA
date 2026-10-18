#!/usr/bin/env python
"""Convert an AURORA prediction file between the modern and legacy layouts.

``code/plot.py`` and ``code/run_istar.py`` read the **legacy** layout (3-D
``exp_pred`` / ``prop_pred`` arrays plus ``patches_loc_before_pad``), while
``code/predict.py`` and ``aurora``'s ``predict_lowres`` write the **modern** one
(dicts keyed by patch size).  The ``plot`` stage builds a legacy view
automatically; this tool is for doing it by hand, or for producing a converted
copy next to the original.

    # what is in the file?
    python tools/convert_prediction_format.py --show \\
        aurora_runs/luad_infer/03_predictions/lowres/pred/TCGA-44-5645-01B-04-BS4-pred.pickle

    # write a legacy projection of the 224 px level
    python tools/convert_prediction_format.py \\
        --resolution 224 \\
        --out /tmp/legacy/pred/TCGA-44-5645-01B-04-BS4-pred.pickle \\
        aurora_runs/luad_infer/03_predictions/lowres/pred/TCGA-44-5645-01B-04-BS4-pred.pickle

Only one resolution can be projected: the modern layout is deduplicated, so each
level has its own patch count and they cannot share a single 3-D array.  For a
true multi-level legacy file, re-run the prediction with
``--output-format legacy``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aurora.errors import AuroraError  # noqa: E402
from aurora.predictions import (  # noqa: E402
    LAYOUT_LEGACY,
    LAYOUT_MODERN,
    Prediction,
    choose_resolution,
    load_prediction,
    write_legacy_view,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an AURORA prediction file to the legacy layout code/plot.py reads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prediction", help="Input prediction pickle (modern or legacy)")
    parser.add_argument("--out", default=None, help="Where to write the legacy file")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Patch size to project (default: the smallest available)",
    )
    parser.add_argument(
        "--all-resolutions",
        action="store_true",
        help="Write one legacy file per resolution into --out-dir",
    )
    parser.add_argument("--out-dir", default=None, help="Directory for --all-resolutions")
    parser.add_argument("--show", action="store_true", help="Describe the file and exit")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    return parser.parse_args(argv)


def describe(prediction: Prediction) -> None:
    print(prediction.describe())
    print()
    print("  code/plot.py can read this file directly" if prediction.layout == LAYOUT_LEGACY
          else "  code/plot.py cannot read this file directly; convert it or use "
               "'aurora run --stage plot'")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    source = Path(args.prediction)
    prediction = load_prediction(source)
    describe(prediction)

    if args.show:
        return 0

    if args.all_resolutions:
        if not args.out_dir:
            raise AuroraError(
                "--all-resolutions needs --out-dir",
                hint="Pass the directory the converted files should be written into.",
            )
        out_dir = Path(args.out_dir)
        written = []
        for resolution in prediction.resolutions:
            if resolution not in prediction.exp_pred:
                continue
            target = out_dir / f"{source.stem}-{resolution}{source.suffix}"
            if target.exists() and not args.force:
                print(f"skipping {target} (exists; pass --force)")
                continue
            write_legacy_view(prediction, target, resolution)
            written.append(target)
            print(f"wrote {target} ({resolution} px, {len(prediction.exp_pred[resolution])} spots)")
        if not written:
            print("nothing written")
        return 0

    if not args.out:
        raise AuroraError(
            "an output path is required",
            hint=(
                "Pass --out <file>, or --show to inspect the file without converting.\n"
                "The plot stage does this automatically:\n"
                "  python -m aurora run --stage plot --gene IL7R KRT8"
            ),
        )
    target = Path(args.out)
    if target.exists() and not args.force:
        raise AuroraError(
            f"{target} already exists",
            hint="Pass --force to overwrite it, or choose a different --out path.",
        )
    chosen = choose_resolution(prediction, args.resolution)
    write_legacy_view(prediction, target, chosen)
    print()
    print(f"wrote {target} ({chosen} px, {len(prediction.exp_pred[chosen])} spots)")
    print()
    print("code/plot.py reads <pred_dir>/pred/<sample>-pred.pickle, so put the file there:")
    print(f"  python -u code/plot.py -p <project> -i 02_features -t {target.parent.parent} \\")
    print(f"      --plot_dir 05_plots --resolution {chosen} --samples <sample> --genes <gene>")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuroraError as exc:
        print(f"error: {exc.format()}", file=sys.stderr)
        sys.exit(2)
