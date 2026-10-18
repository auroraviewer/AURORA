#!/usr/bin/env python
"""Compare two AURORA prediction files, to prove the new code matches the old.

The reorganisation reimplements the two prediction stages inside the ``aurora``
package (they must read the combined checkpoint).  Their numerics are meant to be
identical to ``code/predict.py`` and ``code/predict_highres.py``; this tool is how
you check that on your own data.

Typical use::

    # 1. run the original script, which writes into the old flat layout
    python -u code/predict.py -p . -i AURORA_interim -t AURORA_pred \\
        -j model_parameter_LUAD.json --model_pth AURORA_LUAD.pth \\
        --sample_sheet Inference_samples.csv --bulk TCGA_LUAD_expression.csv

    # 2. run the new stage against the same features
    python -m aurora run --run parity --stage predict_lowres --force

    # 3. compare
    python tools/compare_predictions.py \\
        AURORA_pred/pred/TCGA-86-A4P8-01A-01-TSA-pred.pickle \\
        aurora_runs/parity/03_predictions/lowres/pred/TCGA-86-A4P8-01A-01-TSA-pred.pickle

Both the modern layout (dict keyed by patch size) and the legacy layout
3-D array with ``patches_loc_before_pad``) are understood, and either can be on
either side.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aurora.deps import require  # noqa: E402
from aurora.errors import AuroraError  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two AURORA prediction pickles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("left", help="Reference prediction file (usually the original script's output)")
    parser.add_argument("right", help="Prediction file to check (usually the new stage's output)")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance (default: exact)")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance (default: exact)")
    parser.add_argument(
        "--compare-positions",
        action="store_true",
        help="Also compare the patch coordinates, not just the predictions",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the verdict")
    return parser.parse_args(argv)


def load_prediction(path: Path) -> Dict[str, Any]:
    """Load and normalise a prediction file of either layout."""
    if not path.is_file():
        raise AuroraError(f"{path} does not exist")
    with open(path, "rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise AuroraError(f"{path} does not contain a prediction dictionary (got {type(raw).__name__})")

    resolutions = [int(x) for x in raw.get("resolutions", [])]
    genes = list(raw.get("genes", []))
    cell_types = list(raw.get("cell_types", []))

    exp = raw.get("exp_pred")
    log_prop = raw.get("log_prop_pred")
    prop = raw.get("prop_pred")
    positions = raw.get("pos")
    legacy_positions = raw.get("patches_loc_before_pad")

    # Modern layout: dict keyed by patch size.
    if isinstance(exp, dict):
        exp_by_res = {int(k): v for k, v in exp.items()}
        if isinstance(log_prop, dict):
            log_prop_by_res = {int(k): v for k, v in log_prop.items()}
        elif isinstance(prop, dict):
            import numpy as np

            log_prop_by_res = {int(k): np.log(np.clip(v, 1e-30, None)) for k, v in prop.items()}
        else:
            log_prop_by_res = {}
        pos_by_res = {int(k): v for k, v in positions.items()} if isinstance(positions, dict) else {}
        layout = "modern"
    else:
        # Legacy layout: 3-D array indexed [spot, resolution index, feature].
        import numpy as np

        if exp is None:
            raise AuroraError(f"{path} has no usable 'exp_pred' entry")
        exp_by_res = {res: exp[:, index, :] for index, res in enumerate(resolutions)}
        if isinstance(log_prop, (list, tuple)) or hasattr(log_prop, "shape"):
            log_prop_by_res = {res: log_prop[:, index, :] for index, res in enumerate(resolutions)}
        elif prop is not None and hasattr(prop, "shape"):
            log_prop_by_res = {
                res: np.log(np.clip(prop[:, index, :], 1e-30, None))
                for index, res in enumerate(resolutions)
            }
        else:
            log_prop_by_res = {}
        source_positions = legacy_positions if isinstance(legacy_positions, dict) else positions
        pos_by_res = (
            {int(k): v for k, v in source_positions.items()} if isinstance(source_positions, dict) else {}
        )
        layout = "legacy"

    return {
        "path": path,
        "layout": layout,
        "resolutions": resolutions,
        "genes": genes,
        "cell_types": cell_types,
        "exp_pred": exp_by_res,
        "log_prop_pred": log_prop_by_res,
        "pos": pos_by_res,
    }


def _describe(name: str, left: Any, right: Any, quiet: bool) -> bool:
    if left == right:
        if not quiet:
            print(f"  ok    {name}: {left}")
        return True
    print(f"  DIFF  {name}: {left} != {right}")
    return False


def _compare_arrays(
    name: str,
    left: Any,
    right: Any,
    atol: float,
    rtol: float,
    quiet: bool,
) -> bool:
    import numpy as np

    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        print(f"  DIFF  {name}: shape {a.shape} != {b.shape}")
        return False
    if a.size == 0:
        if not quiet:
            print(f"  ok    {name}: empty")
        return True
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        print(f"  WARN  {name}: contains non-finite values")
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    worst = float(difference.max())
    scale = float(np.abs(a.astype(np.float64)).max()) or 1.0
    close = np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
    if close:
        if not quiet:
            print(f"  ok    {name}: shape {a.shape}, max |diff| = {worst:.3e} (relative {worst / scale:.2e})")
        return True
    print(
        f"  DIFF  {name}: shape {a.shape}, max |diff| = {worst:.3e} (relative {worst / scale:.2e}), "
        f"{int((difference > atol).sum())} element(s) outside the tolerance"
    )
    return False


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    require("numpy", purpose="comparing predictions")

    left = load_prediction(Path(args.left))
    right = load_prediction(Path(args.right))
    print(f"left  : {left['path']}  ({left['layout']} layout)")
    print(f"right : {right['path']}  ({right['layout']} layout)")
    print()

    ok = True
    ok &= _describe("genes", len(left["genes"]), len(right["genes"]), args.quiet)
    ok &= _describe("cell types", len(left["cell_types"]), len(right["cell_types"]), args.quiet)
    ok &= _describe("resolutions", left["resolutions"], right["resolutions"], args.quiet)
    if left["genes"] != right["genes"]:
        print("  DIFF  gene names differ")
        ok = False
    if left["cell_types"] != right["cell_types"]:
        print("  DIFF  cell-type names differ")
        ok = False

    shared = sorted(set(left["exp_pred"]) & set(right["exp_pred"]))
    if not shared:
        print("  DIFF  the two files have no patch size in common")
        return 1
    only_left = sorted(set(left["exp_pred"]) - set(right["exp_pred"]))
    only_right = sorted(set(right["exp_pred"]) - set(left["exp_pred"]))
    if only_left:
        print(f"  note  patch size(s) only on the left: {only_left}")
    if only_right:
        print(f"  note  patch size(s) only on the right: {only_right}")

    print()
    for resolution in shared:
        ok &= _compare_arrays(
            f"exp_pred[{resolution}]",
            left["exp_pred"][resolution],
            right["exp_pred"][resolution],
            args.atol,
            args.rtol,
            args.quiet,
        )
        if resolution in left["log_prop_pred"] and resolution in right["log_prop_pred"]:
            ok &= _compare_arrays(
                f"log_prop_pred[{resolution}]",
                left["log_prop_pred"][resolution],
                right["log_prop_pred"][resolution],
                args.atol,
                args.rtol,
                args.quiet,
            )
        if args.compare_positions and resolution in left["pos"] and resolution in right["pos"]:
            import numpy as np

            a = np.asarray(left["pos"][resolution])
            b = np.asarray(right["pos"][resolution])
            if a.shape == b.shape:
                # One layout stores (x, y) and the other (y, x); accept either.
                same = bool(np.array_equal(a, b))
                flipped = bool(np.array_equal(a[:, ::-1], b))
                if same:
                    print(f"  ok    pos[{resolution}]: identical ({a.shape[0]} patches)")
                elif flipped:
                    print(f"  ok    pos[{resolution}]: identical up to the x/y column order")
                else:
                    print(f"  DIFF  pos[{resolution}]: coordinates differ")
                    ok = False
            else:
                print(
                    f"  note  pos[{resolution}]: {a.shape[0]} vs {b.shape[0]} entries "
                    "(the legacy layout stores one row per coarse patch, the modern one deduplicates)"
                )

    print()
    print("VERDICT:", "IDENTICAL" if ok else "DIFFERENT")
    if not ok:
        print()
        print("If the numbers differ, check that both runs used the same")
        print("  * sample sheet and bulk matrix,")
        print("  * feature directory (emb_*), and")
        print("  * model parameters / checkpoint.")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuroraError as exc:
        print(f"error: {exc.format()}", file=sys.stderr)
        sys.exit(2)
