#!/usr/bin/env python
"""Convert a combined checkpoint container into the optional native single-file form.

Two combined-checkpoint layouts are supported by the reader:

``torch_archive`` (default)
    A ZIP container with a JSON manifest, one embedded ``torch.save`` payload per
    stage, and the assets.  Building it needs no PyTorch, which is why
    ``tools/merge_aurora_checkpoints.py`` produces it.

``inline_state_dict`` (this tool)
    A single ``torch.save`` archive of::

        {"__aurora__": <manifest>, "lowres": <state_dict>, "highres": <state_dict>}

    so that a plain ``torch.load(path)`` returns everything.  Use it when some
    downstream consumer expects a native torch file.

    python tools/export_native_combined.py \
        --model trained_model/AURORA_LUAD_combined.pth \
        --out   trained_model/AURORA_LUAD_combined_native.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aurora.bundle import (  # noqa: E402
    INLINE_MARKER_KEY,
    PAYLOAD_KIND_INLINE,
    CombinedCheckpoint,
    classify_checkpoint,
    validate_bundle,
)
from aurora.deps import require  # noqa: E402
from aurora.errors import AuroraError  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a combined AURORA checkpoint as one native torch.save file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Input combined checkpoint (.pth)")
    parser.add_argument("--out", required=True, help="Output native combined checkpoint (.pth)")
    parser.add_argument("--device", default="cpu", help="map_location used while loading (default: cpu)")
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it exists")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    torch = require("torch", purpose="exporting a native combined checkpoint")

    source = Path(args.model)
    destination = Path(args.out)
    if destination.exists() and not args.force:
        raise AuroraError(
            f"{destination} already exists",
            hint="Pass --force to overwrite it, or choose a different --out path.",
        )

    kind = classify_checkpoint(source)
    if not kind.is_combined:
        raise AuroraError(
            f"{source} is not a combined AURORA checkpoint ({kind.reason or kind.kind})",
            hint=(
                "Build one first:\n"
                "  python tools/merge_aurora_checkpoints.py --lowres <lowres.pth> "
                "--highres <highres.pth> --out trained_model/AURORA_<CANCER>_combined.pth"
            ),
        )

    checkpoint = CombinedCheckpoint.open(source)
    if checkpoint.manifest.payload_kind == PAYLOAD_KIND_INLINE:
        print(f"{source} is already a native inline combined checkpoint; nothing to do.")
        return 0

    print(checkpoint.describe())
    print()
    payload: Dict[str, Any] = {INLINE_MARKER_KEY: checkpoint.manifest.to_json()}
    for stage in checkpoint.stages():
        print(f"Loading the '{stage}' state dict ...")
        payload[stage] = checkpoint.load_stage_state_dict(stage, map_location=args.device)

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {destination} ...")
    torch.save(payload, destination)
    print(f"Wrote {destination} ({destination.stat().st_size / 1e6:.2f} MB)")

    validation = validate_bundle(destination)
    print()
    print(validation.describe())
    if not validation.ok:
        return 1
    print()
    print("The reader auto-detects this layout, so this file is a drop-in replacement:")
    print(f"  python -m aurora run --model {destination} ...")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuroraError as exc:
        print(f"error: {exc.format()}", file=sys.stderr)
        sys.exit(2)
