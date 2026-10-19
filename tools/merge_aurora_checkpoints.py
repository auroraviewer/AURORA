#!/usr/bin/env python
"""Merge the legacy low-resolution and high-resolution AURORA checkpoints.

Requirement #1 of the reorganisation: instead of two ``.pth`` files, AURORA now
ships a single combined checkpoint that the new code system reads.

    python tools/merge_aurora_checkpoints.py \
        --lowres  trained_model/AURORA_LUAD_V2.pth \
        --highres trained_model/AURORA_highres_model_weights.pth \
        --lowres-params  trained_model/model_parameter_LUAD.json \
        --highres-params trained_model/model_parameter_LUAD_highres.json \
        --gene-list    trained_model/LUAD_gene.csv \
        --cell-types   trained_model/LUAD_cell_types.csv \
        --norm-factors trained_model/LUAD_gene_normalize_factor.csv \
        --model-name AURORA_LUAD --cancer-type LUAD \
        --out trained_model/AURORA_LUAD_combined.pth

The original ``.pth`` files are never modified or removed; their bytes are
embedded verbatim inside the container and their SHA-256 is recorded, so the
weights are provably unchanged.

PyTorch is *not* required: the tool inspects the archives with a standard-library
unpickler, which also lets it verify that the two checkpoints really are two
halves of one model before writing anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make 'import aurora' work when the script is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aurora import assets as asset_io  # noqa: E402
from aurora.bundle import (  # noqa: E402
    DEFAULT_REQUIRES,
    BundleManifest,
    build_container_bundle,
    probe_state_dict_shapes,
    sha256_file,
    validate_bundle,
)
from aurora.errors import AuroraError  # noqa: E402

DEFAULT_OUT = "trained_model/AURORA.pth"

#: ``(asset key, argparse attribute, CLI flag)`` for every bundleable asset.
ASSET_FLAGS: Tuple[Tuple[str, str, str], ...] = (
    ("gene_list", "gene_list", "--gene-list"),
    ("cell_types", "cell_types", "--cell-types"),
    ("gene_norm_factor", "norm_factors", "--norm-factors"),
    ("bulk_expression", "bulk", "--bulk"),
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine the low-resolution and high-resolution AURORA checkpoints into one .pth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lowres", required=True, help="Legacy low-resolution checkpoint (.pth)")
    parser.add_argument("--highres", required=True, help="Legacy high-resolution checkpoint (.pth)")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Combined checkpoint to write (default: {DEFAULT_OUT})")
    parser.add_argument("--lowres-params", default=None, help="model_parameter_*.json of the low-resolution model")
    parser.add_argument("--highres-params", default=None, help="model_parameter_*_highres.json of the high-resolution model")
    parser.add_argument("--gene-list", default=None, help="Gene list CSV to bundle")
    parser.add_argument("--cell-types", default=None, help="Cell-type CSV to bundle (no header)")
    parser.add_argument("--norm-factors", default=None, help="Bulk normalisation-factor CSV to bundle")
    parser.add_argument("--bulk", default=None, help="Reference bulk expression matrix to bundle (optional)")
    parser.add_argument("--model-name", default="", help="Name recorded in the manifest, e.g. AURORA_LUAD")
    parser.add_argument("--cancer-type", default="", help="Cancer type recorded in the manifest, e.g. LUAD")
    parser.add_argument("--notes", default="", help="Free-text note stored in the manifest")
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it exists")
    parser.add_argument(
        "--allow-shape-mismatch",
        action="store_true",
        help="Do not fail when the two state dicts differ in more than the input dimension",
    )
    parser.add_argument(
        "--compresslevel",
        type=int,
        default=6,
        help="zlib level for the container (0 = store payloads uncompressed, default 6)",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary")
    return parser.parse_args(argv)


def read_params(path: Optional[str]) -> Dict[str, object]:
    """Read a model parameter JSON, tolerating a missing file."""
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.is_file():
        raise AuroraError(f"parameter file {resolved} does not exist")
    return asset_io.read_params(resolved)


def describe_shapes(label: str, shapes: Dict[str, Optional[tuple]]) -> List[str]:
    lines = [f"  {label}: {len(shapes)} tensor(s)"]
    for key, shape in shapes.items():
        lines.append(f"    {key}: {shape}")
    return lines


def compare_state_dicts(
    lowres: Dict[str, Optional[tuple]],
    highres: Dict[str, Optional[tuple]],
) -> Tuple[List[str], List[str]]:
    """Return ``(differences, problems)`` between the two state dicts."""
    differences: List[str] = []
    problems: List[str] = []
    low_keys = set(lowres)
    high_keys = set(highres)
    for key in sorted(low_keys - high_keys):
        problems.append(f"key present only in --lowres: {key}")
    for key in sorted(high_keys - low_keys):
        problems.append(f"key present only in --highres: {key}")
    for key in sorted(low_keys & high_keys):
        if lowres[key] != highres[key]:
            differences.append(f"{key}: {lowres[key]} -> {highres[key]}")
    return differences, problems


def check_input_dim(params: Dict[str, object], shapes: Dict[str, Optional[tuple]], stage: str) -> List[str]:
    """Verify ``img_emb_dim`` against the LSTM input weight of that stage."""
    problems: List[str] = []
    declared = params.get("img_emb_dim")
    weight = shapes.get("lstm.weight_ih_l0")
    if declared is None:
        problems.append(f"stage '{stage}': the parameter JSON does not declare img_emb_dim")
        return problems
    if weight is None:
        problems.append(f"stage '{stage}': the checkpoint has no 'lstm.weight_ih_l0' tensor")
        return problems
    if len(weight) != 2:
        problems.append(f"stage '{stage}': lstm.weight_ih_l0 has an unexpected shape {weight}")
        return problems
    if int(weight[1]) != int(declared):
        problems.append(
            f"stage '{stage}': img_emb_dim={declared} in the parameter JSON but the checkpoint "
            f"expects {weight[1]} (lstm.weight_ih_l0 shape {weight})"
        )
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out)

    print("Inspecting the two legacy checkpoints (PyTorch is not required) ...")
    lowres_shapes = probe_state_dict_shapes(args.lowres)
    highres_shapes = probe_state_dict_shapes(args.highres)
    if not args.quiet:
        print("\n".join(describe_shapes("lowres ", lowres_shapes)))
        print("\n".join(describe_shapes("highres", highres_shapes)))

    differences, problems = compare_state_dicts(lowres_shapes, highres_shapes)
    if problems:
        raise AuroraError(
            "the two checkpoints are not two halves of the same model:\n  "
            + "\n  ".join(problems),
            hint=(
                "Both stages must share the identical AURORA parameter names. "
                "Check that you passed the low-resolution model with --lowres and the "
                "high-resolution one with --highres."
            ),
        )
    if differences:
        print("\nShape differences between the two stages (expected: the feature dimension only):")
        for line in differences:
            print(f"  - {line}")
    else:
        print("\nThe two state dicts have identical shapes.")

    expected = {"lstm.weight_ih_l0"}
    unexpected = [d.split(":", 1)[0] for d in differences if d.split(":", 1)[0] not in expected]
    if unexpected and not args.allow_shape_mismatch:
        raise AuroraError(
            "the two checkpoints differ in more tensors than the feature dimension: "
            + ", ".join(unexpected),
            hint=(
                "This usually means the two files were not trained with the same parameter JSON. "
                "Pass --allow-shape-mismatch to merge anyway, after checking the shapes above."
            ),
        )

    lowres_params = read_params(args.lowres_params)
    highres_params = read_params(args.highres_params)

    problems = []
    problems += check_input_dim(lowres_params, lowres_shapes, "lowres")
    problems += check_input_dim(highres_params, highres_shapes, "highres")
    for stage, params in (("lowres", lowres_params), ("highres", highres_params)):
        if params and not (params.get("patch_size") or params.get("patch_sizes")):
            problems.append(f"stage '{stage}': the parameter JSON does not declare patch_size")
    if problems:
        for problem in problems:
            print(f"WARNING: {problem}")

    assets = {}
    for key, attribute, flag in ASSET_FLAGS:
        value = getattr(args, attribute, None)
        if value:
            assets[key] = Path(value)

    # Pre-flight the assets so a malformed CSV fails before anything is written.
    gene_list: List[str] = []
    cell_types: List[str] = []
    if "gene_list" in assets:
        gene_list = asset_io.read_gene_list(assets["gene_list"])
        print(f"Gene list      : {len(gene_list)} genes from {assets['gene_list']}")
    if "cell_types" in assets:
        header_guess = 0
        cell_types = asset_io.read_cell_types(assets["cell_types"])
        print(f"Cell types     : {len(cell_types)} types from {assets['cell_types']}")
    if "gene_norm_factor" in assets:
        if not gene_list:
            raise AuroraError(
                "--norm-factors needs --gene-list so the factors can be validated against the genes"
            )
        asset_io.read_norm_factors(assets["gene_norm_factor"], gene_list)
        print(f"Norm factors   : validated for all {len(gene_list)} genes")

    stage_params = {"lowres": lowres_params, "highres": highres_params}
    stage_metadata = {
        "lowres": {
            "patch_sizes": lowres_params.get("patch_size", []),
            "img_emb_dim": lowres_params.get("img_emb_dim"),
            "extractor": "UNI2-h",
            "state_dict_keys": len(lowres_shapes),
        },
        "highres": {
            "patch_sizes": highres_params.get("patch_size", []),
            "img_emb_dim": highres_params.get("img_emb_dim"),
            "extractor": "UNI2-h",
            "state_dict_keys": len(highres_shapes),
        },
    }

    manifest = BundleManifest(
        model_name=args.model_name,
        cancer_type=args.cancer_type,
        created_by=f"tools/merge_aurora_checkpoints.py (aurora {_version()})",
        requires=dict(DEFAULT_REQUIRES),
        notes=args.notes
        or (
            "Combined checkpoint: 'lowres' and 'highres' are the two stages of one AURORA model. "
            "'highres' requires 'lowres' -- the new pipeline enforces this."
        ),
    )

    print(f"\nWriting {out_path} ...")
    checkpoint = build_container_bundle(
        out_path,
        {"lowres": Path(args.lowres), "highres": Path(args.highres)},
        manifest=manifest,
        model_name=args.model_name,
        cancer_type=args.cancer_type,
        assets=assets,
        stage_params=stage_params,
        stage_metadata=stage_metadata,
        notes=manifest.notes,
        overwrite=args.force,
        compresslevel=max(0, int(args.compresslevel)),
    )

    validation = validate_bundle(out_path)
    print()
    print(validation.describe())
    if not validation.ok:
        raise AuroraError(
            f"the combined checkpoint {out_path} failed validation; it was still written so you "
            "can inspect it, but do not use it",
        )

    print()
    print(checkpoint.describe())
    print()
    print("Payload integrity (embedded verbatim; torch.load needs the archive root):")
    for stage in ("lowres", "highres"):
        source = Path(getattr(args, stage))
        entry = checkpoint.stage(stage)
        embedded = checkpoint.payload_bytes(stage)
        actual_source = sha256_file(source)
        marker = "ok " if actual_source == entry.source_sha256 else "DIFFERS"
        print(f"  [{marker}] {stage:<8} {source}  sha256={actual_source[:16]}…")
        if entry.archive_root:
            print(
                f"           archive root: {entry.archive_root!r} "
                "(kept exactly as torch.save wrote it)"
            )
        else:
            print("           WARNING: no archive root detected; torch.load may reject this payload")
        print(
            f"           embedded payload: {len(embedded) / 1e6:.2f} MB  "
            f"sha256={entry.sha256[:16]}…"
        )
    print()
    print("Next steps:")
    print(f"  python -m aurora validate --model {out_path}")
    print(f"  python -m aurora inspect  --model {out_path} --assets")
    print(
        "  python -m aurora run --run my_run --samples examples/Inference_samples.csv "
        f"--model {out_path}"
    )
    print()
    print("The original low-resolution and high-resolution files were left untouched.")
    return 0


def _version() -> str:
    from aurora.version import __version__

    return __version__


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuroraError as exc:
        print(f"error: {exc.format()}", file=sys.stderr)
        sys.exit(2)
