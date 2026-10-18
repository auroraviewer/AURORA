"""High-resolution prediction stage (native, combined-checkpoint only).

Port of ``code/predict_highres.py`` with the same numerical path: the
``emb_{56}`` / ``emb_{14}`` features are stacked as ``[fine, coarse]``, run
through the high-resolution stage of the combined checkpoint, reversed back and
deduplicated per patch size.

Requirement #2 is enforced here as well as in the CLI: the stage refuses to run
unless the low-resolution prediction stage of the same run is complete and its
output files are present.  That is checked against ``run.json`` (see
:func:`aurora.state.assert_prerequisites`), so calling this module directly does
not bypass the gate.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..checkpoint import load_assets, load_stage_model
from ..deps import require
from ..errors import AuroraError, ConfigError, StagePrerequisiteError
from ..naming import highres_output_path
from .. import h5out, masks
from . import common


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aurora.inference.predict_highres",
        description=(
            "Refine the low-resolution prediction with the high-resolution model. "
            "The low-resolution prediction must have been run first."
        ),
    )
    common.add_highres_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    torch = require("torch", purpose="running the high-resolution prediction")
    np = require("numpy", purpose="running the high-resolution prediction")
    h5py = require("h5py", purpose="writing the high-resolution predictions")
    tqdm = require("tqdm", purpose="running the high-resolution prediction").tqdm

    layout = common.load_run_layout(args.run_root)
    checkpoint = common.open_checkpoint(args.model)
    sheet = common.read_sample_sheet_for_run(layout.root, args.samples)
    if args.limit:
        sheet = sheet.with_names(sheet.sample_names[: args.limit])

    device = common.resolve_device(args.device)
    assets = load_assets(checkpoint, search_dirs=[layout.root, layout.config_dir, Path(args.model).parent])

    # ------------------------------------------------------------------ #
    # Requirement #2 -- the high-resolution model never runs on its own.  #
    # ------------------------------------------------------------------ #
    if not args.skip_prerequisite_check:
        common.enforce_prerequisites(
            layout,
            checkpoint,
            sheet,
            assets,
            "predict_highres",
            options={"device": str(device), "mode": "inference"},
        )

    bad_names = [name for name in sheet.sample_names if "." in name]
    if bad_names:
        raise ConfigError(
            "sample names must not contain a '.' for high-resolution feature extraction "
            f"(offending: {', '.join(bad_names[:3])})",
            hint=(
                "code/extract_features_highres.py derives the file name with "
                "image_file.split('.')[0], so a dot would truncate it. Rename the Sample_name "
                "values (the image file name may still contain dots)."
            ),
        )

    stage_model = load_stage_model(checkpoint, "highres", device=device)
    parameters = stage_model.params
    gene_list: List[str] = list(stage_model.gene_list)
    major_cell_type: List[str] = list(stage_model.cell_types)
    output_size = len(gene_list)
    input_size = stage_model.img_emb_dim
    bidirectional = bool(parameters.get("bidirectional", False))
    use_bulk = bool(parameters.get("use_bulk", True))
    pred_celltype = bool(parameters.get("pred_celltype", True))

    patch_sizes = stage_model.patch_sizes
    if len(patch_sizes) < 2:
        raise ConfigError(
            f"the high-resolution stage expects two patch sizes, the checkpoint records {patch_sizes}"
        )
    superpatch_minlowres_patch = int(parameters.get("superpatch_minlowres_patch", 10))

    interim_dir = args.interim_dir
    pred_dir = args.pred_dir
    cache_dir = f"{pred_dir}/processed_pkl_test_highres"
    out_dir = Path(layout.root) / pred_dir / "pred"
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path(layout.root) / cache_dir).mkdir(parents=True, exist_ok=True)

    from .._legacy import ensure_code_on_path

    ensure_code_on_path()
    import train_utils  # noqa: E402

    dataset_dir = str(Path(layout.root) / cache_dir)
    dataset_exist = [
        x.replace(".pkl", "") for x in os.listdir(dataset_dir) if x.endswith(".pkl")
    ] if os.path.exists(dataset_dir) else []

    lowres_pred_dir = Path(layout.root) / args.lowres_dir / "pred"
    # The rescaled slides and the HistoSweep masks written beside them.
    rescaled_dir = Path(layout.root) / args.rescaled_dir
    common.describe_plan(
        "predict_highres",
        sheet,
        device,
        extra=[
            f"checkpoint        : {checkpoint.path}",
            f"model             : {checkpoint.model_id} (highres)",
            f"patch sizes       : {patch_sizes}",
            f"low-res prior from: {lowres_pred_dir}",
            f"histosweep masks  : {rescaled_dir}",
            f"output            : {out_dir}",
        ],
    )

    model = stage_model.model
    model.eval()
    test_batch_size = max(1, int(args.batch_size))
    written: List[str] = []

    for idx in range(len(sheet.sample_names)):
        dataset_test_use = sheet.sample_names[idx]
        # Resume on the file this run will actually produce.  Checking for *any*
        # of the three formats would skip a sample whose old output used a
        # different --uint8-output / --compress-output combination, leaving the
        # stage with nothing to verify.
        expected_path = common.expected_output(
            out_dir,
            dataset_test_use,
            highres=True,
            uint8_output=bool(args.uint8_output),
            compress_output=bool(args.compress_output),
        )
        if not args.force and expected_path.exists():
            print(f"{dataset_test_use} already predicted, skipping (use --force to recompute)")
            continue
        if not args.force and not _lowres_prior_exists(lowres_pred_dir, dataset_test_use):
            raise StagePrerequisiteError(
                stage="predict_highres",
                missing=["predict_lowres"],
                reason=(
                    f"the low-resolution prediction for sample '{dataset_test_use}' is missing: "
                    f"expected {lowres_pred_dir / (dataset_test_use + '-pred.pickle')}"
                ),
                hint=(
                    "Re-run the low-resolution stage for this sample before the high-resolution one:\n"
                    f"  python -m aurora run --run {layout.run_name} --stage predict_lowres"
                ),
            )
        with torch.no_grad():
            if dataset_test_use in set(dataset_exist):
                print("Samples loading from disk")
                test_data = train_utils.load_data2_pred(
                    [dataset_test_use],
                    main_dir=str(layout.root),
                    interim_dir=interim_dir,
                    pred_dir=pred_dir,
                    patch_sizes=patch_sizes,
                    from_file=True,
                    file_dir=dataset_dir,
                    num_workers=int(args.num_workers),
                    superpatch_size=1e8,
                    superpatch_stride=1e8,
                    superpatch_minsize=0,
                    superpatch_minlowres_patch=superpatch_minlowres_patch,
                )
            else:
                print("Samples loading from scratch")
                test_data = train_utils.load_data2_pred(
                    [dataset_test_use],
                    main_dir=str(layout.root),
                    interim_dir=interim_dir,
                    pred_dir=pred_dir,
                    patch_sizes=patch_sizes,
                    from_file=False,
                    file_dir=dataset_dir,
                    num_workers=int(args.num_workers),
                    superpatch_size=1e8,
                    superpatch_stride=1e8,
                    superpatch_minsize=0,
                    superpatch_minlowres_patch=superpatch_minlowres_patch,
                )

            for index in tqdm(np.arange(len(test_data)), desc="Evaluation", unit="batch", leave=False):
                emb, _father_lowres, father_patches, patches_loc_before_pad = test_data[index]
                for key in emb.keys():
                    emb[key] = torch.as_tensor(emb[key], dtype=torch.float32)
                for key in father_patches.keys():
                    father_patches[key] = torch.as_tensor(father_patches[key], dtype=torch.int)

                n_fine = emb[patch_sizes[1]].shape[0]
                father_coarse = father_patches[patch_sizes[0]]
                emb_coarse = emb[patch_sizes[0]]
                emb_fine = emb[patch_sizes[1]]
                output_tmp = []
                major_prop_tmp = []
                total_batch = n_fine // test_batch_size + 1
                for batch in tqdm(
                    range(total_batch), desc=f"Processing {dataset_test_use}", unit="batch", leave=False
                ):
                    batch_start = batch * test_batch_size
                    batch_end = min((batch + 1) * test_batch_size, n_fine)
                    if batch_start >= batch_end:
                        continue
                    emb_hierrachy_batch = torch.stack(
                        [
                            emb_fine[batch_start:batch_end, :],
                            emb_coarse[father_coarse[batch_start:batch_end], :],
                        ],
                        dim=1,
                    ).to(device)
                    output_batch, major_prop_pred_batch = model(emb_hierrachy_batch, None, highres=True)
                    output_tmp.append(output_batch.cpu())
                    major_prop_tmp.append(major_prop_pred_batch.cpu())
                output = torch.cat(output_tmp, dim=0)
                major_prop_pred = torch.cat(major_prop_tmp, dim=0)
                output = output[:, [1, 0], :]
                major_prop_pred = major_prop_pred[:, [1, 0], :]
                output = output.numpy()
                major_prop_pred = major_prop_pred.numpy()

                # HistoSweep is a mandatory stage: its mask decides which patches
                # are kept, so a missing mask is an error, not a warning.
                histosweep_mask = masks.load_mask(
                    masks.require_mask(rescaled_dir, dataset_test_use), np
                )
                print(
                    f"    HistoSweep mask: {masks.mask_path(rescaled_dir, dataset_test_use).name} "
                    f"{histosweep_mask.shape}"
                )

                _write_sample(
                    sample_name=dataset_test_use,
                    output=output,
                    major_prop_pred=major_prop_pred,
                    father_patches={k: v.cpu() for k, v in father_patches.items()},
                    patches_loc_before_pad=patches_loc_before_pad,
                    patch_sizes=patch_sizes,
                    gene_list=gene_list,
                    major_cell_type=major_cell_type,
                    parameters=parameters,
                    out_dir=out_dir,
                    uint8_output=bool(args.uint8_output),
                    compress_output=bool(args.compress_output),
                    h5py=h5py,
                    np=np,
                    train_utils=train_utils,
                    pred_celltype=pred_celltype,
                    histosweep_mask=histosweep_mask,
                    histosweep_patch_size=int(args.histosweep_patch_size),
                    h5_layout=str(args.h5_layout),
                    h5_compression=str(args.h5_compression),
                    h5_compression_level=int(args.h5_compression_level),
                )
                written.append(dataset_test_use)
                del output, major_prop_pred

    print(f"[predict_highres] wrote {len(written)} sample(s) into {out_dir}")
    return 0


def _lowres_prior_exists(lowres_pred_dir: Path, sample: str) -> bool:
    """The low-resolution stage output for one sample.

    The high-resolution model does not consume this file numerically, but the
    pipeline contract is that the refinement is always built on a completed
    low-resolution prediction, so its presence is required.
    """
    if not lowres_pred_dir.is_dir():
        return False
    return any(p.name.startswith(f"{sample}-pred") for p in lowres_pred_dir.iterdir())


def _write_sample(
    *,
    sample_name: str,
    output: Any,
    major_prop_pred: Any,
    father_patches: Dict[int, Any],
    patches_loc_before_pad: Dict[int, Any],
    patch_sizes: List[int],
    gene_list: List[str],
    major_cell_type: List[str],
    parameters: Dict[str, Any],
    out_dir: Path,
    uint8_output: bool,
    compress_output: bool,
    h5py: Any,
    np: Any,
    train_utils: Any,
    pred_celltype: bool,
    histosweep_mask: Any = None,
    histosweep_patch_size: int = 14,
    h5_layout: str = "spatial_gene_delta",
    h5_compression: str = "zstd",
    h5_compression_level: int = 9,
) -> None:
    get_unique_indices = train_utils.get_unique_indices
    as_smallest_int_array = train_utils.as_smallest_int_array
    minimum = min(patch_sizes)

    output_dedup: Dict[int, Any] = {}
    major_prop_dedup: Dict[int, Any] = {}
    patches_loc_dedup: Dict[int, Any] = {}
    father_unique_all: Dict[int, Any] = {}
    inverse_index_all: Dict[int, Any] = {}
    for patch_size_index, patch_size in enumerate(patch_sizes):
        if patch_size != minimum:
            father_unique, inverse_index = get_unique_indices(
                father_patches[patch_size].cpu(), device="cpu"
            )
            father_unique_all[patch_size] = father_unique.numpy()
            inverse_index_all[patch_size] = inverse_index.numpy()
            output_dedup[patch_size] = output[inverse_index.numpy(), patch_size_index, :]
            major_prop_dedup[patch_size] = major_prop_pred[inverse_index.numpy(), patch_size_index, :]
            patches_loc_dedup[patch_size] = patches_loc_before_pad[patch_size][father_unique.numpy(), :]
        else:
            output_dedup[patch_size] = output[:, patch_size_index, :]
            major_prop_dedup[patch_size] = major_prop_pred[:, patch_size_index, :]
            patches_loc_dedup[patch_size] = patches_loc_before_pad[patch_size]
        patches_loc_dedup[patch_size] = patches_loc_dedup[patch_size][:, [1, 0]]

    # Drop the points HistoSweep marked as background / low quality.  This has to
    # happen after the dedup and the flip: the mask is indexed (row=y, col=x) and
    # ``patches_loc_dedup`` is in exactly that (y, x) order at this point (it is
    # the same array saved as ``pos``).  Arrays and positions are filtered by the
    # same flag so they stay row-aligned.
    if histosweep_mask is not None:
        dropped = masks.apply_mask(
            output_dedup,
            major_prop_dedup,
            patches_loc_dedup,
            histosweep_mask,
            patch_sizes,
            patch_size_for_mask=histosweep_patch_size,
            np=np,
        )
        kept = sum(len(v) for v in output_dedup.values())
        print(
            f"    HistoSweep filtered {dropped} point(s); {kept} kept "
            f"({len(patch_sizes)} patch size(s))"
        )

    if not uint8_output:
        import pickle

        # NOTE: ``patches_loc_before_pad`` is written as the (y, x) dictionary here
        # because that is what this port has always done; ``aurora.predictions``
        # detects that the two keys agree and flips, so readers still see (x, y).
        # See aurora/predictions.py::_reconcile_positions.
        # ``cell_types`` is a NumPy array, matching code/predict_highres.py
        # (``pd.read_csv(...).values.squeeze()``); code/plot.py calls .tolist().
        payload = {
            "exp_pred": output_dedup,
            "log_prop_pred": major_prop_dedup,
            "pos": patches_loc_dedup,
            "patches_loc_before_pad": patches_loc_dedup,
            "genes": gene_list,
            "cell_types": np.asarray(major_cell_type),
            "resolutions": parameters.get("patch_size", patch_sizes),
        }
        # Same helper the stage's declared output uses, so the two cannot drift.
        path = highres_output_path(out_dir, sample_name, uint8_output, compress_output)
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)
        print(f"Prediction saved to {path}")
        return

    prop_probs_dedup = {ps: np.exp(arr) for ps, arr in major_prop_dedup.items()}
    exp_max = {ps: output_dedup[ps].max(axis=0) for ps in patch_sizes}
    prop_max = {ps: prop_probs_dedup[ps].max(axis=0) for ps in patch_sizes}
    exp_max_safe = {ps: np.where(exp_max[ps] > 0, exp_max[ps], 1.0) for ps in patch_sizes}
    prop_max_safe = {ps: np.where(prop_max[ps] > 0, prop_max[ps], 1.0) for ps in patch_sizes}
    exp_pred_uint8 = {
        ps: np.rint((output_dedup[ps] / exp_max_safe[ps] * 255.0).clip(0, 255)).astype(np.uint8)
        for ps in patch_sizes
    }
    prop_pred_uint8 = {
        ps: np.rint((prop_probs_dedup[ps] / prop_max_safe[ps] * 255.0).clip(0, 255)).astype(np.uint8)
        for ps in patch_sizes
    }
    exp_pred_mem = {ps: np.ascontiguousarray(exp_pred_uint8[ps]) for ps in patch_sizes}
    prop_pred_mem = {ps: np.ascontiguousarray(prop_pred_uint8[ps]) for ps in patch_sizes}
    exp_max_mem = {ps: exp_max[ps].astype(np.float32) for ps in patch_sizes}
    prop_max_mem = {ps: prop_max[ps].astype(np.float32) for ps in patch_sizes}
    patch_loc_mem = {
        ps: as_smallest_int_array(loc, allow_unsigned=True) for ps, loc in patches_loc_dedup.items()
    }

    if not compress_output:
        h5_path = highres_output_path(out_dir, sample_name, uint8_output, compress_output)
        # The compact layout from command.sh: rows sorted spatially and
        # delta-encoded, filtered with zstd.  aurora/h5out.py is a port of
        # code/predict_highres_TCGA_optimized.py, and aurora/h5in.py reads it back.
        h5out.write_highres_uint8_h5(
            h5_path,
            gene_list=gene_list,
            major_cell_type=major_cell_type,
            resolutions=parameters.get("patch_size", patch_sizes),
            patch_sizes=patch_sizes,
            exp_pred_mem=exp_pred_mem,
            prop_pred_mem=prop_pred_mem,
            exp_max_mem=exp_max_mem,
            prop_max_mem=prop_max_mem,
            patch_loc_mem=patch_loc_mem,
            layout=h5_layout,
            compression=h5_compression,
            compression_level=h5_compression_level,
        )
        print(f"HDF5 result saved to {h5_path} ({os.path.getsize(h5_path) / 1e6:.2f} MB)")
        return

    import io

    zstd = require("zstandard", purpose="compressing the high-resolution predictions")
    save_dict = {
        "genes": np.array(gene_list),
        "cell_types": np.array(major_cell_type),
        "resolutions": np.array(parameters.get("patch_size", patch_sizes)),
        "patch_sizes": np.array(patch_sizes),
    }
    for ps in patch_sizes:
        save_dict[f"exp_pred_{ps}"] = exp_pred_mem[ps]
        save_dict[f"prop_pred_{ps}"] = prop_pred_mem[ps]
        save_dict[f"exp_max_{ps}"] = exp_max_mem[ps]
        save_dict[f"prop_max_{ps}"] = prop_max_mem[ps]
        save_dict[f"patch_loc_{ps}"] = patch_loc_mem[ps]
        if ps != minimum:
            save_dict[f"father_unique_{ps}"] = father_unique_all[ps]
            save_dict[f"inverse_index_{ps}"] = inverse_index_all[ps]
    buffer = io.BytesIO()
    np.savez(buffer, **save_dict)
    buffer.seek(0)
    compressed = zstd.ZstdCompressor(level=18).compress(buffer.read())
    out_path = highres_output_path(out_dir, sample_name, uint8_output, compress_output)
    with open(out_path, "wb") as handle:
        handle.write(compressed)
    print(f"Compressed result saved to {out_path} ({len(compressed) / 1e6:.2f} MB)")


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main())
    except AuroraError as exc:
        print(f"error: {exc.format()}", file=sys.stderr)
        sys.exit(2)
