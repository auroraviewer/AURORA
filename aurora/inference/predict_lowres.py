"""Low-resolution prediction stage (native, combined-checkpoint only).

This is a faithful port of ``code/predict.py``: the data assembly, the dedup
logic and the saved keys are identical, so results are unchanged.  The three
deliberate differences are:

1. the weights come from the **combined** checkpoint
   (:class:`aurora.bundle.CombinedCheckpoint`), never from a loose
   ``AURORA_*.pth``;
2. the gene list, the cell types and the bulk normalisation factors come from
   the checkpoint's assets, so no loose CSV files are needed;
3. the stage gate runs first, which is what stops the high-resolution model
   from being run on its own (see ``aurora/inference/predict_highres.py``).

Run it as ``python -m aurora.inference.predict_lowres --help``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._legacy import ensure_code_on_path
from ..checkpoint import load_assets, load_stage_model
from ..deps import require
from ..errors import AuroraError, ConfigError
from . import common


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aurora.inference.predict_lowres",
        description="Predict gene expression and cell-type proportions with the low-resolution model.",
    )
    common.add_lowres_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    torch = require("torch", purpose="running the low-resolution prediction")
    np = require("numpy", purpose="running the low-resolution prediction")
    pd = require("pandas", purpose="running the low-resolution prediction")
    tqdm = require("tqdm", purpose="running the low-resolution prediction").tqdm

    layout = common.load_run_layout(args.run_root)
    checkpoint = common.open_checkpoint(args.model)
    sheet = common.read_sample_sheet_for_run(layout.root, args.samples)
    if args.limit:
        sheet = sheet.with_names(sheet.sample_names[: args.limit])

    device = common.resolve_device(args.device)
    assets = load_assets(checkpoint, search_dirs=[layout.root, layout.config_dir, Path(args.model).parent])

    # --- requirement #2 gate ------------------------------------------------ #
    common.enforce_prerequisites(
        layout,
        checkpoint,
        sheet,
        assets,
        "predict_lowres",
        options={"device": str(device), "mode": "inference"},
    )

    stage_model = load_stage_model(checkpoint, "lowres", device=device)
    parameters = stage_model.params
    gene_list: List[str] = list(stage_model.gene_list)
    major_cell_type: List[str] = list(stage_model.cell_types)
    bulk_norm_factor = np.asarray(stage_model.norm_factors, dtype=np.float64)

    patch_sizes = stage_model.patch_sizes
    if not patch_sizes:
        raise ConfigError("the low-resolution stage of the checkpoint does not record patch sizes")
    output_size = len(gene_list)
    input_size = stage_model.img_emb_dim
    bidirectional = bool(parameters.get("bidirectional", False))
    use_bulk = bool(parameters.get("use_bulk", True))

    interim_dir = args.interim_dir
    pred_dir = args.pred_dir
    emb_folder = f"{interim_dir}/UNI_multiscale_patches"
    cache_dir = f"{pred_dir}/processed_pkl"
    out_dir = Path(layout.root) / pred_dir / "pred"
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path(layout.root) / cache_dir).mkdir(parents=True, exist_ok=True)

    bulk = common.resolve_bulk_matrix(
        checkpoint, sheet, layout.root, args.bulk, args.no_bulk or not use_bulk
    )
    if bulk is None and use_bulk:
        raise ConfigError(
            "the checkpoint was trained with bulk conditioning but no bulk matrix is available"
        )
    if bulk is None:
        # Deterministic zero matrix keeps the code path identical without
        # pretending we have signal.
        bulk = pd.DataFrame(index=[], columns=sheet.bulk_sample_names)

    ensure_code_on_path()
    import train_utils  # noqa: E402  (legacy numerical helpers)

    dataset_test = np.asarray(sheet.sample_names)
    bulk_sample_names = np.asarray(sheet.bulk_sample_names)

    dataset_dir = str(Path(layout.root) / cache_dir)
    if os.path.exists(dataset_dir):
        dataset_exist = [x.replace(".pkl", "") for x in os.listdir(dataset_dir) if x.endswith(".pkl")]
    else:
        dataset_exist = []

    common.describe_plan(
        "predict_lowres",
        sheet,
        device,
        extra=[
            f"checkpoint       : {checkpoint.path}",
            f"model            : {checkpoint.model_id} (lowres)",
            f"genes / celltypes: {output_size} / {len(major_cell_type)}",
            f"patch sizes      : {patch_sizes}",
            f"output           : {out_dir}  ({args.output_format} layout)",
        ],
    )

    model = stage_model.model
    model.eval()

    sample_num = max(1, int(args.batch_size))
    idx_total = len(dataset_test) // sample_num + 1
    written: List[str] = []

    for idx in range(idx_total):
        idx_end = min(len(dataset_test), (idx + 1) * sample_num)
        dataset_test_use = list(dataset_test[idx * sample_num : idx_end])
        bulk_sample_names_use = list(bulk_sample_names[idx * sample_num : idx_end])
        if not dataset_test_use:
            continue
        if not args.force:
            pending = [
                name
                for name in dataset_test_use
                if not common.expected_output(out_dir, name, highres=False).exists()
            ]
            if not pending:
                print(f"All {len(dataset_test_use)} sample(s) in this batch already predicted, skipping.")
                continue

        with torch.no_grad():
            if set(dataset_test_use) <= set(dataset_exist):
                print("Samples loading from disk")
                test_data = train_utils.load_data_pred(
                    dataset_test_use,
                    bulk_sample_names_use,
                    bulk,
                    main_dir=str(layout.root),
                    bulk_hvg=gene_list,
                    bulk_norm_factor=bulk_norm_factor,
                    interim_dir=interim_dir,
                    patch_sizes=patch_sizes,
                    from_file=True,
                    file_dir=dataset_dir,
                    num_workers=int(args.num_workers),
                )
            else:
                print("Samples loading from scratch")
                test_data = train_utils.load_data_pred(
                    dataset_test_use,
                    bulk_sample_names_use,
                    bulk,
                    main_dir=str(layout.root),
                    bulk_hvg=gene_list,
                    bulk_norm_factor=bulk_norm_factor,
                    interim_dir=interim_dir,
                    patch_sizes=patch_sizes,
                    from_file=False,
                    file_dir=dataset_dir,
                    num_workers=int(args.num_workers),
                )

            for index in tqdm(np.arange(len(test_data)), desc="Evaluation", unit="batch", leave=False):
                emb, bulk_values, bulk_rank, father_patches, patches_loc_before_pad = test_data[index]
                for key in emb.keys():
                    emb[key] = torch.tensor(emb[key], dtype=torch.float32).to(device)
                for key in father_patches.keys():
                    father_patches[key] = torch.tensor(father_patches[key], dtype=torch.int).to(device)
                bulk_values = torch.tensor(bulk_values, dtype=torch.float32).to(device)
                bulk_rank_t = torch.tensor(bulk_rank, dtype=torch.float32).to(device)

                # Same stacking order as code/predict.py: [coarse, mid, fine].
                emb_hierrachy = torch.stack(
                    [
                        emb[patch_sizes[0]][father_patches[patch_sizes[0]], :],
                        emb[patch_sizes[1]][father_patches[patch_sizes[1]], :],
                        emb[patch_sizes[2]],
                    ],
                    dim=1,
                )
                output, major_prop_pred = model(emb_hierrachy, bulk_rank_t)
                output = output.cpu().numpy()
                major_prop_pred = major_prop_pred.detach().cpu().numpy()

                sample_name = dataset_test_use[index]
                written.append(sample_name)
                _write_sample(
                    sample_name=sample_name,
                    output=output,
                    major_prop_pred=major_prop_pred,
                    father_patches={k: v.cpu() for k, v in father_patches.items()},
                    patches_loc_before_pad=patches_loc_before_pad,
                    patch_sizes=patch_sizes,
                    gene_list=gene_list,
                    major_cell_type=major_cell_type,
                    parameters=parameters,
                    out_dir=out_dir,
                    output_format=args.output_format,
                    uint8_output=bool(args.uint8_output),
                    only_224=bool(args.only_224),
                    np=np,
                )
                del output, major_prop_pred

    print(f"[predict_lowres] wrote {len(written)} sample(s) into {out_dir}")
    return 0


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
    output_format: str,
    uint8_output: bool,
    only_224: bool,
    np: Any,
) -> None:
    """Persist one sample, in either the modern or the legacy layout."""
    from .._legacy import import_legacy

    train_utils = import_legacy("train_utils")
    get_unique_indices = train_utils.get_unique_indices
    as_smallest_int_array = train_utils.as_smallest_int_array

    resolutions = [int(p) for p in parameters.get("patch_size", patch_sizes)]
    minimum = min(patch_sizes)

    # code/predict.py builds ``major_cell_type`` with
    # ``pd.read_csv(...).values.squeeze()``, i.e. a NumPy array, and code/plot.py
    # relies on that: it calls ``results['cell_types'].tolist().index(...)``.
    # ``genes`` stays a list, because plot.py calls ``results['genes'].index(...)``.
    # Writing a list here makes the cell-type plotting branch fail with
    # "AttributeError: 'list' object has no attribute 'tolist'".
    cell_types_payload = np.asarray(major_cell_type)

    # ---- legacy (3-D array) payload ------------------------------------- #
    legacy_payload: Optional[Dict[str, Any]] = None
    if output_format == common.FORMAT_LEGACY:
        legacy_payload = {
            "exp_pred": output,
            "prop_pred": np.exp(major_prop_pred),
            # load_data_pred already returns (x, y) ordering, which is what the
            # legacy reader (plot.py) expects -- do NOT flip here.
            "patches_loc_before_pad": {ps: patches_loc_before_pad[ps] for ps in patch_sizes},
            "genes": gene_list,
            "cell_types": cell_types_payload,
            "resolutions": resolutions,
        }

    # ---- modern (dict-of-patch-sizes) payload --------------------------- #
    output_dedup: Dict[int, Any] = {}
    major_prop_dedup: Dict[int, Any] = {}
    patches_loc_dedup: Dict[int, Any] = {}
    for patch_size_index, patch_size in enumerate(patch_sizes):
        if patch_size != minimum:
            father_unique, inverse_index = get_unique_indices(
                father_patches[patch_size].cpu(), device="cpu"
            )
            output_dedup[patch_size] = output[inverse_index.numpy(), patch_size_index, :]
            major_prop_dedup[patch_size] = major_prop_pred[inverse_index.numpy(), patch_size_index, :]
            patches_loc_dedup[patch_size] = patches_loc_before_pad[patch_size][
                father_unique.numpy(), :
            ]
        else:
            output_dedup[patch_size] = output[:, patch_size_index, :]
            major_prop_dedup[patch_size] = major_prop_pred[:, patch_size_index, :]
            patches_loc_dedup[patch_size] = patches_loc_before_pad[patch_size]
        patches_loc_dedup[patch_size] = patches_loc_dedup[patch_size][:, [1, 0]]

    # ``pos`` is stored in (y, x) to match code/predict.py exactly, but the
    # legacy key ``patches_loc_before_pad`` must stay in (x, y): code/plot.py
    # flips it a second time before using index 0 as y.  Storing the same
    # (y, x) dictionary under both names -- as an earlier revision did -- makes
    # the plots transposed.
    positions_xy = {ps: patches_loc_before_pad[ps] for ps in patch_sizes}
    positions_yx = {ps: patches_loc_dedup[ps] for ps in patch_sizes}

    # ``--only-224`` keeps just the finest low-resolution level, exactly as
    # code/predict.py does: `ps_use = [224]` and `resolutions = [224]`.  The model
    # still consumes all three embedding levels; only the saved output is narrowed.
    if only_224:
        if 224 not in output_dedup:
            raise ConfigError(
                f"--only-224 was requested but the model has no 224 px level "
                f"(levels: {sorted(output_dedup)})",
                hint="Pass --no-only-224 to keep every level this model produces.",
            )
        saved_sizes = [224]
    else:
        saved_sizes = list(patch_sizes)

    output_dedup = {ps: output_dedup[ps] for ps in saved_sizes}
    major_prop_dedup = {ps: major_prop_dedup[ps] for ps in saved_sizes}
    positions_xy = {ps: positions_xy[ps] for ps in saved_sizes}
    positions_yx = {ps: positions_yx[ps] for ps in saved_sizes}
    saved_resolutions = [224] if only_224 else list(parameters.get("patch_size", patch_sizes))

    if output_format == common.FORMAT_LEGACY:
        legacy_payload["exp_pred"] = output_dedup
        legacy_payload["prop_pred"] = {ps: np.exp(major_prop_dedup[ps]) for ps in saved_sizes}
        legacy_payload["patches_loc_before_pad"] = positions_xy
        legacy_payload["resolutions"] = saved_resolutions
        legacy_payload["exp_pred_by_resolution"] = output_dedup
        legacy_payload["log_prop_pred_by_resolution"] = major_prop_dedup
        legacy_payload["pos_by_resolution"] = positions_yx
        common.write_prediction_pickle(legacy_payload, out_dir / f"{sample_name}-pred.pickle")
        return

    if not uint8_output:
        common.write_prediction_pickle(
            {
                "exp_pred": output_dedup,
                "log_prop_pred": major_prop_dedup,
                "pos": positions_yx,
                "patches_loc_before_pad": positions_xy,
                "genes": gene_list,
                "cell_types": cell_types_payload,
                "resolutions": saved_resolutions,
            },
            out_dir / f"{sample_name}-pred.pickle",
        )
        return

    prop_probs_dedup = {ps: np.exp(arr) for ps, arr in major_prop_dedup.items()}
    exp_max = {ps: output_dedup[ps].max(axis=0) for ps in saved_sizes}
    prop_max = {ps: prop_probs_dedup[ps].max(axis=0) for ps in saved_sizes}
    exp_max_safe = {ps: np.where(exp_max[ps] > 0, exp_max[ps], 1.0) for ps in saved_sizes}
    prop_max_safe = {ps: np.where(prop_max[ps] > 0, prop_max[ps], 1.0) for ps in saved_sizes}
    exp_pred_uint8 = {
        ps: np.rint((output_dedup[ps] / exp_max_safe[ps] * 255.0).clip(0, 255)).astype(np.uint8)
        for ps in saved_sizes
    }
    prop_pred_uint8 = {
        ps: np.rint((prop_probs_dedup[ps] / prop_max_safe[ps] * 255.0).clip(0, 255)).astype(np.uint8)
        for ps in saved_sizes
    }
    exp_pred_mem = {ps: np.ascontiguousarray(exp_pred_uint8[ps]) for ps in saved_sizes}
    prop_pred_mem = {ps: np.ascontiguousarray(prop_pred_uint8[ps]) for ps in saved_sizes}
    exp_max_mem = {ps: exp_max[ps].astype(np.float32) for ps in saved_sizes}
    prop_max_mem = {ps: prop_max[ps].astype(np.float32) for ps in saved_sizes}
    patch_loc_mem = {
        ps: as_smallest_int_array(patches_loc_dedup[ps], allow_unsigned=True) for ps in saved_sizes
    }
    common.write_prediction_pickle(
        {
            "exp_pred": exp_pred_mem,
            "prop_pred": prop_pred_mem,
            "exp_max": exp_max_mem,
            "prop_max": prop_max_mem,
            "pos": patch_loc_mem,
            "genes": gene_list,
            "cell_types": cell_types_payload,
            "resolutions": saved_resolutions,
        },
        out_dir / f"{sample_name}-pred.pickle",
    )


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main())
    except AuroraError as exc:
        print(f"error: {exc.format()}", file=sys.stderr)
        sys.exit(2)
