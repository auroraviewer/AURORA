#!/usr/bin/env bash
# =============================================================================
# AURORA unified code system -- end-to-end example
#
# This is the new-CLI equivalent of command.sh.  The original command.sh is
# untouched and still works with the legacy separate checkpoints.
#
# Everything is driven by one entry point:
#
#     python -m aurora <command>
#
# Run `python -m aurora --help`, `python -m aurora stages --packages` and
# `python -m aurora doctor` for the full picture.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
export PROJECT_PATH="${PROJECT_PATH:-.}"                 # holds the input data
export MODEL_PATH="${MODEL_PATH:-trained_model/AURORA_LUAD_combined.pth}"
export UNI_LOCAL_PATH="${UNI_LOCAL_PATH:-./uni2h}"       # local clone of UNI2-h
export SAMPLES="${SAMPLES:-examples/Inference_samples.csv}"
export BULK="${BULK:-examples/TCGA_LUAD_expression.csv}" # bulk expression matrix

# ---------------------------------------------------------------------------
# 1. Build the combined checkpoint (once)
#
#    The new code system reads ONE file that bundles the low-resolution and the
#    high-resolution model.  PyTorch is not needed for this step, and the two
#    original .pth files are only read -- never modified or removed.
# ---------------------------------------------------------------------------
build_combined_checkpoint() {
    python -u tools/merge_aurora_checkpoints.py \
        --lowres  trained_model/AURORA_LUAD_V2.pth \
        --highres trained_model/AURORA_highres_model_weights.pth \
        --lowres-params  trained_model/model_parameter_LUAD.json \
        --highres-params trained_model/model_parameter_LUAD_highres.json \
        --gene-list    trained_model/LUAD_gene.csv \
        --cell-types   trained_model/LUAD_cell_types.csv \
        --norm-factors trained_model/LUAD_gene_normalize_factor.csv \
        --bulk     "${BULK}" \
        --model-name AURORA_LUAD --cancer-type LUAD \
        --out "${MODEL_PATH}" --force

    python -u -m aurora validate --model "${MODEL_PATH}"
    python -u -m aurora inspect  --model "${MODEL_PATH}" --assets
}

# ---------------------------------------------------------------------------
# 2. Apply an existing model to new samples  (inference only)
#
#    Stages, in order (aurora works this out for you):
#        segment -> rescale -> features_lowres -> features_highres
#                -> predict_lowres -> predict_highres
#
#    The high-resolution model is NOT allowed to run on its own: it refines the
#    low-resolution prediction.  Asking for it first fails with a message that
#    tells you what to run; --auto-prerequisites runs it for you instead.
# ---------------------------------------------------------------------------
inference() {
    # 2a. The whole chain, in one command.
    python -u -m aurora run \
        --project "${PROJECT_PATH}" \
        --run luad_infer \
        --samples "${SAMPLES}" \
        --model "${MODEL_PATH}" \
        --image-path wsis \
        --preset examples/lung.csv \
        --uni-local-path "${UNI_LOCAL_PATH}" \
        --bulk "${BULK}"

    # 2b. See what happened, and where.
    python -u -m aurora status --run luad_infer
    #   aurora_runs/luad_infer/03_predictions/lowres/pred/<sample>-pred.pickle
    #   aurora_runs/luad_infer/03_predictions/highres/pred/<sample>-highres-pred.h5
    #   aurora_runs/luad_infer/logs/<stage>.log

    # 2c. Optional: figures (code/plot.py needs at least one gene).
    python -u -m aurora run --run luad_infer --stage plot \
        --gene IL7R KRT8 --celltype Malignant T_cell --auto-prerequisites
}

# ---------------------------------------------------------------------------
# 3. Run the steps one at a time (the step-wise form of the old command.sh)
#
#    Finished stages are reused, so you can re-run these safely.  Every stage
#    refuses to start if its prerequisites are missing, unless you pass
#    --auto-prerequisites.
# ---------------------------------------------------------------------------
inference_stepwise() {
    python -u -m aurora run --run luad_infer --stage segment
    python -u -m aurora run --run luad_infer --stage rescale
    python -u -m aurora run --run luad_infer --stage features_lowres --uni-local-path "${UNI_LOCAL_PATH}"
    python -u -m aurora run --run luad_infer --stage features_highres --uni-local-path "${UNI_LOCAL_PATH}"

    # The low-resolution prediction must exist before this one will start.
    python -u -m aurora run --run luad_infer --stage predict_lowres
    python -u -m aurora run --run luad_infer --stage predict_highres

    # What you get if you try the last one first:
    #   error: refusing to run stage 'predict_highres': ...
    #   The high-resolution model refines the low-resolution prediction, ...
}

# ---------------------------------------------------------------------------
# 4. Training
#
#    Stages, in order:
#        rescale -> features_lowres -> prepare_st -> smooth_cnts -> train_lowres
#                -> features_highres -> prepare_highres_training -> train_highres
#                -> combine_checkpoints
#
#    Needs the training dependencies:  pip install -r requirements/training.txt
# ---------------------------------------------------------------------------
training() {
    # 4a. Optional: turn raw Visium data into HEST-1K format.
    python -u -m aurora run --mode training --run lung_train \
        --samples examples/Training_samples.csv \
        --model "${MODEL_PATH}" \
        --stage preprocess --auto-prerequisites

    # 4b. Train both stages and merge them into one combined checkpoint.
    python -u -m aurora run --mode training --run lung_train \
        --samples examples/Training_samples.csv \
        --model "${MODEL_PATH}" \
        --processed-data processed_data \
        --auto-prerequisites

    # The result is what the inference part consumes:
    #   aurora_runs/lung_train/checkpoints/aurora_combined.pth
    python -u -m aurora validate --model aurora_runs/lung_train/checkpoints/aurora_combined.pth

    # 4c. Optional: cell-type deconvolution as a supervised signal.
    python -u -m aurora run --mode training --run lung_train --stage deconvolution \
        --scrna-reference Kim2020_Lung.rds --auto-prerequisites

    # 4d. Optional: fine-tune extra genes on top of a trained model.
    python -u -m aurora run --mode training --run lung_train --stage finetune \
        --finetune-gene-list examples/finetune_gene.csv --num-epochs 200 --auto-prerequisites

    # TensorBoard logs:
    #   tensorboard --logdir aurora_runs/lung_train/checkpoints
}

# ---------------------------------------------------------------------------
# 5. Move an old, flat output tree into the new layout
#
#    Copies only; the original directories are left exactly as they are.
# ---------------------------------------------------------------------------
migrate_old_outputs() {
    python -u -m aurora migrate-layout --project "${PROJECT_PATH}" --run migrated --dry-run
    python -u -m aurora migrate-layout --project "${PROJECT_PATH}" --run migrated
}

# ---------------------------------------------------------------------------
# 6. Checks
# ---------------------------------------------------------------------------
verify() {
    python -u -m aurora doctor
    python -u tools/audit_inference_deps.py
    python -u tools/sync_requirements.py --check
    python -u tests/selftest.py
}

# ---------------------------------------------------------------------------
# Pick what to run.
# ---------------------------------------------------------------------------
main() {
    local action="${1:-inference}"
    case "${action}" in
        combine)   build_combined_checkpoint ;;
        inference) inference ;;
        stepwise)  inference_stepwise ;;
        training)  training ;;
        migrate)   migrate_old_outputs ;;
        verify)    verify ;;
        help|--help|-h)
            cat <<'USAGE'
Usage: ./run.sh [combine|inference|stepwise|training|migrate|verify]

  combine    build the single combined checkpoint from the legacy pair
  inference  apply the model to new samples (whole chain)
  stepwise   the same, one stage at a time, showing the prerequisite gate
  training   train both stages and merge them into a new combined checkpoint
  migrate    copy an old flat output tree into the new run layout
  verify     doctor + dependency audit + requirements check + self-test
USAGE
            ;;
        *)
            echo "unknown action '${action}'; try ./run.sh help" >&2
            exit 2
            ;;
    esac
}

main "$@"
