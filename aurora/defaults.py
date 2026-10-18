"""Default training parameters, used when no checkpoint is supplied.

A from-scratch training run needs the low- and high-resolution models configured,
but nothing in it loads the weights of an existing model: ``code/train.py`` and
``code/train_highres.py`` build their networks and save their own ``.pth`` files,
and the only ``load_state_dict`` in the chain (``prepare_highres_training.py``)
loads the *low-resolution model this run just trained*, from
``<run>/checkpoints/``.

So the combined checkpoint contributes only configuration: the values below, plus
the three assets (gene list, cell types, normalisation factors).  Supplying those
three with ``--gene-list/--cell-types/--norm-factors`` makes the checkpoint
optional, and these defaults fill in the parameter files.

The values mirror ``trained_model/model_parameter_LUAD.json`` and
``model_parameter_LUAD_highres.json``; the three asset-path keys are omitted
because they are generated from the run itself (see ``prepare.stage_params``).
"""

from __future__ import annotations

from typing import Any, Dict

#: Defaults for ``code/train.py`` (low resolution).
#:
#: ``num_epochs`` runs the expression head for ``exp_epochs`` epochs and then the
#: cell-type head for the remaining ones: 400 + 200 here.  The values mirror
#: ``trained_model/model_parameter_LUAD.json``; only the three asset-path keys are
#: omitted, because those are generated per run (see ``prepare.stage_params``).
DEFAULT_LOWRES_PARAMS: Dict[str, Any] = {
    "use_bulk": True,
    "pred_celltype": True,
    "bidirectional": False,
    "retrain_cellprop_only": False,
    "use_context_loss": True,
    "pixel_size": 0.5,
    "num_workers": 4,
    "batch_size": 4,
    "img_emb_dim": 1536,
    "patch_size": [3584, 896, 224],
    "mse_weight": 10,
    "prop_weight": 0.1,
    "focal_gamma": 0.5,
    "num_epochs": 600,
    "exp_epochs": 400,
    "lr": 0.001,
    "lr_prop": 0.001,
    "weight_decay": 0.00001,
    "stepLR_step_exp": 200,
    "stepLR_gamma_exp": 0.1,
    "stepLR_step_prop": 50,
    "stepLR_gamma_prop": 0.5,
}

#: Defaults for ``code/train_highres.py`` and ``code/prepare_highres_training.py``.
#: Same two phases: 100 epochs of expression, then 100 of cell type.  Mirrors
#: ``trained_model/model_parameter_LUAD_highres.json``.
DEFAULT_HIGHRES_PARAMS: Dict[str, Any] = {
    "use_bulk": True,
    "pred_celltype": True,
    "bidirectional": False,
    "retrain_cellprop_only": False,
    "use_context_loss": True,
    "pixel_size": 0.5,
    "num_workers": 4,
    "batch_size": 8,
    "img_emb_dim": 3072,
    "patch_size": [56, 14],
    "mse_weight": 10,
    "prop_weight": 0.1,
    "focal_gamma": 0.5,
    "num_epochs": 200,
    "exp_epochs": 100,
    "lr": 0.001,
    "lr_prop": 0.001,
    "weight_decay": 0.00001,
    "stepLR_step_exp": 50,
    "stepLR_gamma_exp": 0.1,
    "stepLR_step_prop": 50,
    "stepLR_gamma_prop": 0.5,
    "superpatch_size": 3000,
    "superpatch_stride": 2000,
    "superpatch_minsize": 3000,
    "superpatch_minlowres_patch": 50,
}
