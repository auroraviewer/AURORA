"""AURORA unified code system (new API).

Quick start
-----------
Apply an existing model to new samples (inference only)::

    python -m aurora run --run my_run --samples examples/Inference_samples.csv \\
        --model trained_model/AURORA_LUAD_combined.pth

Train a new model::

    python -m aurora run --mode training --run lung_train \\
        --samples examples/Training_samples.csv \\
        --model trained_model/AURORA_LUAD_combined.pth --auto-prerequisites

Importing this package never imports NumPy, PyTorch or pandas; every heavy
dependency is loaded on demand through :mod:`aurora.deps`, so ``import aurora``
works in a bare interpreter and the inference/training split stays honest.
"""

from __future__ import annotations

from .errors import (
    AuroraError,
    CheckpointFormatError,
    ConfigError,
    LayoutError,
    MissingDependencyError,
    SampleSheetError,
    StageFailedError,
    StagePrerequisiteError,
)
from .version import (
    BUNDLE_FORMAT,
    BUNDLE_FORMAT_VERSION,
    LAYOUT_FORMAT,
    LAYOUT_FORMAT_VERSION,
    RUN_FORMAT,
    RUN_FORMAT_VERSION,
    __version__,
)

__all__ = [
    "__version__",
    "AuroraError",
    "CheckpointFormatError",
    "ConfigError",
    "LayoutError",
    "MissingDependencyError",
    "SampleSheetError",
    "StageFailedError",
    "StagePrerequisiteError",
    "BUNDLE_FORMAT",
    "BUNDLE_FORMAT_VERSION",
    "RUN_FORMAT",
    "RUN_FORMAT_VERSION",
    "LAYOUT_FORMAT",
    "LAYOUT_FORMAT_VERSION",
]
