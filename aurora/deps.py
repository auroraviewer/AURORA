"""Dependency policy for the AURORA unified code system.

Why this module exists
----------------------
Requirement #3 of the reorganisation: a user who only wants to *apply* an
existing model to new samples must not have to install anything that is only
needed for *training*.

This module is the single place that decides which third-party package belongs
to which group, and it is used in three ways:

1. :func:`require` imports a package on demand and, when it is missing, raises
   :class:`~aurora.errors.MissingDependencyError` naming the exact
   ``requirements/*.txt`` file to install.  Nothing in the ``aurora`` package
   imports a heavy dependency at module import time, so ``import aurora`` works
   with a bare Python interpreter.
2. :func:`check_group` and the ``aurora doctor`` command report what is missing
   for a chosen group.
3. ``tools/audit_inference_deps.py`` reads :data:`DEPENDENCIES` to statically
   prove that the inference entry points never reach a training-only package.

Groups are cumulative: ``inference`` contains ``core``; ``training`` contains
``inference``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .errors import MissingDependencyError, format_error_list

# --------------------------------------------------------------------------- #
# Group definitions
# --------------------------------------------------------------------------- #

#: Human readable purpose of each group, used in error messages and ``doctor``.
GROUP_PURPOSE: Dict[str, str] = {
    "core": "always installed; the model code itself",
    "inference": "apply an existing model to new samples (no training)",
    "training": "train or fine-tune an AURORA model",
    "hest": "preprocess raw Visium data into HEST-1K format (training only)",
    "deconvolution": "cell-type deconvolution with RCTD (training only, R)",
    "histosweep": "HistoSweep tissue-quality masking before the high-resolution stage",
    "catching": "optional CATCH / pyvips based high-resolution feature backend",
    "dev": "running the test-suite and the parity/audit tools",
}

#: Requirements file that installs each group.
GROUP_REQUIREMENTS_FILE: Dict[str, str] = {
    "core": "requirements/core.txt",
    "inference": "requirements/inference.txt",
    "training": "requirements/training.txt",
    "hest": "requirements/training-hest.txt",
    "deconvolution": None,  # conda + R, see README
    "histosweep": "requirements/optional-histosweep.txt",
    "catching": "requirements/optional-catching.txt",
    "dev": "requirements/dev.txt",
}

#: Groups that inherit the packages of another group.
GROUP_PARENT: Dict[str, Optional[str]] = {
    "core": None,
    "inference": "core",
    "training": "inference",
    "hest": "training",
    "deconvolution": "training",
    "histosweep": "inference",
    "catching": "inference",
    "dev": "training",
}


@dataclass(frozen=True)
class Dependency:
    """One third-party package the code base may import."""

    #: Name passed to :func:`importlib.import_module`.  Empty for packages that
    #: only ship native libraries and expose no importable module.
    import_name: str
    #: Name used by pip / conda.
    distribution: str
    #: Lowest group that provides it.
    group: str
    #: Pipeline stages / modules that need it (documentation + audit).
    needed_by: Tuple[str, ...] = ()
    #: Minimum version written into the generated requirements files.
    min_version: str = ""
    #: Extra sentence shown when the package is missing.
    note: str = ""
    #: Alternative import names that also satisfy the requirement.
    aliases: Tuple[str, ...] = ()
    #: ``False`` when the package has no usable PyPI name and can only be
    #: installed from source.  Such a dependency is left out of the generated
    #: ``requirements/*.txt`` lines and out of the packaging extras, because pip
    #: would either fail or install an unrelated project of the same name.
    installable: bool = True
    #: Shell command that installs a non-installable dependency.
    source_command: str = ""
    #: ``True`` when importing the package is not enough to know it works -- it
    #: binds a native library at import time (``pyvips`` -> ``libvips``,
    #: ``openslide`` -> ``libopenslide``).  ``importlib.util.find_spec`` does not
    #: execute the module, so it happily reports such a package as available and
    #: the stage then dies with ``cannot load library 'libvips.so.42'``.  For these
    #: the check performs a real import.
    probe_import: bool = False
    #: How to obtain the native library a probed package needs.
    native_hint: str = ""

    @property
    def import_names(self) -> Tuple[str, ...]:
        if not self.import_name:
            return ()
        return (self.import_name,) + self.aliases

    @property
    def requirement_line(self) -> str:
        return f"{self.distribution}>={self.min_version}" if self.min_version else self.distribution


#: The dependency table.  Keep sorted by group then import name.
DEPENDENCIES: Tuple[Dependency, ...] = (
    # ---------------- core ---------------- #
    Dependency("numpy", "numpy", "core", ("all",), min_version="1.26"),
    Dependency("pandas", "pandas", "core", ("all",), min_version="2.2.3"),
    Dependency("PIL", "Pillow", "core", ("io", "segmentation"), min_version="10.0"),
    Dependency("tqdm", "tqdm", "core", ("all",), min_version="4.67.1"),
    Dependency("yaml", "PyYAML", "core", ("segmentation", "config"), min_version="6.0"),
    Dependency("h5py", "h5py", "core", ("predict_highres", "io"), min_version="3.10"),
    # The compact high-resolution HDF5 output is compressed with zstd by default
    # (see command.sh: --h5_compression zstd), and h5py cannot write or read that
    # filter without this plugin.  Without it the default predict_highres run
    # fails, so it belongs to the inference group rather than being optional.
    Dependency(
        "hdf5plugin",
        "hdf5plugin",
        "inference",
        ("predict_highres", "plot"),
        note="Provides the Zstd HDF5 filter used by the default high-resolution output.",
    ),
    Dependency(
        "torch",
        "torch",
        "core",
        ("all",),
        min_version="2.0",
        note="Install the build that matches your CUDA version, see https://pytorch.org/",
    ),
    # ---------------- inference ---------------- #
    Dependency("torchvision", "torchvision", "inference", ("features_lowres", "features_highres"), min_version="0.15"),
    Dependency(
        "timm",
        "timm",
        "inference",
        ("features_lowres", "features_highres"),
        min_version="1.0.14",
        note="Provides the UNI2-h backbone used for H&E feature extraction.",
    ),
    Dependency(
        "huggingface_hub",
        "huggingface_hub",
        "inference",
        ("features_lowres", "features_highres"),
        min_version="0.27.1",
        note="Only needed when UNI2-h is fetched from the Hugging Face hub.",
    ),
    Dependency("tifffile", "tifffile", "inference", ("rescale", "features_lowres"), min_version="2024.12.12"),
    Dependency("imagecodecs", "imagecodecs", "inference", ("rescale",), min_version="2024.6.1", note="tifffile compression codecs."),
    Dependency("cv2", "opencv-python", "inference", ("rescale", "features_lowres", "plot"), min_version="4.11.0"),
    Dependency(
        "openslide",
        "openslide-python",
        "inference",
        ("segment", "rescale"),
        min_version="1.4.1",
        note="Pair it with the 'openslide-bin' wheel if your system has no OpenSlide library.",
        # binds libopenslide at import time
        probe_import=True,
        native_hint=(
            "pip install openslide-bin   (ships the libraries), or install them yourself: "
            "apt-get install libopenslide0 / conda install -c conda-forge openslide"
        ),
    ),
    Dependency(
        "",
        "openslide-bin",
        "inference",
        ("segment", "rescale"),
        note="Ships the native OpenSlide libraries; no Python module to import.",
    ),
    Dependency("scipy", "scipy", "inference", ("features_highres", "predict_highres"), min_version="1.13"),
    Dependency(
        "rasterio",
        "rasterio",
        "inference",
        ("segment", "rescale"),
        min_version="1.4.1",
        note=(
            "Guarantees pixel-identical tissue masks to the original pipeline. "
            "The code falls back to a dependency-free even-odd rasteriser when it is absent."
        ),
    ),
    Dependency("matplotlib", "matplotlib", "inference", ("plot", "segment", "features_*"), min_version="3.8", note="Only for --plot and figures."),
    Dependency("zstandard", "zstandard", "inference", ("predict_highres",), min_version="0.22.0", note="Only for --compress-output."),
    # The histosweep stage is mandatory and part of the default inference chain,
    # and it imports `textureAnalysis`, which does `from sklearn.mixture import
    # GaussianMixture` at module level.  So scikit-learn is an *inference*
    # dependency; `training` inherits it (GROUP_PARENT), which is why the
    # training-only use is listed here too rather than declared twice.
    Dependency(
        "sklearn",
        "scikit-learn",
        "inference",
        ("histosweep", "train_lowres", "finetune"),
        min_version="1.3",
        note=(
            "HistoSweep's texture analysis needs it at import time; training uses it for "
            "the smooth-count targets."
        ),
    ),
    # Same reasoning: HistoSweep's computeMetrics and textureAnalysis do
    # `from numba import jit, prange` at module level.
    Dependency(
        "numba",
        "numba",
        "inference",
        ("histosweep",),
        note="HistoSweep's computeMetrics/textureAnalysis are JIT-compiled with numba.",
    ),
    # ---------------- training ---------------- #
    Dependency("scanpy", "scanpy", "training", ("prepare_st", "rescale_train", "smooth_cnts"), min_version="1.10.4"),
# smooth_cnt.py builds the spot-neighbour graph with squidpy before smoothing.
Dependency("squidpy", "squidpy", "training", ("smooth_cnts",), min_version="1.5"),
    Dependency("geopandas", "geopandas", "training", ("segment_train", "rescale_train"), min_version="1.0.1"),
    Dependency(
        "tensorboard",
        "tensorboard",
        "training",
        ("train_lowres", "train_highres", "finetune"),
        min_version="2.18.0",
        note="Imported through torch.utils.tensorboard for --log.",
    ),
    Dependency("pybiomart", "pybiomart", "training", ("gene_annotation",), min_version="0.2.0", note="TCGA protein-coding gene annotation."),
    Dependency(
        "torch_geometric",
        "torch_geometric",
        "training",
        ("train_GraphLSTM", "test_GraphLSTM"),
        note="Only the GCN / GraphLSTM model variants under code/ need it.",
    ),
    # ---------------- hest ---------------- #
    # pyvips is a thin cffi binding: `import pyvips` runs `ffi.dlopen('libvips')`,
    # so it can be installed and still be unusable without the native library.
    Dependency(
        "pyvips",
        "pyvips",
        "hest",
        ("hest_preprocess", "catching"),
        note="A Python binding -- also needs the native libvips library.",
        probe_import=True,
        native_hint=(
            "conda install -c conda-forge pyvips   (installs the library too), or "
            "apt-get install libvips-dev / brew install vips"
        ),
    ),
    # `hest` has no usable PyPI release: the project on PyPI under that name is not
    # MahmoodLab's HEST, so listing it as a requirement makes pip install the wrong
    # thing (or fail), and `pip install .[hest]` therefore cannot provide it.  It
    # is declared here so the dependency check can name it and explain the source
    # install, but it is marked non-installable so no requirements file or extra
    # ever promises it.
    Dependency(
        "hest",
        "hest",
        "hest",
        ("hest_preprocess",),
        note="Not on PyPI -- install from source.",
        installable=False,
        source_command=(
            "git clone https://github.com/mahmoodlab/HEST.git && pip install -e HEST"
        ),
    ),
    # ---------------- histosweep ---------------- #
    # HistoSweep sweeps the *rescaled H&E image* and writes a 0/1 tissue mask
    # (<sample>_HistoSweep_mask_rescaled.tif) that predict_highres applies to drop
    # low-quality / background patches.  It does not consume iStar output.
    #
    # scikit-image is declared in the *inference* group, not a group of its own,
    # because the histosweep stage is mandatory and part of the default inference
    # chain: `pip install .[inference]` has to be enough to run that chain, and a
    # per-stage extra that a user must discover would make the default run fail.
    # Its other imports (cv2, tifffile) already come from that group.
    Dependency(
        "skimage",
        "scikit-image",
        "inference",
        ("histosweep",),
        min_version="0.21",
        note="Used by the mandatory histosweep stage when masking tissue.",
    ),
)


def _group_chain(group: str) -> List[str]:
    """Return ``group`` and all of its ancestors, closest first."""
    if group not in GROUP_PARENT:
        raise KeyError(
            f"unknown dependency group '{group}'; known groups: {', '.join(sorted(GROUP_PARENT))}"
        )
    chain: List[str] = []
    current: Optional[str] = group
    while current is not None:
        chain.append(current)
        current = GROUP_PARENT[current]
    return chain


def dependencies_for_group(group: str, include_children: bool = False) -> List[Dependency]:
    """Return every :class:`Dependency` available in ``group``."""
    chain = set(_group_chain(group))
    if include_children:
        # add every group whose ancestor chain reaches one of ours
        for other in GROUP_PARENT:
            if chain & set(_group_chain(other)):
                chain.add(other)
    return [d for d in DEPENDENCIES if d.group in chain]


def requirements_file_for(group: str) -> Optional[str]:
    """Path (relative to the repository root) of the requirements file for a group."""
    return GROUP_REQUIREMENTS_FILE.get(group)


def is_module_available(import_name: str) -> bool:
    """True when ``import_name`` is *installed* in this interpreter.

    This uses ``find_spec``, which locates the package without executing it, so it
    says nothing about whether the import would succeed.  Use
    :func:`import_works` for a package that binds a native library.
    """
    if import_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


#: Why the last real import of each module failed, for the error message.
_import_errors: Dict[str, str] = {}


def import_works(import_name: str) -> bool:
    """Actually import the module.

    Needed for packages that bind a native library at import time: ``pyvips`` is
    installed and still raises ``OSError: cannot load library 'libvips.so.42'``
    when the library is absent.  ``find_spec`` cannot see that, so a stage would
    pass the dependency check and then crash on its first line.
    """
    if import_name in sys.modules:
        return True
    try:
        importlib.import_module(import_name)
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "unusable"
        # ImportError, the OSError cffi raises, or a RuntimeError from a broken
        # build.  Deliberately broad, but not BaseException: a Ctrl-C during a slow
        # import must not read as "package missing".
        _import_errors[import_name] = f"{type(exc).__name__}: {exc}".strip()
        return False


def import_error_for(import_name: str) -> str:
    """The recorded failure reason for ``import_name``, if it was probed."""
    return _import_errors.get(import_name, "")


def resolve_import_name(dep: Dependency) -> Optional[str]:
    """Return the first import name of ``dep`` that is usable, else ``None``.

    A dependency with no importable module (a native-library-only wheel such as
    ``openslide-bin``) always resolves to the empty string, meaning "present or
    not, it cannot be probed".
    """
    if not dep.import_names:
        return ""
    probe = import_works if dep.probe_import else is_module_available
    for name in dep.import_names:
        if probe(name):
            return name
    return None


@dataclass
class GroupCheck:
    """Result of :func:`check_group`."""

    group: str
    present: List[str] = field(default_factory=list)
    #: Packages of this group *and its ancestors* that are not installed.
    missing: List[str] = field(default_factory=list)
    #: Packages that belong to this group itself (not inherited) and are missing.
    missing_own: List[str] = field(default_factory=list)
    #: Packages inherited from an ancestor group that are missing.
    missing_inherited: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


def check_group(group: str) -> GroupCheck:
    """Report which packages of ``group`` (including inherited ones) are installed.

    Ancestors count: a group can only be complete when everything it inherits is
    complete, so a group never looks healthy while it inherits a missing package.
    """
    result = GroupCheck(group=group)
    for dep in dependencies_for_group(group):
        if resolve_import_name(dep) is not None:
            result.present.append(dep.distribution)
        else:
            result.missing.append(dep.distribution)
            if dep.group == group:
                result.missing_own.append(dep.distribution)
            else:
                result.missing_inherited.append(dep.distribution)
    result.missing = sorted(set(result.missing))
    result.missing_own = sorted(set(result.missing_own))
    result.missing_inherited = sorted(set(result.missing_inherited))
    return result


def _dependency_for(import_name: str) -> Optional[Dependency]:
    base = import_name.split(".")[0]
    for dep in DEPENDENCIES:
        if base in dep.import_names:
            return dep
    return None


def install_hint(import_name: str) -> Tuple[str, str]:
    """Return ``(group, hint_text)`` for a missing import."""
    dep = _dependency_for(import_name)
    if dep is None:
        return "unknown", (
            f"'{import_name}' is not part of the AURORA dependency policy. "
            "If it is a genuine new requirement, add it to aurora/deps.py so the "
            "inference/training split stays accurate."
        )
    group = dep.group
    req = requirements_file_for(group)
    lines = [f"'{dep.distribution}' belongs to the '{group}' group ({GROUP_PURPOSE.get(group, '')})."]
    if dep.note:
        lines.append(dep.note)
    failure = import_error_for(dep.import_name)
    if failure:
        # Show the real reason: "installed but cannot load libvips" is a very
        # different fix from "not installed".
        lines.append(f"It is installed but cannot be imported: {failure}")
    if not dep.installable:
        # A requirements file cannot provide this one, so do not point at it.
        if dep.source_command:
            lines.append("Install it from source instead:")
            lines.append(f"  {dep.source_command}")
        elif req:
            lines.append(f"See {req} for how to install it.")
    elif req:
        lines.append(f"Install it with:  pip install -r {req}")
    elif group == "deconvolution":
        lines.append("See the 'Cell type deconvolution' section of the README for the conda/R commands.")
    if dep.native_hint and (not dep.installable or dep.probe_import):
        # Installing the Python package alone does not provide the native library.
        lines.append(f"It needs a native library as well: {dep.native_hint}")
    if group in ("inference", "core") and dep.needed_by:
        lines.append(f"Needed by: {', '.join(dep.needed_by)}")
    return group, "\n".join(lines)


def require(import_name: str, purpose: str = "") -> ModuleType:
    """Import ``import_name``, raising a helpful error when it is unavailable.

    Use this instead of a bare ``import`` for anything that is not part of the
    standard library, so that a missing *training* package never breaks an
    *inference* run and the message tells the user exactly what to install.
    """
    try:
        return importlib.import_module(import_name)
    except ImportError as exc:  # pragma: no cover - depends on environment
        group, hint = install_hint(import_name)
        where = f" while running {purpose}" if purpose else ""
        raise MissingDependencyError(
            f"missing required package '{import_name}'{where}: {exc}",
            hint=hint,
        ) from exc


def require_any(import_names: Sequence[str], purpose: str = "") -> ModuleType:
    """Import the first available name from ``import_names``."""
    last_error: Optional[Exception] = None
    for name in import_names:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            last_error = exc
    group, hint = install_hint(import_names[0])
    where = f" while running {purpose}" if purpose else ""
    raise MissingDependencyError(
        f"missing required package (tried {', '.join(import_names)}){where}: {last_error}",
        hint=hint,
    )


def optional(import_name: str) -> Optional[ModuleType]:
    """Import ``import_name`` if present, otherwise return ``None``.

    Never raises: use it for genuinely optional accelerators whose absence the
    code handles explicitly.
    """
    try:
        return importlib.import_module(import_name)
    except ImportError:
        return None


def format_missing(check: GroupCheck) -> str:
    """Render a :class:`GroupCheck` failure as a printable block."""
    req = requirements_file_for(check.group)
    lines = [
        f"The '{check.group}' dependency group is incomplete.",
        f"Purpose of this group: {GROUP_PURPOSE.get(check.group, '')}",
        "Missing packages:",
        format_error_list(sorted(set(check.missing))),
    ]
    if req:
        lines.append("")
        lines.append(f"Install everything for this group with:\n  pip install -r {req}")
    return "\n".join(lines)


def group_of(import_name: str) -> str:
    """Return the group a top-level import name belongs to (``'unknown'`` if new)."""
    dep = _dependency_for(import_name)
    return dep.group if dep else "unknown"


def all_import_names() -> Iterable[str]:
    """Every third-party top-level import name the policy knows about."""
    for dep in DEPENDENCIES:
        for name in dep.import_names:
            yield name
