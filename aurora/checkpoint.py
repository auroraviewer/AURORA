"""Building AURORA modules from a combined checkpoint.

Requirement #1: the new code system is *only* compatible with the combined
checkpoint.  There is deliberately no code path here that accepts a loose
low-resolution / high-resolution ``.pth`` pair -- :class:`aurora.bundle.CombinedCheckpoint`
raises :class:`~aurora.errors.CheckpointFormatError` instead, with the command
that merges the old files.

Everything that needs NumPy/PyTorch is imported lazily so that
``python -m aurora inspect --model x.pth`` works in a bare environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import assets as asset_io
from ._legacy import import_legacy
from .bundle import CombinedCheckpoint, StageEntry
from .deps import require
from .errors import CheckpointFormatError, ConfigError

#: Asset keys the checkpoint is expected to carry.
ASSET_GENE_LIST = "gene_list"
ASSET_CELL_TYPES = "cell_types"
ASSET_NORM_FACTORS = "gene_norm_factor"
ASSET_PARAMS_LOWRES = "params_lowres"
ASSET_PARAMS_HIGHRES = "params_highres"

#: Filenames to look for on disk when the checkpoint does not bundle an asset.
FALLBACK_FILENAMES: Dict[str, Sequence[str]] = {
    ASSET_GENE_LIST: ("Gene_to_predict.csv", "genes.csv"),
    ASSET_CELL_TYPES: ("cell_types.csv",),
    ASSET_NORM_FACTORS: ("Gene_normalize_factor.csv",),
}


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #


@dataclass
class CheckpointAssets:
    """Everything a prediction run needs besides the tensors themselves."""

    gene_list: List[str]
    cell_types: List[str]
    norm_factors: List[float]
    sources: Dict[str, str] = field(default_factory=dict)

    @property
    def num_genes(self) -> int:
        return len(self.gene_list)

    @property
    def num_cell_types(self) -> int:
        return len(self.cell_types)


def _find_on_disk(filename: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _asset_source(
    checkpoint: CombinedCheckpoint,
    key: str,
    filename: Optional[str],
    search_dirs: Sequence[Path],
) -> Tuple[Optional[bytes], Optional[Path]]:
    if key in checkpoint.manifest.assets:
        return checkpoint.asset_bytes(key), None
    if filename:
        found = _find_on_disk(Path(filename).name, search_dirs)
        if found is not None:
            return None, found
    for candidate_name in FALLBACK_FILENAMES.get(key, ()):
        found = _find_on_disk(candidate_name, search_dirs)
        if found is not None:
            return None, found
    return None, None


def _read_asset(
    checkpoint: CombinedCheckpoint,
    key: str,
    filename: Optional[str],
    search_dirs: Sequence[Path],
    description: str,
) -> Any:
    data, path = _asset_source(checkpoint, key, filename, search_dirs)
    if data is None and path is None:
        raise CheckpointFormatError(
            f"the combined checkpoint {checkpoint.path} does not bundle the {description} asset "
            f"and no file named {filename or FALLBACK_FILENAMES.get(key, ('?',))[0]!r} was found",
            hint=(
                "Rebuild the checkpoint including the assets:\n"
                "  python tools/merge_aurora_checkpoints.py --lowres … --highres … "
                "--gene-list trained_model/LUAD_gene.csv "
                "--cell-types trained_model/LUAD_cell_types.csv "
                "--norm-factors trained_model/LUAD_gene_normalize_factor.csv "
                "--out trained_model/AURORA_LUAD_combined.pth"
            ),
        )
    return data if data is not None else path


def load_assets(
    checkpoint: CombinedCheckpoint,
    search_dirs: Sequence[Path] = (),
) -> CheckpointAssets:
    """Load the gene list, cell types and normalisation factors."""
    dirs = [Path(d) for d in search_dirs]
    params = checkpoint.manifest.params_for("lowres") or checkpoint.manifest.params_for("highres")

    gene_source = _read_asset(
        checkpoint,
        ASSET_GENE_LIST,
        str(params.get("gene_list_file", "") or "") or None,
        dirs,
        "gene list",
    )
    cell_source = _read_asset(
        checkpoint,
        ASSET_CELL_TYPES,
        str(params.get("celltype_file", "") or "") or None,
        dirs,
        "cell-type list",
    )
    norm_source = _read_asset(
        checkpoint,
        ASSET_NORM_FACTORS,
        str(params.get("gene_norm_factor_file", "") or "") or None,
        dirs,
        "bulk normalisation factor",
    )

    gene_list = asset_io.read_gene_list(gene_source)
    cell_types = asset_io.read_cell_types(cell_source)
    norm_factors = asset_io.read_norm_factors(norm_source, gene_list)
    return CheckpointAssets(
        gene_list=gene_list,
        cell_types=cell_types,
        norm_factors=norm_factors,
        sources={
            ASSET_GENE_LIST: str(_describe_source(checkpoint, ASSET_GENE_LIST, gene_source)),
            ASSET_CELL_TYPES: str(_describe_source(checkpoint, ASSET_CELL_TYPES, cell_source)),
            ASSET_NORM_FACTORS: str(_describe_source(checkpoint, ASSET_NORM_FACTORS, norm_source)),
        },
    )


def _describe_source(checkpoint: CombinedCheckpoint, key: str, source: Any) -> str:
    if key in checkpoint.manifest.assets:
        return f"bundled:{checkpoint.manifest.assets[key]}"
    if isinstance(source, Path):
        return f"file:{source}"
    return "unknown"


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #


@dataclass
class StageModel:
    """A ready-to-run AURORA module for one stage of the combined checkpoint."""

    stage: str
    entry: StageEntry
    params: Dict[str, Any]
    gene_list: List[str]
    cell_types: List[str]
    norm_factors: List[float]
    state_dict: Mapping[str, Any]
    model: Any  # torch.nn.Module

    @property
    def patch_sizes(self) -> List[int]:
        sizes = self.entry.patch_sizes or self.params.get("patch_size") or self.params.get("patch_sizes")
        return [int(x) for x in (sizes or [])]

    @property
    def img_emb_dim(self) -> int:
        if self.entry.img_emb_dim is not None:
            return int(self.entry.img_emb_dim)
        value = self.params.get("img_emb_dim")
        if value is None:
            raise ConfigError(f"stage '{self.stage}' does not declare img_emb_dim")
        return int(value)

    def to(self, device: Any) -> "StageModel":
        self.model.to(device)
        return self

    def eval(self) -> "StageModel":
        self.model.eval()
        return self


def _module_class():
    """Import the legacy :class:`AURORA` model class."""
    model_module = import_legacy("model")
    return model_module.AURORA


def build_aurora(
    stage: str,
    params: Mapping[str, Any],
    gene_list: Sequence[str],
    cell_types: Sequence[str],
    device: Any = "cpu",
) -> Any:
    """Instantiate an untrained :class:`AURORA` matching ``stage``'s parameters."""
    torch = require("torch", purpose="building the AURORA model")
    AURORA = _module_class()
    input_size = int(params.get("img_emb_dim", 1536))
    output_size = len(gene_list)
    module = AURORA(
        input_size,
        output_size,
        bulk_size=len(gene_list),
        celltype_num=len(cell_types),
        latent_size=int(params.get("latent_size", 256)),
        device=device,
        bidirectional=bool(params.get("bidirectional", False)),
        use_bulk=bool(params.get("use_bulk", True)),
        pred_celltype=bool(params.get("pred_celltype", True)),
    )
    return module


def load_stage_model(
    checkpoint: CombinedCheckpoint,
    stage: str,
    device: Any = None,
    search_dirs: Sequence[Path] = (),
    strict: bool = True,
) -> StageModel:
    """Load one stage of a combined checkpoint into a ready-to-run module."""
    torch = require("torch", purpose=f"loading the '{stage}' model")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entry = checkpoint.stage(stage)
    params = dict(entry.params)
    assets = load_assets(checkpoint, search_dirs)
    model = build_aurora(stage, params, assets.gene_list, assets.cell_types, device=device)
    # Both published checkpoints were saved from CUDA, so map_location is
    # mandatory on CPU-only machines.
    state_dict = checkpoint.load_stage_state_dict(stage, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise CheckpointFormatError(
            f"the '{stage}' weights in {checkpoint.path} do not match the model described by "
            f"the checkpoint manifest: {exc}",
            hint=(
                "The manifest records img_emb_dim="
                f"{params.get('img_emb_dim')} and patch_size={params.get('patch_size')}. "
                "Rebuild the combined checkpoint so the parameters and the weights come from "
                "the same training run."
            ),
        ) from exc
    model.to(device)
    model.eval()
    return StageModel(
        stage=stage,
        entry=entry,
        params=params,
        gene_list=assets.gene_list,
        cell_types=assets.cell_types,
        norm_factors=assets.norm_factors,
        state_dict=state_dict,
        model=model,
    )


def summarise_state_dict(state_dict: Mapping[str, Any]) -> List[str]:
    """Human-readable ``key: shape`` lines, used by ``aurora inspect``."""
    lines: List[str] = []
    for key, value in state_dict.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            lines.append(f"    {key}: {tuple(shape)} {dtype}")
        else:
            lines.append(f"    {key}: {type(value).__name__}")
    return lines


def expected_state_dict_keys(stage: str, gene_list: Sequence[str], cell_types: Sequence[str]) -> List[str]:
    """The 18 parameter names an AURORA stage is expected to contain."""
    del stage  # both stages share the same key set
    del gene_list, cell_types
    keys = ["bulk_encoder.0.weight", "bulk_encoder.0.bias"]
    keys += [
        "lstm.weight_ih_l0",
        "lstm.weight_hh_l0",
        "lstm.bias_ih_l0",
        "lstm.bias_hh_l0",
    ]
    for layer in (0, 2, 4):
        keys += [f"decoder.{layer}.weight", f"decoder.{layer}.bias"]
    for layer in (0, 2, 4):
        keys += [f"celltype_decoder.{layer}.weight", f"celltype_decoder.{layer}.bias"]
    return keys
