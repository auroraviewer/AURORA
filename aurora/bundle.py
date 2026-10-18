"""The combined AURORA checkpoint container.

Requirement #1 of the reorganisation: the low-resolution and high-resolution
models used to be two separate ``.pth`` files.  They are now a single file, and
the new code system only accepts that combined file.

File format
-----------
A combined checkpoint is a plain ZIP container (readable with any zip tool)
whose first entry is a JSON manifest::

    AURORA_LUAD_combined.pth
    ├── aurora_bundle.json          # manifest, see below
    ├── payload/lowres.pt           # the original low-resolution torch archive
    ├── payload/highres.pt          # the original high-resolution torch archive
    └── assets/…                    # gene list, normalisation factors, params …

The payloads are the *byte-for-byte* original ``torch.save`` archives.  Keeping
them verbatim means:

* merging needs no PyTorch at all, so the merge tool runs anywhere;
* the tensors are provably identical to the published checkpoints (the manifest
  records the SHA-256 of every payload);
* :func:`load_stage_state_dict` simply feeds the embedded bytes to
  ``torch.load``, so nothing about the weights changes.

Manifest schema (``aurora_bundle.json``)::

    {
      "format": "aurora.combined",
      "format_version": 1,
      "payload_kind": "torch_archive",
      "model_name": "AURORA_LUAD",
      "cancer_type": "LUAD",
      "stages": {
        "lowres":  {"payload": "payload/lowres.pt",  "sha256": "…",
                    "patch_sizes": [3584, 896, 224], "img_emb_dim": 1536,
                    "extractor": "UNI2-h", "params": {…}},
        "highres": {"payload": "payload/highres.pt", "sha256": "…",
                    "patch_sizes": [56, 14], "img_emb_dim": 3072,
                    "extractor": "UNI2-h", "params": {…}}
      },
      "assets": {"gene_list": "assets/LUAD_gene.csv", …},
      "requires": {"highres": ["lowres"]}
    }

Two payload kinds are supported:

``torch_archive`` (default, no PyTorch required to build)
    Each stage payload is a separate embedded ``torch.save`` archive.

``inline_state_dict`` (optional, written by ``tools/export_native_combined.py``)
    The whole file is one ``torch.save`` archive of
    ``{"__aurora__": manifest, "lowres": state_dict, "highres": state_dict}``,
    so ``torch.load(path)`` returns everything directly.  The reader
    auto-detects this form.

This module is pure standard library; importing it never requires PyTorch.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import CheckpointFormatError
from .version import BUNDLE_FORMAT, BUNDLE_FORMAT_VERSION, __version__

#: Name of the manifest entry inside the ZIP container.
MANIFEST_ENTRY = "aurora_bundle.json"

#: Key that marks an inline (native ``torch.save``) combined checkpoint.
INLINE_MARKER_KEY = "__aurora__"

#: Pipeline stages a combined checkpoint may contain.
KNOWN_STAGES: Tuple[str, ...] = ("lowres", "highres")

#: Stage prerequisites recorded inside the bundle itself.
DEFAULT_REQUIRES: Dict[str, List[str]] = {"highres": ["lowres"]}

#: Payload kinds understood by the reader.
PAYLOAD_KIND_ARCHIVE = "torch_archive"
PAYLOAD_KIND_INLINE = "inline_state_dict"

_MERGE_COMMAND = (
    "python tools/merge_aurora_checkpoints.py "
    "--lowres <low-resolution.pth> --highres <high-resolution.pth> "
    "--out trained_model/AURORA_<CANCER>_combined.pth"
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike | str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed so large checkpoints do not blow up RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_from_zip(zf: zipfile.ZipFile, name: str) -> Dict[str, Any]:
    with zf.open(name) as handle:
        return json.loads(handle.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# Torch-free native-archive probing
# --------------------------------------------------------------------------- #


class _Opaque:
    """Stand-in for an unknown pickled object when probing without torch."""

    __slots__ = ("args", "kwargs")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<opaque {type(self).__name__} with {len(self.args)} args>"


class TensorProbe:
    """Shape/dtype-free description of one tensor in a ``torch.save`` archive."""

    __slots__ = ("storage_key", "storage_offset", "size", "stride", "numel", "location")

    def __init__(
        self,
        storage_key: Any = None,
        storage_offset: Any = None,
        size: Any = None,
        stride: Any = None,
        numel: Any = None,
        location: Any = None,
    ) -> None:
        self.storage_key = storage_key
        self.storage_offset = storage_offset
        self.size = tuple(size) if isinstance(size, (tuple, list)) else None
        self.stride = tuple(stride) if isinstance(stride, (tuple, list)) else None
        self.numel = numel
        self.location = location

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TensorProbe(size={self.size}, storage={self.storage_key})"


class _ProbeDict(dict):
    """``OrderedDict`` stand-in that tolerates the pickle ``BUILD`` opcode."""

    def __setstate__(self, state: Any) -> None:  # noqa: D105
        if isinstance(state, dict):
            self.update(state)
        elif isinstance(state, (list, tuple)):
            for item in state:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    self[item[0]] = item[1]


class _ProbeUnpickler(pickle.Unpickler):
    """Unpickler that reconstructs a torch archive's structure without torch.

    ``torch.Size`` becomes a plain ``tuple``, tensor rebuild calls become
    :class:`TensorProbe`, storages become small dicts, and anything else becomes
    an :class:`_Opaque`.  Plain JSON-like values (such as the inline manifest)
    round-trip exactly, which is what the rest of the module needs.
    """

    def find_class(self, module: str, name: str) -> Any:  # noqa: D102
        if module == "torch" and name == "Size":
            return tuple
        if module == "collections" and name in ("OrderedDict", "defaultdict"):
            return _ProbeDict
        if module in ("torch._utils",) and name in (
            "_rebuild_tensor_v2",
            "_rebuild_tensor_v3",
            "_rebuild_tensor",
            "_rebuild_parameter",
            "_rebuild_parameter_with_state",
            "_rebuild_qtensor",
        ):

            def _rebuild(storage: Any, storage_offset: Any, size: Any, stride: Any, *rest: Any) -> TensorProbe:
                numel = None
                location = None
                if isinstance(storage, dict):
                    numel = storage.get("numel")
                    location = storage.get("location")
                return TensorProbe(storage, storage_offset, size, stride, numel, location)

            return _rebuild
        if module.startswith("torch") and name.endswith("Storage"):
            return _Opaque
        return _Opaque

    def persistent_load(self, pid: Any) -> Any:  # noqa: D102
        if isinstance(pid, tuple) and len(pid) >= 5:
            _, _storage_type, key, location, numel = pid[:5]
            return {"key": key, "location": location, "numel": numel}
        if isinstance(pid, tuple):
            return {"key": pid[-1] if pid else None, "location": None, "numel": None}
        return {"key": pid, "location": None, "numel": None}


def probe_torch_archive(path: os.PathLike | str) -> Dict[str, Any]:
    """Return the top-level object of a ``torch.save`` archive without torch.

    Values that were tensors come back as :class:`TensorProbe`; their
    ``storage_key`` refers to an entry of ``probe_torch_storages()``.

    Raises :class:`CheckpointFormatError` when the file is not a torch archive.
    """
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise CheckpointFormatError(
            f"{path} is not a zip-based PyTorch checkpoint "
            "(it is neither an AURORA combined checkpoint nor a torch.save archive)",
            hint=(
                "Combined AURORA checkpoints produced by tools/merge_aurora_checkpoints.py "
                "are zip containers; validate the file with 'python -m aurora inspect --model <file>'."
            ),
        )
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        entry = locate_pickle_entry(names)
        if entry is None:
            raise CheckpointFormatError(
                f"{path} is a zip archive but has no 'data.pkl' entry, so it is not a torch.save file",
                hint="Re-download the checkpoint, or rebuild it with tools/merge_aurora_checkpoints.py.",
            )
        raw = zf.read(entry)
    try:
        return _ProbeUnpickler(io.BytesIO(raw)).load()
    except Exception as exc:  # pragma: no cover - depends on torch internals
        raise CheckpointFormatError(
            f"could not probe the torch archive {path} without PyTorch: {exc}",
            hint="Install torch, or run the same command in an environment that has it.",
        ) from exc


def probe_state_dict_shapes(path: os.PathLike | str) -> Dict[str, Optional[tuple]]:
    """Map every tensor key of a ``torch.save`` state_dict to its shape.

    Works without PyTorch, which lets the merge tool verify that the two
    checkpoints really are the low- and high-resolution halves of one model.
    Non-tensor entries (``nn.Module.state_dict`` attaches a ``_metadata``
    attribute that shows up as a key in a plain pickle) are skipped.
    """
    top = probe_torch_archive(path)
    if not isinstance(top, dict):
        raise CheckpointFormatError(
            f"{path} does not contain a state_dict at its top level (found {type(top).__name__})",
            hint="AURORA checkpoints are plain torch.save(state_dict) files.",
        )
    shapes: Dict[str, Optional[tuple]] = {}
    for key, value in top.items():
        if isinstance(value, TensorProbe):
            shapes[str(key)] = value.size
        elif str(key).startswith("_"):
            continue  # e.g. the '_metadata' attribute of nn.Module.state_dict()
        else:
            shapes[str(key)] = None
    return shapes


# --------------------------------------------------------------------------- #
# Payload layout
# --------------------------------------------------------------------------- #


def common_zip_prefix(names: Sequence[str]) -> str:
    """Return ``'root/'`` when every entry lives under one top-level directory.

    ``torch.save(state_dict, "AURORA_LUAD.pth")`` writes every record under a
    root directory named after the output file (``AURORA_LUAD/data.pkl``,
    ``AURORA_LUAD/data/0``, ...), and ``torch.load`` *requires* that common root:
    the reader derives its prefix from the first record and rejects the archive
    otherwise::

        RuntimeError: [enforce fail at inline_container.cc:174]
        . file in archive is not in a subdirectory: data.pkl

    Returns ``""`` when the entries are flat or span more than one directory.
    """
    roots = set()
    for name in names:
        if "/" not in name:
            return ""
        roots.add(name.split("/", 1)[0])
        if len(roots) > 1:
            return ""
    if len(roots) != 1:
        return ""
    root = roots.pop()
    return f"{root}/"


def locate_pickle_entry(names: Sequence[str]) -> Optional[str]:
    """Name of the ``data.pkl`` record, under the archive root when there is one."""
    if "data.pkl" in names:
        return "data.pkl"
    prefix = common_zip_prefix(list(names))
    if prefix and f"{prefix}data.pkl" in names:
        return f"{prefix}data.pkl"
    return None


def torch_archive_layout(names: Sequence[str]) -> Tuple[bool, str, str]:
    """Check a zip entry list the way PyTorch's ``PyTorchStreamReader`` does.

    Returns ``(ok, archive_root, problem)``.  A ``torch.save`` archive must have
    every record under a single root directory; a flat archive cannot be loaded
    by ``torch.load`` at all.  Reproducing the rule here means the merge tool and
    ``aurora validate`` can reject a payload that would fail at run time, without
    needing PyTorch.
    """
    if not names:
        return False, "", "the archive is empty"
    prefix = common_zip_prefix(list(names))
    if not prefix:
        flat = [name for name in names if "/" not in name]
        if flat:
            return (
                False,
                "",
                "the archive is flat: PyTorch requires every record under one root "
                f"directory (as torch.save writes it), but found {flat[0]!r} at the top level",
            )
        return False, "", "the archive has more than one top-level directory"
    if f"{prefix}data.pkl" not in names:
        return False, prefix, f"the archive has no 'data.pkl' under its root directory {prefix!r}"
    return True, prefix, ""


def read_torch_archive_bytes(path: os.PathLike | str) -> Tuple[bytes, str]:
    """Return ``(verbatim_bytes, archive_root)`` for a ``torch.save`` file.

    The bytes are **never rewritten**.  ``torch.load`` requires the archive's
    root directory exactly as ``torch.save`` wrote it, so stripping or adding a
    prefix would make the payload unloadable:

        RuntimeError: file in archive is not in a subdirectory: data.pkl

    The file is only inspected, so the caller can record and print the root it
    found.
    """
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise CheckpointFormatError(
            f"{path} is not a zip-based PyTorch checkpoint",
            hint="AURORA checkpoints are produced by torch.save; re-download the file.",
        )
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    ok, prefix, problem = torch_archive_layout(names)
    if not ok:
        raise CheckpointFormatError(
            f"{path} cannot be loaded by torch.load: {problem}",
            hint=(
                "A checkpoint written by torch.save always nests its records under a root "
                "directory named after the output file. Rebuild the file with torch.save, "
                "or re-download it."
            ),
        )
    return path.read_bytes(), prefix


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


@dataclass
class StageEntry:
    """One model stage inside a combined checkpoint."""

    name: str
    payload: str
    sha256: str = ""
    size_bytes: int = 0
    source_file: str = ""
    #: SHA-256 of the source file that was embedded (payload bytes are verbatim,
    #: so this equals :attr:`sha256`).
    source_sha256: str = ""
    #: Root directory the ``torch.save`` archive nests its records under, e.g.
    #: ``"AURORA_LUAD/"``.  ``torch.load`` requires it, so it is recorded for
    #: diagnostics; the payload is embedded with it intact.
    archive_root: str = ""
    patch_sizes: List[int] = field(default_factory=list)
    img_emb_dim: Optional[int] = None
    extractor: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    params_file: str = ""
    state_dict_keys: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "payload": self.payload,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if self.source_file:
            data["source_file"] = self.source_file
        if self.source_sha256:
            data["source_sha256"] = self.source_sha256
        if self.archive_root:
            data["archive_root"] = self.archive_root
        if self.patch_sizes:
            data["patch_sizes"] = list(self.patch_sizes)
        if self.img_emb_dim is not None:
            data["img_emb_dim"] = self.img_emb_dim
        if self.extractor:
            data["extractor"] = self.extractor
        if self.params:
            data["params"] = self.params
        if self.params_file:
            data["params_file"] = self.params_file
        if self.state_dict_keys is not None:
            data["state_dict_keys"] = self.state_dict_keys
        data.update(self.extra)
        return data

    @classmethod
    def from_json(cls, name: str, data: Mapping[str, Any]) -> "StageEntry":
        known = {
            "payload",
            "sha256",
            "size_bytes",
            "source_file",
            "source_sha256",
            "archive_root",
            # Written by an earlier bundle format; dropped on read.
            "source_prefix_stripped",
            "patch_sizes",
            "img_emb_dim",
            "extractor",
            "params",
            "params_file",
            "state_dict_keys",
        }
        return cls(
            name=name,
            payload=str(data.get("payload", "")),
            sha256=str(data.get("sha256", "")),
            size_bytes=int(data.get("size_bytes", 0) or 0),
            source_file=str(data.get("source_file", "")),
            source_sha256=str(data.get("source_sha256", "")),
            archive_root=str(data.get("archive_root", "")),
            patch_sizes=[int(x) for x in data.get("patch_sizes", []) or []],
            img_emb_dim=data.get("img_emb_dim"),
            extractor=str(data.get("extractor", "")),
            params=dict(data.get("params", {}) or {}),
            params_file=str(data.get("params_file", "")),
            state_dict_keys=data.get("state_dict_keys"),
            extra={k: v for k, v in data.items() if k not in known},
        )


@dataclass
class BundleManifest:
    """Parsed ``aurora_bundle.json``."""

    format: str = BUNDLE_FORMAT
    format_version: int = BUNDLE_FORMAT_VERSION
    payload_kind: str = PAYLOAD_KIND_ARCHIVE
    model_name: str = ""
    cancer_type: str = ""
    created_utc: str = ""
    created_by: str = ""
    stages: Dict[str, StageEntry] = field(default_factory=dict)
    assets: Dict[str, str] = field(default_factory=dict)
    requires: Dict[str, List[str]] = field(default_factory=lambda: dict(DEFAULT_REQUIRES))
    notes: str = ""

    # -- convenience -------------------------------------------------------- #
    @property
    def stage_names(self) -> List[str]:
        return [s for s in KNOWN_STAGES if s in self.stages]

    def stage(self, name: str) -> StageEntry:
        try:
            return self.stages[name]
        except KeyError as exc:
            raise CheckpointFormatError(
                f"the combined checkpoint has no '{name}' stage "
                f"(it contains: {', '.join(self.stage_names) or 'nothing'})",
                hint=(
                    "Rebuild it with tools/merge_aurora_checkpoints.py, passing both "
                    "--lowres and --highres."
                ),
            ) from exc

    def params_for(self, name: str) -> Dict[str, Any]:
        return dict(self.stage(name).params)

    def to_json(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "format_version": self.format_version,
            "payload_kind": self.payload_kind,
            "model_name": self.model_name,
            "cancer_type": self.cancer_type,
            "created_utc": self.created_utc or utc_now_iso(),
            "created_by": self.created_by or f"aurora {__version__}",
            "stages": {name: entry.to_json() for name, entry in self.stages.items()},
            "assets": dict(self.assets),
            "requires": {k: list(v) for k, v in self.requires.items()},
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "BundleManifest":
        stages = {
            name: StageEntry.from_json(name, payload)
            for name, payload in (data.get("stages", {}) or {}).items()
        }
        return cls(
            format=str(data.get("format", BUNDLE_FORMAT)),
            format_version=int(data.get("format_version", BUNDLE_FORMAT_VERSION) or 1),
            payload_kind=str(data.get("payload_kind", PAYLOAD_KIND_ARCHIVE)),
            model_name=str(data.get("model_name", "")),
            cancer_type=str(data.get("cancer_type", "")),
            created_utc=str(data.get("created_utc", "")),
            created_by=str(data.get("created_by", "")),
            stages=stages,
            assets=dict(data.get("assets", {}) or {}),
            requires={
                str(k): [str(x) for x in v] for k, v in (data.get("requires", DEFAULT_REQUIRES) or {}).items()
            },
            notes=str(data.get("notes", "")),
        )


# --------------------------------------------------------------------------- #
# Detection + reading
# --------------------------------------------------------------------------- #


@dataclass
class CheckpointKind:
    """What a ``.pth`` file turned out to be."""

    path: Path
    kind: str  # 'combined_container' | 'combined_inline' | 'legacy_torch' | 'unknown'
    manifest: Optional[BundleManifest] = None
    reason: str = ""

    @property
    def is_combined(self) -> bool:
        return self.kind in ("combined_container", "combined_inline")


def classify_checkpoint(path: os.PathLike | str) -> CheckpointKind:
    """Decide whether ``path`` is a combined checkpoint, a legacy one, or neither."""
    path = Path(path)
    if not path.exists():
        return CheckpointKind(path, "unknown", None, "file does not exist")
    if not path.is_file():
        return CheckpointKind(path, "unknown", None, "not a regular file")
    if not zipfile.is_zipfile(path):
        return CheckpointKind(path, "unknown", None, "not a zip archive")
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
    if MANIFEST_ENTRY in names:
        try:
            with zipfile.ZipFile(path) as zf:
                manifest = BundleManifest.from_json(_read_json_from_zip(zf, MANIFEST_ENTRY))
        except Exception as exc:
            return CheckpointKind(path, "unknown", None, f"manifest is unreadable: {exc}")
        return CheckpointKind(path, "combined_container", manifest, "aurora_bundle.json present")
    if locate_pickle_entry(sorted(names)) is not None:
        try:
            top = probe_torch_archive(path)
        except CheckpointFormatError as exc:
            return CheckpointKind(path, "legacy_torch", None, str(exc))
        if isinstance(top, dict) and INLINE_MARKER_KEY in top:
            try:
                manifest = BundleManifest.from_json(top[INLINE_MARKER_KEY])
            except Exception as exc:
                return CheckpointKind(path, "unknown", None, f"inline manifest is unreadable: {exc}")
            manifest.payload_kind = PAYLOAD_KIND_INLINE
            return CheckpointKind(path, "combined_inline", manifest, "inline __aurora__ manifest present")
        keys = list(top.keys())[:6] if isinstance(top, dict) else []
        return CheckpointKind(
            path,
            "legacy_torch",
            None,
            "torch.save archive without an AURORA manifest; "
            f"top-level keys start with: {keys}",
        )
    return CheckpointKind(path, "unknown", None, f"zip archive without a known AURORA entry: {sorted(names)[:5]}")


def is_combined_checkpoint(path: os.PathLike | str) -> bool:
    """True when ``path`` is a combined AURORA checkpoint."""
    return classify_checkpoint(path).is_combined


def _legacy_error(path: Path, detail: str) -> CheckpointFormatError:
    return CheckpointFormatError(
        f"{path} is not a combined AURORA checkpoint ({detail}).",
        hint=(
            "The new code system is only compatible with a combined checkpoint that "
            "bundles the low-resolution and high-resolution models in one file.\n"
            "Build one from the two legacy files with:\n"
            f"  {_MERGE_COMMAND}\n"
            "Or point --model at an existing combined file. "
            "The legacy scripts under code/ still read the old separate files."
        ),
    )


class CombinedCheckpoint:
    """Read-only handle on a combined AURORA checkpoint.

    Opening a checkpoint never imports torch: the manifest, the parameters and
    the asset files are read straight from the container.  Torch is only needed
    when the tensors themselves are requested through
    :meth:`load_stage_state_dict`.
    """

    def __init__(self, path: os.PathLike | str, kind: CheckpointKind) -> None:
        self.path = Path(path)
        self.kind = kind
        self._manifest = kind.manifest
        assert self._manifest is not None

    # -- construction ------------------------------------------------------- #
    @classmethod
    def open(cls, path: os.PathLike | str) -> "CombinedCheckpoint":
        """Open ``path``, raising :class:`CheckpointFormatError` when it is not combined."""
        kind = classify_checkpoint(path)
        if kind.is_combined:
            return cls(path, kind)
        raise _legacy_error(Path(path), kind.reason or kind.kind)

    # -- manifest access ---------------------------------------------------- #
    @property
    def manifest(self) -> BundleManifest:
        return self._manifest

    @property
    def model_name(self) -> str:
        return self._manifest.model_name

    @property
    def cancer_type(self) -> str:
        return self._manifest.cancer_type

    @property
    def model_id(self) -> str:
        """``model_name`` with a fallback derived from the file name."""
        return self._manifest.model_name or self.path.stem

    def stages(self) -> List[str]:
        return self._manifest.stage_names

    def stage(self, name: str) -> StageEntry:
        return self._manifest.stage(name)

    def params(self, stage: str) -> Dict[str, Any]:
        return self._manifest.params_for(stage)

    def requires(self, stage: str) -> List[str]:
        return list(self._manifest.requires.get(stage, []))

    # -- payload access ----------------------------------------------------- #
    def payload_bytes(self, stage: str) -> bytes:
        """Raw bytes of a stage payload (a ``torch.save`` archive)."""
        entry = self.stage(stage)
        if self._manifest.payload_kind == PAYLOAD_KIND_INLINE:
            raise CheckpointFormatError(
                f"{self.path} stores its state dicts inline, so stage '{stage}' has no separate payload",
                hint="Use CombinedCheckpoint.load_stage_state_dict() instead of payload_bytes().",
            )
        if not entry.payload:
            raise CheckpointFormatError(
                f"stage '{stage}' has no payload entry in the manifest of {self.path}",
                hint="Rebuild the combined checkpoint with tools/merge_aurora_checkpoints.py.",
            )
        with zipfile.ZipFile(self.path) as zf:
            try:
                data = zf.read(entry.payload)
            except KeyError as exc:
                raise CheckpointFormatError(
                    f"the manifest of {self.path} references '{entry.payload}' but the file has no such entry",
                    hint="The checkpoint is truncated or was modified; rebuild it from the source files.",
                ) from exc
        if entry.sha256:
            actual = sha256_bytes(data)
            if actual != entry.sha256:
                raise CheckpointFormatError(
                    f"payload '{entry.payload}' of {self.path} is corrupted "
                    f"(expected sha256 {entry.sha256[:12]}…, found {actual[:12]}…)",
                    hint="Rebuild the combined checkpoint from the original .pth files.",
                )
        return data

    def asset_bytes(self, key: str) -> bytes:
        """Bytes of a named asset (gene list, normalisation factors, ...)."""
        try:
            entry = self._manifest.assets[key]
        except KeyError as exc:
            raise CheckpointFormatError(
                f"the combined checkpoint {self.path} has no asset '{key}' "
                f"(available: {', '.join(sorted(self._manifest.assets)) or 'none'})",
                hint="Rebuild it with tools/merge_aurora_checkpoints.py --gene-list ... --cell-types ...",
            ) from exc
        with zipfile.ZipFile(self.path) as zf:
            try:
                return zf.read(entry)
            except KeyError as exc:  # pragma: no cover - corrupt container
                raise CheckpointFormatError(
                    f"the manifest of {self.path} references asset '{entry}' but the file has no such entry",
                    hint="Rebuild the combined checkpoint.",
                ) from exc

    def write_asset_to(self, key: str, destination: os.PathLike | str) -> Path:
        """Materialise a bundled asset on disk (used to feed the legacy scripts)."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.asset_bytes(key))
        return destination

    # -- tensors ------------------------------------------------------------ #
    def payload_layout(self, stage: str) -> Tuple[bool, str, str]:
        """``(ok, archive_root, problem)`` for one stage's embedded payload."""
        data = self.payload_bytes(stage)
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = archive.namelist()
        except zipfile.BadZipFile as exc:
            return False, "", f"the payload is not a zip archive ({exc})"
        return torch_archive_layout(names)

    def ensure_payload_loadable(self, stage: str) -> str:
        """Raise unless ``torch.load`` could read this stage's payload.

        ``torch.load`` needs the root directory ``torch.save`` wrote and fails
        with a bare ``RuntimeError`` otherwise.  Checking first turns that into
        an actionable error **before** PyTorch is involved, which also means the
        check works while planning a run, not only while loading weights.
        """
        ok, root, problem = self.payload_layout(stage)
        if ok:
            return root
        raise CheckpointFormatError(
            f"the '{stage}' payload inside {self.path} cannot be loaded by torch.load: {problem}",
            hint=self.rebuild_hint(stage),
        )

    def rebuild_hint(self, stage: Optional[str] = None) -> str:
        """A concrete merge command that rebuilds this checkpoint correctly."""
        manifest = self._manifest
        stages = manifest.stage_names or list(manifest.stages)
        lowres = manifest.stages.get("lowres")
        highres = manifest.stages.get("highres")
        low_name = (lowres.source_file if lowres else "") or "<low-resolution.pth>"
        high_name = (highres.source_file if highres else "") or "<high-resolution.pth>"
        lines = [
            "This checkpoint was built by an older revision of the merge tool, which "
            "rewrote the torch.save archive layout. Rebuild it from the original files "
            "(they are only read, never modified):",
            "",
            "  python tools/merge_aurora_checkpoints.py \\",
            f"      --lowres  <path to {low_name}> \\",
            f"      --highres <path to {high_name}> \\",
            "      --gene-list    <...> \\",
            "      --cell-types   <...> \\",
            "      --norm-factors <...> \\",
            f"      --out {self.path} --force",
            "",
            "Then re-run the same command: finished stages are reused, so nothing "
            "already computed is lost.",
        ]
        if stage:
            lines.insert(0, f"affected stage: {stage} (of: {', '.join(stages) or 'none'})")
            lines.insert(1, "")
        return "\n".join(lines)

    def load_stage_state_dict(self, stage: str, map_location: Any = "cpu") -> Mapping[str, Any]:
        """Load one stage's ``state_dict``.

        This is the only method that requires PyTorch.
        """
        # Check the artefact *before* asking for PyTorch: the layout check needs
        # no torch, and a broken checkpoint should be reported even in an
        # environment that has not installed it yet.
        if self._manifest.payload_kind != PAYLOAD_KIND_INLINE:
            self.ensure_payload_loadable(stage)

        from .deps import require  # local import: keep this module torch-free

        torch = require("torch", purpose=f"loading the '{stage}' weights")
        if self._manifest.payload_kind == PAYLOAD_KIND_INLINE:
            with open(self.path, "rb") as handle:
                blob = torch.load(handle, map_location=map_location, weights_only=False)
            try:
                return blob[stage]
            except (KeyError, TypeError) as exc:
                raise CheckpointFormatError(
                    f"{self.path} is an inline combined checkpoint but has no '{stage}' entry",
                    hint="Rebuild it with tools/export_native_combined.py.",
                ) from exc
        data = self.payload_bytes(stage)
        return torch.load(io.BytesIO(data), map_location=map_location, weights_only=False)

    def load_all_state_dicts(self, map_location: Any = "cpu") -> Dict[str, Mapping[str, Any]]:
        """Load every stage's ``state_dict`` at once."""
        return {name: self.load_stage_state_dict(name, map_location) for name in self.stages()}

    # -- reporting ---------------------------------------------------------- #
    def describe(self) -> str:
        manifest = self._manifest
        lines = [
            f"Combined AURORA checkpoint: {self.path}",
            f"  format          : {manifest.format} v{manifest.format_version} "
            f"({manifest.payload_kind})",
            f"  model           : {manifest.model_name or '(unnamed)'}"
            + (f"  cancer type: {manifest.cancer_type}" if manifest.cancer_type else ""),
            f"  created         : {manifest.created_utc or 'unknown'} by {manifest.created_by or 'unknown'}",
            f"  size on disk    : {self.path.stat().st_size / 1e6:.2f} MB",
            "  stages:",
        ]
        for name in manifest.stage_names:
            entry = manifest.stage(name)
            detail = []
            if entry.patch_sizes:
                detail.append(f"patch_sizes={entry.patch_sizes}")
            if entry.img_emb_dim is not None:
                detail.append(f"img_emb_dim={entry.img_emb_dim}")
            if entry.extractor:
                detail.append(f"extractor={entry.extractor}")
            if entry.size_bytes:
                detail.append(f"{entry.size_bytes / 1e6:.2f} MB")
            req = manifest.requires.get(name) or []
            if req:
                detail.append(f"requires={req}")
            if entry.archive_root:
                detail.append(f"archive_root={entry.archive_root!r}")
            lines.append(f"    - {name}: " + ", ".join(detail))
        if manifest.assets:
            lines.append("  assets:")
            for key, entry in sorted(manifest.assets.items()):
                lines.append(f"    - {key}: {entry}")
        if manifest.notes:
            lines.append(f"  notes: {manifest.notes}")
        return "\n".join(lines)


def open_checkpoint(path: os.PathLike | str) -> CombinedCheckpoint:
    """Module-level alias for :meth:`CombinedCheckpoint.open`."""
    return CombinedCheckpoint.open(path)


# --------------------------------------------------------------------------- #
# Writing (pure standard library)
# --------------------------------------------------------------------------- #


def build_container_bundle(
    destination: os.PathLike | str,
    stages: Mapping[str, os.PathLike | str],
    *,
    manifest: Optional[BundleManifest] = None,
    model_name: str = "",
    cancer_type: str = "",
    assets: Optional[Mapping[str, os.PathLike | str]] = None,
    stage_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    stage_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    notes: str = "",
    overwrite: bool = False,
    compresslevel: int = 1,
) -> CombinedCheckpoint:
    """Write a combined checkpoint container.

    ``stages`` maps a stage name (``'lowres'`` / ``'highres'``) to the legacy
    ``.pth`` file to embed.  The files are copied verbatim; their SHA-256 is
    recorded so the reader can prove nothing changed.

    Returns the freshly written checkpoint, already opened for reading.
    """
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise CheckpointFormatError(
            f"{destination} already exists",
            hint="Pass --force to overwrite it, or choose a different --out path.",
        )
    if not stages:
        raise CheckpointFormatError(
            "no stages given to build_container_bundle()",
            hint="Pass at least {'lowres': <path>} and {'highres': <path>}.",
        )

    source_paths: Dict[str, Path] = {}
    for name, source in stages.items():
        source = Path(source)
        if not source.is_file():
            raise CheckpointFormatError(
                f"stage '{name}' points at {source}, which is not a file",
                hint="Check the --lowres / --highres paths.",
            )
        if not zipfile.is_zipfile(source):
            raise CheckpointFormatError(
                f"{source} (stage '{name}') is not a torch.save zip archive",
                hint=(
                    "AURORA checkpoints are produced by torch.save with the default "
                    "zip serialisation. Re-download the checkpoint if it looks truncated."
                ),
            )
        source_paths[name] = source

    base = manifest or BundleManifest()
    base.format = BUNDLE_FORMAT
    base.format_version = BUNDLE_FORMAT_VERSION
    base.payload_kind = PAYLOAD_KIND_ARCHIVE
    base.model_name = model_name or base.model_name
    base.cancer_type = cancer_type or base.cancer_type
    base.created_utc = utc_now_iso()
    base.created_by = base.created_by or f"aurora {__version__} (merge_aurora_checkpoints)"
    base.notes = notes or base.notes
    if not base.requires:
        base.requires = dict(DEFAULT_REQUIRES)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(destination.name + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    asset_entries: Dict[str, str] = dict(base.assets)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
            # assets first so their names can be recorded in the manifest
            for key, source in (assets or {}).items():
                source = Path(source)
                if not source.is_file():
                    raise CheckpointFormatError(
                        f"asset '{key}' points at {source}, which is not a file",
                        hint="Check the asset paths passed to the merge tool.",
                    )
                entry_name = f"assets/{source.name}"
                zf.write(source, arcname=entry_name)
                asset_entries[key] = entry_name
            for name, source in source_paths.items():
                entry = base.stages.get(name) or StageEntry(name=name, payload=f"payload/{name}.pt")
                entry.payload = f"payload/{name}.pt"
                entry.source_file = source.name
                # Embed the archive verbatim: torch.load requires the root
                # directory torch.save wrote.  read_torch_archive_bytes() fails
                # early if the layout could never be loaded.
                payload_bytes, archive_root = read_torch_archive_bytes(source)
                entry.archive_root = archive_root
                entry.size_bytes = len(payload_bytes)
                entry.sha256 = sha256_bytes(payload_bytes)
                entry.source_sha256 = entry.sha256
                if stage_params and name in stage_params:
                    entry.params = dict(stage_params[name])
                if stage_metadata and name in stage_metadata:
                    meta = dict(stage_metadata[name])
                    entry.patch_sizes = [int(x) for x in meta.pop("patch_sizes", entry.patch_sizes) or []]
                    entry.img_emb_dim = meta.pop("img_emb_dim", entry.img_emb_dim)
                    entry.extractor = str(meta.pop("extractor", entry.extractor))
                    entry.state_dict_keys = meta.pop("state_dict_keys", entry.state_dict_keys)
                    entry.extra.update(meta)
                base.stages[name] = entry
                zf.write(source, arcname=entry.payload)
            base.assets = asset_entries
            zf.writestr(MANIFEST_ENTRY, json.dumps(base.to_json(), indent=2, sort_keys=False) + "\n")
        os.replace(tmp_path, destination)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return CombinedCheckpoint.open(destination)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass
class BundleValidation:
    """Outcome of :func:`validate_bundle`."""

    checkpoint: Path
    ok: bool = True
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_problem(self, message: str) -> None:
        self.ok = False
        self.problems.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def describe(self) -> str:
        lines = [f"{'OK  ' if self.ok else 'FAIL'}  {self.checkpoint}"]
        for problem in self.problems:
            lines.append(f"  problem: {problem}")
        for warning in self.warnings:
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)


def validate_bundle(path: os.PathLike | str, verify_payloads: bool = True) -> BundleValidation:
    """Check that a combined checkpoint is self-consistent.

    Verifies the manifest, that ``lowres`` is present (it is the mandatory
    stage), that every declared payload exists and hashes correctly, and that
    the ``requires`` graph only references known stages.
    """
    path = Path(path)
    result = BundleValidation(checkpoint=path)
    kind = classify_checkpoint(path)
    if not kind.is_combined:
        result.add_problem(kind.reason or f"not a combined checkpoint ({kind.kind})")
        return result

    manifest = kind.manifest
    assert manifest is not None
    if manifest.format != BUNDLE_FORMAT:
        result.add_problem(f"unexpected format marker {manifest.format!r}, expected {BUNDLE_FORMAT!r}")
    if manifest.format_version > BUNDLE_FORMAT_VERSION:
        result.add_warning(
            f"format version {manifest.format_version} is newer than this code understands "
            f"({BUNDLE_FORMAT_VERSION}); upgrade the aurora package"
        )
    if "lowres" not in manifest.stages:
        result.add_problem("the mandatory 'lowres' stage is missing from the manifest")
    for stage in manifest.stage_names:
        entry = manifest.stage(stage)
        if not entry.patch_sizes:
            result.add_warning(f"stage '{stage}' does not record its patch sizes")
        if entry.img_emb_dim is None:
            result.add_warning(f"stage '{stage}' does not record img_emb_dim")
    for stage, requirements in manifest.requires.items():
        for requirement in requirements:
            if requirement not in manifest.stages:
                result.add_problem(
                    f"stage '{stage}' requires '{requirement}', which is not in the checkpoint"
                )
    if manifest.payload_kind == PAYLOAD_KIND_ARCHIVE and verify_payloads:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        missing_entries = [
            (stage, manifest.stage(stage).payload)
            for stage in manifest.stage_names
            if manifest.stage(stage).payload not in names
        ]
        for stage, entry_name in missing_entries:
            result.add_problem(f"stage '{stage}' payload '{entry_name}' is missing from the archive")
        if not missing_entries:
            checkpoint = CombinedCheckpoint(path, kind)
            for stage in manifest.stage_names:
                try:
                    checkpoint.payload_bytes(stage)
                except CheckpointFormatError as exc:
                    result.add_problem(exc.message)
        for key, entry_name in manifest.assets.items():
            if entry_name not in names:
                result.add_problem(f"asset '{key}' ('{entry_name}') is missing from the archive")
        # Every embedded payload must still be loadable by torch.load, which
        # requires the root directory torch.save wrote.  A flat payload fails at
        # run time with "file in archive is not in a subdirectory: data.pkl", so
        # it is reported here instead.
        for stage in manifest.stage_names:
            entry = manifest.stage(stage)
            if entry.payload not in names:
                continue
            try:
                with zipfile.ZipFile(path) as zf:
                    payload = zf.read(entry.payload)
                with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                    inner_names = inner.namelist()
            except (KeyError, zipfile.BadZipFile) as exc:
                result.add_problem(f"stage '{stage}' payload cannot be read: {exc}")
                continue
            ok, root, problem = torch_archive_layout(inner_names)
            if not ok:
                result.add_problem(f"stage '{stage}' payload would fail torch.load: {problem}")
                continue
            if entry.archive_root and entry.archive_root != root:
                result.add_warning(
                    f"stage '{stage}' payload root is {root!r} but the manifest records "
                    f"{entry.archive_root!r}"
                )
    elif manifest.payload_kind == PAYLOAD_KIND_ARCHIVE:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        for stage in manifest.stage_names:
            if manifest.stage(stage).payload not in names:
                result.add_problem(
                    f"stage '{stage}' payload '{manifest.stage(stage).payload}' is missing"
                )
    return result


def list_combined_checkpoints(directory: os.PathLike | str, pattern: str = "*.pth") -> List[Path]:
    """Return every combined checkpoint found in ``directory`` (sorted)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = [p for p in sorted(directory.glob(pattern)) if is_combined_checkpoint(p)]
    return found
