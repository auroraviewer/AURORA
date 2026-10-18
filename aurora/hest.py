"""The model registry that HEST's tissue segmentation actually reads.

HEST delegates segmentation to **TRIDENT**::

    HEST/src/hest/HESTData.py         segment_tissue()
      -> trident_compat.py            segment_tissue_deep()
        -> trident.segmentation_models.load.segmentation_model_factory("hest")

TRIDENT keeps a local registry of checkpoints so it can run offline.  When the
weights are absent it raises::

    FileNotFoundError: Internet connection not available and checkpoint not found
    locally in model registry at trident/segmentation_models/local_ckpts.json.
    To proceed, please manually download deeplabv3_seg_v4.ckpt from: ...
    and place it at: local_ckpts.json

Two things in that message mislead:

* ``trident/segmentation_models/local_ckpts.json`` is *relative to the installed
  ``trident`` package*, not to the current directory -- it lives in
  ``<site-packages>/trident/segmentation_models/``;
* ``and place it at: local_ckpts.json`` is not a destination at all.  The registry
  ships with that entry already present and its **value** is the literal string
  ``"local_ckpts.json"``, a placeholder; the error prints the recorded value.  The
  fix is to point that entry at the downloaded file.

So this module edits TRIDENT's registry in place, repairing the placeholder rather
than inventing a new file somewhere.  HEST's own tree is searched too, because some
installs vendor a copy, but TRIDENT is where the message points and the entry is
looked up by its placeholder value rather than by guessing the key.

Standard library only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .errors import ConfigError

#: The registry file name, as named in TRIDENT's error.
CKPT_JSON_NAME = "local_ckpts.json"

#: Where TRIDENT keeps it, relative to its package directory.
TRIDENT_REGISTRY_SUBPATH = ("segmentation_models", CKPT_JSON_NAME)

#: The model HEST's tissue segmentation needs, and where it comes from.
SEGMENTATION_CKPT = "deeplabv3_seg_v4.ckpt"
SEGMENTATION_CKPT_URL = "https://huggingface.co/MahmoodLab/hest-tissue-seg/"

#: The registry is keyed by *model name*, not by file name -- TRIDENT asks for
#: ``segmentation_model_factory("hest")`` (see the traceback), so the entry that has
#: to be filled in is the one named ``hest``.  The shipped registry also lists
#: ``grandqc`` and ``grandqc_artifact`` with empty values, which is what an
#: unconfigured registry looks like.
MODEL_NAME = "hest"

#: Values that mean "not configured": the shipped registry uses an empty string.
PLACEHOLDER_VALUE = CKPT_JSON_NAME


def _package_dir(name: str) -> Optional[Path]:
    """Directory of an installed package, if it is importable."""
    try:
        import importlib.util

        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or [])
    if locations:
        return Path(locations[0])
    if spec.origin:
        return Path(spec.origin).parent
    return None


def find_trident_package() -> Optional[Path]:
    """The installed ``trident`` package directory."""
    return _package_dir("trident")


def find_hest_package() -> Optional[Path]:
    """The installed ``hest`` package directory."""
    return _package_dir("hest")


def registry_candidates() -> List[Path]:
    """Every place the registry may live, TRIDENT first."""
    candidates: List[Path] = []
    trident = find_trident_package()
    if trident is not None:
        # This is the path the error names: trident/segmentation_models/local_ckpts.json
        candidates.append(trident.joinpath(*TRIDENT_REGISTRY_SUBPATH))
        candidates.append(trident / CKPT_JSON_NAME)
    for hest in (find_hest_package(),):
        if hest is not None:
            # Some installs vendor a copy; cheaper to check than to argue.
            candidates.extend([hest / CKPT_JSON_NAME, hest.parent / CKPT_JSON_NAME])
    candidates.append(Path.cwd() / CKPT_JSON_NAME)

    seen: List[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def find_registry() -> Optional[Path]:
    """The registry that exists, preferring TRIDENT's canonical location."""
    for candidate in registry_candidates():
        if candidate.is_file():
            return candidate.resolve()
    return None


def default_registry_path(explicit: Optional[str] = None) -> Path:
    """Where to write, when nothing exists yet."""
    if explicit:
        return Path(explicit).expanduser()
    existing = find_registry()
    if existing is not None:
        return existing
    trident = find_trident_package()
    if trident is None:
        raise ConfigError(
            "TRIDENT is not importable, so its checkpoint registry cannot be located",
            hint=(
                "HEST segments tissue through TRIDENT. Install it, then re-run:\n"
                "  pip install trident   # or: pip install -e <trident checkout>\n"
                "The file that needs editing is the one the error names:\n"
                "  <site-packages>/trident/segmentation_models/local_ckpts.json"
            ),
        )
    return trident.joinpath(*TRIDENT_REGISTRY_SUBPATH)


def read_registry(path: Path) -> Dict[str, Any]:
    """Load the registry, tolerating an empty or missing file."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError) as exc:
        raise ConfigError(
            f"{path} is not valid JSON: {exc}",
            hint="Fix or delete it; it should map a checkpoint name to a local path.",
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} should contain a JSON object mapping a checkpoint name to its path, "
            f"found {type(data).__name__}",
            hint=f'For example: {{"{SEGMENTATION_CKPT}": "/data/ckpts/{SEGMENTATION_CKPT}"}}',
        )
    return data


def placeholder_keys(data: Dict[str, Any]) -> List[str]:
    """Entries with no usable path: empty, or the shipped placeholder."""
    keys: List[str] = []
    for key, value in data.items():
        text = str(value).strip()
        if not text or text == PLACEHOLDER_VALUE or text.endswith("/" + PLACEHOLDER_VALUE):
            keys.append(key)
    return keys


def target_key(data: Dict[str, Any]) -> str:
    """The entry TRIDENT reads: the model name it asks for.

    Decided from the registry rather than assumed -- TRIDENT calls
    ``segmentation_model_factory("hest")``, so the key is ``hest``.  Falling back to
    a filename key (as an earlier version did) silently adds an entry nothing reads
    while leaving the real one empty.
    """
    if MODEL_NAME in data:
        return MODEL_NAME
    for key in data:
        if key.lower() == MODEL_NAME:
            return key
    # A differently-named registry: accept an entry that already names the weights.
    for key in data:
        if SEGMENTATION_CKPT in key:
            return key
    placeholders = placeholder_keys(data)
    if len(placeholders) == 1:
        return placeholders[0]
    return MODEL_NAME


def record_checkpoint(registry_path: Path, ckpt_path: Path) -> Tuple[Dict[str, Any], str]:
    """Point the registry's model entry at ``ckpt_path`` and return ``(data, key)``."""
    data = read_registry(registry_path)
    key = target_key(data)
    data[key] = str(ckpt_path.expanduser().resolve())
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data, key


def check_setup() -> Tuple[bool, str]:
    """``(ok, message)`` describing whether TRIDENT can supply the weights."""
    if find_trident_package() is None:
        return False, "TRIDENT is not importable, so HEST cannot segment at all"
    registry = find_registry()
    if registry is None:
        looked = "\n".join(f"  - {c}" for c in registry_candidates())
        return False, f"no {CKPT_JSON_NAME} was found. Looked in:\n{looked}"
    data = read_registry(registry)
    key = target_key(data)
    recorded = str(data.get(key, "")).strip()
    if not recorded:
        return False, (
            f"{registry} has no path for the {key!r} entry (it is empty). "
            "TRIDENT asks for the model by name, so that entry is the one it reads."
        )
    if key in placeholder_keys(data):
        return False, (
            f"{registry} still holds the shipped placeholder for {key!r} "
            f"({recorded!r} is not a path)"
        )
    resolved = Path(recorded).expanduser()
    if not resolved.is_file():
        return False, f"{registry} points at {resolved}, which does not exist"
    return True, f"{registry} [{key}] -> {resolved}"


def require_setup() -> None:
    """Raise with the fix when HEST/TRIDENT cannot find the segmentation weights."""
    if find_hest_package() is None and find_trident_package() is None:
        return  # not installed at all; the dependency check reports that instead
    ok, message = check_setup()
    if ok:
        return
    raise ConfigError(
        f"HEST needs its tissue-segmentation weights before it can segment: {message}",
        hint=(
            f"Download {SEGMENTATION_CKPT} from:\n"
            f"  {SEGMENTATION_CKPT_URL}\n"
            "then record it (this edits TRIDENT's registry in place):\n"
            f"  python -m aurora hest-setup --ckpt /path/to/{SEGMENTATION_CKPT}\n"
            "TRIDENT's own error names trident/segmentation_models/local_ckpts.json "
            "relative to its package, and prints the placeholder value where the "
            "checkpoint path belongs."
        ),
    )


def diagnose() -> List[str]:
    """What TRIDENT/HEST are looking for, read off the installed packages."""
    lines: List[str] = []
    trident = find_trident_package()
    hest = find_hest_package()
    lines.append(f"trident package : {trident if trident else '(not importable)'}")
    lines.append(f"HEST package    : {hest if hest else '(not importable)'}")

    registry = find_registry()
    lines.append(f"registry        : {registry if registry else '(none found)'}")
    if registry is None:
        lines.append("  looked in:")
        for candidate in registry_candidates():
            lines.append(f"    - {candidate}")
    else:
        try:
            data = read_registry(registry)
        except ConfigError as exc:
            lines.append(f"  unreadable: {exc.message}")
        else:
            for key in sorted(data):
                value = str(data[key])
                resolved = Path(value).expanduser()
                if not value.strip():
                    state = "EMPTY -- not configured"
                elif value == PLACEHOLDER_VALUE:
                    state = "PLACEHOLDER -- not a path"
                elif resolved.is_file():
                    state = "exists"
                else:
                    state = "MISSING"
                lines.append(f"  {key} -> {value}  [{state}]")
            lines.append(f"  (TRIDENT asks for the model by name: {MODEL_NAME!r})")
    return lines
