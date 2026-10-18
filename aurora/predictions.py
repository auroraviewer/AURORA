"""Reading AURORA prediction files, in whichever layout they were written.

Two layouts exist, and they are not interchangeable:

``modern`` (what ``code/predict.py`` and ``aurora``'s ``predict_lowres`` write by
default)
    ``exp_pred`` and ``log_prop_pred`` are **dicts keyed by patch size**, so each
    resolution keeps only the patches it actually resolved.

``legacy`` (what ``code/plot.py`` and ``code/run_istar.py`` read)
    ``exp_pred`` and ``prop_pred`` are **3-D arrays** ``[spot, level, feature]``.

The two differ in more than shape:

* ``code/plot.py`` also flips ``patches_loc_before_pad`` a second time and then
  treats column 0 as ``y``, so that key must be in ``(x, y)`` order, whereas the
  modern ``pos`` key is in ``(y, x)`` order to match ``code/predict.py``;
* the modern arrays are **deduplicated**, so a faithful multi-level legacy array
  cannot be reconstructed from them — level *L* has its own patch count.
  A single level, however, is exact: its predictions and its positions have the
  same length and can be projected into the legacy shape.

:func:`to_legacy` therefore converts **one resolution at a time**, which is all
``code/plot.py`` needs (it plots one ``--resolution`` per run).  To get a true
multi-level legacy file, run the prediction with
``--output-format legacy``.

NumPy is imported lazily, so importing this module stays dependency-free.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import ConfigError

#: Layout names.
LAYOUT_MODERN = "modern"
LAYOUT_LEGACY = "legacy"
#: The compact HDF5 output written when ``--uint8-output`` is on (the default).
LAYOUT_H5 = "h5"

#: Keys that may hold per-patch positions, in the order we trust them.
#: ``patches_loc_before_pad`` is ``(x, y)``; ``pos`` is ``(y, x)``.
POSITION_KEYS_XY: Tuple[str, ...] = ("patches_loc_before_pad",)
POSITION_KEYS_YX: Tuple[str, ...] = ("pos",)


def _numpy():
    from .deps import require

    return require("numpy", purpose="reading prediction files")


@dataclass
class Prediction:
    """One prediction file, normalised across both layouts."""

    path: Optional[Path]
    layout: str
    genes: List[str] = field(default_factory=list)
    cell_types: List[str] = field(default_factory=list)
    resolutions: List[int] = field(default_factory=list)
    #: ``{resolution: array[#spots, #genes]}``
    exp_pred: Dict[int, Any] = field(default_factory=dict)
    #: ``{resolution: array[#spots, #cell types]}`` -- log probabilities
    log_prop_pred: Dict[int, Any] = field(default_factory=dict)
    #: ``{resolution: array[#spots, 2]}`` -- always ``(x, y)``
    positions: Dict[int, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.stem if self.path else "(in memory)"

    def has_cell_types(self) -> bool:
        return bool(self.log_prop_pred)

    def describe(self) -> str:
        lines = [
            f"Prediction: {self.path or '(in memory)'}",
            f"  layout     : {self.layout}",
            f"  genes      : {len(self.genes)}",
            f"  cell types : {len(self.cell_types)}",
            f"  resolutions: {self.resolutions}",
        ]
        for resolution in self.resolutions:
            spots = len(self.exp_pred[resolution]) if resolution in self.exp_pred else 0
            lines.append(f"    {resolution:>6} px : {spots} spots")
        for warning in self.warnings:
            lines.append(f"  warning    : {warning}")
        return "\n".join(lines)


def detect_layout(payload: Dict[str, Any]) -> str:
    """``'modern'`` when ``exp_pred`` is a dict, ``'legacy'`` when it is an array."""
    exp = payload.get("exp_pred")
    if isinstance(exp, dict):
        return LAYOUT_MODERN
    return LAYOUT_LEGACY


def _reconcile_positions(
    payload: Dict[str, Any], resolution: int, warnings: List[str]
) -> Any:
    """Positions for one resolution, always in ``(x, y)`` order.

    ``pos`` is ``(y, x)`` -- that is the definition ``code/predict.py`` uses --
    while ``patches_loc_before_pad`` must be ``(x, y)`` because ``code/plot.py``
    flips it a second time.  The two keys are therefore redundant, which makes a
    wrong file *detectable* rather than a guess:

    * ``patches_loc_before_pad == pos``
        They cannot both be right (that needs a transpose-symmetric array), and
        ``pos`` is the one with a fixed definition, so
        ``patches_loc_before_pad`` is the transposed one.  Some files were
        written this way; flip it.
    * ``patches_loc_before_pad == flip(pos)``
        Both correct; use it unchanged.
    * anything else
        Report it and trust ``patches_loc_before_pad``, the key the legacy reader
        is documented to use.

    With only one of the two keys present, the documented order for that key is
    assumed and a warning records it.
    """
    xy = _lookup(payload, POSITION_KEYS_XY, resolution)
    yx = _lookup(payload, POSITION_KEYS_YX, resolution)
    if xy is None and yx is None:
        return None
    if yx is None:
        return xy
    if xy is None:
        warnings.append(
            f"no 'patches_loc_before_pad' for {resolution} px; "
            "flipped 'pos' from (y, x) to (x, y)"
        )
        return _flip(yx)
    if xy.shape != yx.shape:
        warnings.append(
            f"'pos' {yx.shape} and 'patches_loc_before_pad' {xy.shape} disagree in shape "
            f"for {resolution} px; using 'patches_loc_before_pad' as (x, y)"
        )
        return xy
    if _array_equal(xy, yx):
        warnings.append(
            f"the file stores the same (y, x) array under both 'pos' and "
            f"'patches_loc_before_pad' for {resolution} px (written before the orientation "
            "fix); using the transposed 'patches_loc_before_pad'"
        )
        return _flip(yx)
    if _array_equal(xy, _flip(yx)):
        return xy
    warnings.append(
        f"'pos' and 'patches_loc_before_pad' are consistent with neither orientation "
        f"for {resolution} px; using 'patches_loc_before_pad' as (x, y)"
    )
    return xy


def _lookup(payload: Dict[str, Any], keys: Sequence[str], resolution: int) -> Any:
    """The first of ``keys`` that is present, for a given resolution."""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            if resolution in value:
                return np_asarray(value[resolution])
        elif value is not None and _is_array(value):
            return np_asarray(value)
    return None


def np_asarray(value: Any) -> Any:
    """``numpy.asarray`` without importing NumPy at module load."""
    return _numpy().asarray(value)


def _array_equal(first: Any, second: Any) -> bool:
    np = _numpy()
    return bool(np.array_equal(first, second))


def _is_array(value: Any) -> bool:
    np = _numpy()
    return isinstance(value, np.ndarray)


def _flip(array: Any) -> Any:
    np = _numpy()
    array = np.asarray(array)
    return array[:, ::-1] if array.ndim == 2 and array.shape[1] == 2 else array


def load_prediction(path: Path | str) -> Prediction:
    """Read a prediction pickle of either layout into a :class:`Prediction`."""
    np = _numpy()
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"prediction file {path} does not exist")
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ConfigError(
            f"{path} does not contain a prediction dictionary (found {type(payload).__name__})"
        )

    warnings: List[str] = []
    layout = detect_layout(payload)
    resolutions = [int(x) for x in payload.get("resolutions", [])]
    genes = list(payload.get("genes", []))
    cell_types = list(payload.get("cell_types", []))

    exp_pred: Dict[int, Any] = {}
    log_prop_pred: Dict[int, Any] = {}
    raw_exp = payload.get("exp_pred")
    if layout == LAYOUT_MODERN:
        for resolution, array in raw_exp.items():
            exp_pred[int(resolution)] = np.asarray(array)
        if isinstance(payload.get("log_prop_pred"), dict):
            for resolution, array in payload["log_prop_pred"].items():
                log_prop_pred[int(resolution)] = np.asarray(array)
        elif isinstance(payload.get("prop_pred"), dict):
            for resolution, array in payload["prop_pred"].items():
                log_prop_pred[int(resolution)] = _log(np.asarray(array))
        if not resolutions:
            resolutions = sorted(exp_pred)
    else:
        array = np.asarray(raw_exp)
        if array.ndim != 3:
            raise ConfigError(
                f"{path}: the legacy 'exp_pred' must be 3-D [spot, level, gene], got shape {array.shape}"
            )
        if not resolutions:
            resolutions = list(range(1, array.shape[1] + 1))
        for index, resolution in enumerate(resolutions):
            if index < array.shape[1]:
                exp_pred[resolution] = array[:, index, :]
        for key in ("log_prop_pred", "prop_pred"):
            value = payload.get(key)
            if value is None or isinstance(value, dict):
                continue
            value = np.asarray(value)
            if value.ndim != 3:
                continue
            for index, resolution in enumerate(resolutions):
                if index < value.shape[1]:
                    log_prop_pred[resolution] = value[:, index, :] if key == "log_prop_pred" else _log(value[:, index, :])
            break

    positions: Dict[int, Any] = {}
    for resolution in resolutions:
        found = _reconcile_positions(payload, resolution, warnings)
        if found is not None:
            positions[resolution] = np.asarray(found)

    if not exp_pred:
        raise ConfigError(f"{path} contains no usable 'exp_pred' entry")
    missing_positions = [r for r in resolutions if r not in positions]
    if missing_positions:
        warnings.append(
            f"no positions recorded for {missing_positions}; code/plot.py needs "
            "'patches_loc_before_pad' for the resolution it plots"
        )
    return Prediction(
        path=path,
        layout=layout,
        genes=genes,
        cell_types=cell_types,
        resolutions=resolutions,
        exp_pred=exp_pred,
        log_prop_pred=log_prop_pred,
        positions=positions,
        warnings=warnings,
    )


def _log(array: Any) -> Any:
    np = _numpy()
    return np.log(np.clip(array.astype(np.float64), 1e-30, None))


def choose_resolution(
    prediction: Prediction, requested: Optional[int] = None
) -> int:
    """The resolution to build a view for.

    Defaults to the smallest one, which is the level ``code/plot.py`` shows by
    default and the only level whose modern array is the complete patch set.
    """
    available = [r for r in prediction.resolutions if r in prediction.exp_pred]
    if not available:
        raise ConfigError(f"{prediction.path} has no resolutions to plot")
    if requested is None:
        return min(available)
    if requested not in available:
        raise ConfigError(
            f"resolution {requested} is not in {prediction.path} "
            f"(available: {', '.join(str(r) for r in available)})",
            hint="Pass --plot-resolution with one of the available values.",
        )
    return requested


def to_legacy(prediction: Prediction, resolution: Optional[int] = None) -> Dict[str, Any]:
    """Project one resolution of ``prediction`` into the legacy array layout.

    Returns a payload ``code/plot.py`` can read:

    * ``exp_pred``   ``[spots, 1, genes]`` -- level 0 is the requested resolution
    * ``prop_pred``  ``[spots, 1, cell types]`` -- **probabilities**
    * ``patches_loc_before_pad`` ``{resolution: [spots, 2]}`` in ``(x, y)``
    * ``genes`` (list) and ``cell_types`` (array), matching what
      ``code/plot.py`` calls on each, plus ``resolutions`` = ``[resolution]``

    Only one level is produced: the modern payload is deduplicated, so other
    levels have a different spot count and cannot share the array.
    """
    np = _numpy()
    chosen = choose_resolution(prediction, resolution)
    if chosen not in prediction.positions:
        raise ConfigError(
            f"{prediction.path} has no patch positions for {chosen} px, so it cannot be plotted",
            hint=(
                "code/plot.py needs 'patches_loc_before_pad'. Re-run the low-resolution "
                "prediction, or run it with --output-format legacy."
            ),
        )
    exp = np.asarray(prediction.exp_pred[chosen])
    if exp.ndim != 2:
        raise ConfigError(f"{prediction.path}: exp_pred[{chosen}] has shape {exp.shape}, expected 2-D")
    spots = exp.shape[0]
    positions = np.asarray(prediction.positions[chosen])
    if positions.shape[0] != spots:
        raise ConfigError(
            f"{prediction.path}: exp_pred[{chosen}] has {spots} spots but "
            f"patches_loc_before_pad has {positions.shape[0]}"
        )

    payload: Dict[str, Any] = {
        "exp_pred": exp[:, None, :].copy(),
        "patches_loc_before_pad": {chosen: positions},
        "genes": list(prediction.genes),
        # code/plot.py calls ``results['cell_types'].tolist()`` but
        # ``results['genes'].index()``: genes must be a list, cell types must be
        # an array.  code/predict.py writes them the same asymmetric way.
        "cell_types": np.asarray(prediction.cell_types),
        "resolutions": [chosen],
        "converted_from": str(prediction.path) if prediction.path else "",
        "converted_layout": prediction.layout,
    }
    if chosen in prediction.log_prop_pred:
        prop = np.asarray(prediction.log_prop_pred[chosen])
        payload["prop_pred"] = np.exp(prop)[:, None, :].copy()
    return payload


def write_legacy_view(
    prediction: Prediction,
    destination: Path | str,
    resolution: Optional[int] = None,
) -> Path:
    """Write the legacy projection of one resolution to ``destination``."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = to_legacy(prediction, resolution)
    with open(destination, "wb") as handle:
        pickle.dump(payload, handle)
    return destination


def describe_payload(path: Path | str) -> str:
    """Short report used by ``tools/convert_prediction_format.py``."""
    return load_prediction(path).describe()


def load_highres_h5_prediction(path: Path | str) -> Prediction:
    """Read the compact high-resolution HDF5 output as a :class:`Prediction`.

    The file stores *quantised* uint8 matrices normalised by a per-channel
    maximum, so it is read back with ``dequantize=True``; rows are restored to
    their original order so the values line up with the stored positions.

    ``patch_loc_<ps>`` is in ``(y, x)`` order -- the same array the pickle stores
    under ``pos`` -- and is flipped here, because a :class:`Prediction` always
    exposes positions as ``(x, y)``.
    """
    from . import h5in

    np = _numpy()
    path = Path(path)
    data = h5in.load_highres_h5(path, restore_original_order=True, dequantize=True)
    exp_pred = {int(ps): np.asarray(arr) for ps, arr in data["exp_pred"].items()}
    prop_pred = {int(ps): np.asarray(arr) for ps, arr in data["prop_pred"].items()}
    positions_yx = {int(ps): np.asarray(arr) for ps, arr in data["pos"].items()}
    # code/plot.py expects (x, y); the file holds (y, x).
    positions = {ps: arr[:, ::-1] for ps, arr in positions_yx.items()}
    log_prop_pred = {
        ps: np.log(np.clip(arr.astype(np.float64), 1e-30, None)) for ps, arr in prop_pred.items()
    }
    resolutions = [int(r) for r in data.get("resolutions", [])] or sorted(exp_pred)
    warnings: List[str] = []
    if str(data.get("layout", "")) == "patch_major":
        warnings.append(
            "this HDF5 file uses the patch_major layout, so its rows are in the "
            "original order rather than a spatial order"
        )
    return Prediction(
        path=path,
        layout=LAYOUT_H5,
        genes=[str(g) for g in data.get("genes", [])],
        cell_types=[str(c) for c in data.get("cell_types", [])],
        resolutions=resolutions,
        exp_pred=exp_pred,
        log_prop_pred=log_prop_pred,
        positions=positions,
        warnings=warnings,
    )


#: Suffixes that carry a high-resolution prediction, best first.
HIGHRES_SOURCE_SUFFIXES: Tuple[str, ...] = (
    "-highres-pred.pickle",
    "-highres-pred.h5",
)

#: Suffix of the low-resolution prediction.
LOWRES_SOURCE_SUFFIX = "-pred.pickle"


def load_any_prediction(path: Path | str) -> Prediction:
    """Read a prediction in any supported format, chosen by suffix.

    The HDF5 writer is the default at high resolution, so the readers have to
    accept it; everything else is a pickle.
    """
    path = Path(path)
    if path.suffix == ".h5":
        return load_highres_h5_prediction(path)
    return load_prediction(path)
