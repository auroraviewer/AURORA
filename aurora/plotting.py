"""Feeding ``code/plot.py`` the layout it expects.

``code/plot.py`` reads the **legacy** prediction layout (3-D ``exp_pred`` /
``prop_pred`` arrays plus ``patches_loc_before_pad``), while
``aurora predict_lowres`` writes the **modern** layout by default, to stay
byte-compatible with ``code/predict.py``.

Rather than make the user remember ``--output-format legacy`` — and re-run the
whole prediction to get it — the ``plot`` stage projects the existing predictions
into a small, derived *view* directory and points ``code/plot.py`` at that:

    aurora_runs/<run>/tmp/legacy_pred_view/pred/<sample>-pred.pickle

The view holds **one** resolution, which is all ``code/plot.py`` needs for a run
(``--resolution`` selects the level).  Nothing in ``03_predictions`` is touched,
the view is disposable, and it is rebuilt when the source file is newer.

Standard library only; NumPy is imported through :mod:`aurora.predictions`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import ConfigError
from .paths import RunLayout
from .predictions import (
    HIGHRES_SOURCE_SUFFIXES,
    LAYOUT_LEGACY,
    LOWRES_SOURCE_SUFFIX,
    choose_resolution,
    load_any_prediction,
    write_legacy_view,
)
from .version import __version__

#: Directory inside ``<run>/tmp`` that mirrors ``pred/`` for the legacy reader.
VIEW_DIRNAME = "legacy_pred_view"

#: Which prediction stage to plot.
KIND_LOWRES = "lowres"
KIND_HIGHRES = "highres"

#: One view per kind, so plotting one never clobbers the other.
VIEW_DIRNAMES: Dict[str, str] = {
    KIND_LOWRES: VIEW_DIRNAME,
    KIND_HIGHRES: "legacy_pred_view_highres",
}

#: Files ``predict_*`` may have written, best first.  High resolution defaults to
#: the compact HDF5 that ``--uint8-output`` produces, so it must be accepted here
#: or plotting would break the moment the default is used.
SOURCE_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    KIND_LOWRES: (LOWRES_SOURCE_SUFFIX,),
    KIND_HIGHRES: HIGHRES_SOURCE_SUFFIXES,
}

#: The name ``code/plot.py`` opens unconditionally (line 69), whatever produced
#: it.  The view renames into this, which is why high-resolution predictions --
#: written as ``<sample>-highres-pred.pickle`` or ``.h5`` -- can be plotted.
TARGET_FILENAME = "{sample}-pred.pickle"

#: High-resolution outputs ``code/plot.py`` cannot read even after conversion.
HIGHRES_NON_PICKLE: Tuple[str, ...] = ("-highres-pred.npz.zst",)

#: Name of the build record written next to the view.
VIEW_META_FILENAME = "view.json"

#: Bump whenever the projection changes shape or types.  A view built by an older
#: converter is rebuilt rather than reused, so a fix cannot be defeated by a
#: stale file on disk.
#: 1 -- initial (``cell_types`` written as a list, which code/plot.py rejects)
#: 2 -- ``cell_types`` written as an array, as code/predict.py does
VIEW_FORMAT_VERSION = 2


def source_filename(kind: str, sample: str) -> str:
    """The prediction file ``predict_<kind>`` writes for ``sample``."""
    return f"{sample}{SOURCE_SUFFIXES[validate_kind(kind)]}"


def validate_kind(kind: str) -> str:
    if kind not in SOURCE_SUFFIXES:
        raise ConfigError(
            f"unknown prediction kind {kind!r}",
            hint=f"Choose one of: {', '.join(sorted(SOURCE_SUFFIXES))}.",
        )
    return kind


def legacy_view_root(layout: RunLayout, kind: str = KIND_LOWRES) -> Path:
    return layout.tmp_dir / VIEW_DIRNAMES[validate_kind(kind)]


def source_filename(kind: str, sample: str) -> str:
    """The preferred prediction filename for ``sample`` (used in messages)."""
    kind = validate_kind(kind)
    return f"{sample}{SOURCE_SUFFIXES[kind][0]}"


def locate_source(prediction_dir: Path, kind: str, sample: str) -> Optional[Path]:
    """The prediction file for ``sample``, in whichever supported format exists."""
    kind = validate_kind(kind)
    for suffix in SOURCE_SUFFIXES[kind]:
        candidate = prediction_dir / "pred" / f"{sample}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def prediction_not_a_pickle_message(prediction_dir: Path, samples: Sequence[str]) -> str:
    """Explain a high-resolution output that cannot be plotted at all.

    HDF5 *is* readable (``aurora.h5in`` converts it into the legacy view), so only
    the compressed ``npz.zst`` remains unplottable.
    """
    found: List[str] = []
    for name in samples:
        for suffix in HIGHRES_NON_PICKLE:
            candidate = prediction_dir / "pred" / f"{name}{suffix}"
            if candidate.is_file():
                found.append(candidate.name)
    if not found:
        return ""
    return (
        "code/plot.py reads pickles only, but the high-resolution predictions were "
        f"written as {', '.join(sorted(set(found))[:3])}. "
        "Re-run the high-resolution stage without --compress-output to get a plottable "
        "file (.h5 or .pickle both work)."
    )


def _available_resolutions(path: Path) -> List[int]:
    """The patch levels a prediction file holds, without failing on a bad file."""
    try:
        prediction = load_any_prediction(path)
    except Exception:  # noqa: BLE001 - this only feeds an error message
        return []
    return sorted(r for r in prediction.resolutions if r in prediction.exp_pred)


def _resolution_hint(
    layout: RunLayout,
    prediction_dir: Path,
    kind: str,
    sample: str,
    requested: Optional[int],
    error: ConfigError,
) -> ConfigError:
    """Improve "resolution N is not in ..." when another prediction has that level.

    ``--plot-resolution 14`` against the low-resolution prediction used to say only
    "available: 224"; 14 is the high-resolution patch size, and the fix is one flag.
    """
    other_kind = KIND_HIGHRES if kind == KIND_LOWRES else KIND_LOWRES
    other_dir = prediction_dir.parent / other_kind
    elsewhere: List[int] = []
    located = locate_source(other_dir, other_kind, sample)
    if located is not None:
        elsewhere = _available_resolutions(located)

    here = _available_resolutions(locate_source(prediction_dir, kind, sample) or Path())
    if requested is not None and requested in elsewhere:
        return ConfigError(
            error.message,
            hint=(
                f"resolution {requested} belongs to the {other_kind} prediction "
                f"(available there: {', '.join(str(r) for r in elsewhere)}).\n"
                f"Plot it with:\n"
                f"  --plot-predictions {other_kind} --plot-resolution {requested}\n"
                f"The {kind} prediction has: {', '.join(str(r) for r in here) or 'none'}."
            ),
        )
    return ConfigError(
        error.message,
        hint=(
            f"{kind} resolutions: {', '.join(str(r) for r in here) or 'none'}; "
            f"{other_kind} resolutions: {', '.join(str(r) for r in elsewhere) or 'none'}.\n"
            "Pass --plot-resolution with one of the values for the prediction you are "
            "plotting (--plot-predictions selects it)."
        ),
    )


def build_view(
    layout: RunLayout,
    prediction_dir: Path,
    samples: Sequence[str],
    resolution: Optional[int] = None,
    force: bool = False,
    dry_run: bool = False,
    kind: str = KIND_LOWRES,
) -> Tuple[Path, List[str]]:
    """Project ``prediction_dir/pred/*.pickle`` into a legacy view.

    ``prediction_dir`` is the directory the pipeline uses as the legacy ``-t``
    argument (``03_predictions/lowres`` or ``03_predictions/highres``).  Returns
    ``(view_root, notes)`` where ``view_root`` is the value to hand to
    ``code/plot.py -t``.

    The source filename depends on ``kind``; the target is always the
    ``<sample>-pred.pickle`` that ``code/plot.py`` opens.

    When the source predictions are *already* legacy, the view is skipped and the
    directory itself is returned.  With ``dry_run`` nothing is written.
    """
    kind = validate_kind(kind)
    notes: List[str] = []
    source_dir = prediction_dir / "pred"
    sources: Dict[str, Path] = {}
    missing: List[str] = []
    for name in samples:
        located = locate_source(prediction_dir, kind, name)
        if located is None:
            missing.append(name)
        else:
            sources[name] = located
    if missing:
        described = ", ".join(missing[:3]) + (
            " (+%d more)" % (len(missing) - 3) if len(missing) > 3 else ""
        )
        if dry_run:
            # A dry run reports the plan; it must not fail because an input that
            # an earlier stage will produce is not there yet.
            notes.append(f"{len(missing)} prediction file(s) not present yet in {source_dir}: {described}")
            notes.append(
                f"would project them into {legacy_view_root(layout, kind)} for code/plot.py"
            )
            return legacy_view_root(layout, kind), notes
        other_format = prediction_not_a_pickle_message(prediction_dir, missing)
        raise ConfigError(
            f"{len(missing)} {kind} prediction file(s) are missing from {source_dir}: {described}",
            hint=other_format
            or (
                f"Run the {kind} prediction first:\n"
                f"  python -m aurora run --run {layout.run_name} --stage predict_{kind}"
            ),
        )

    # Already legacy for every sample?  Then plot straight from the source.
    layouts = {}
    loaded: Dict[str, Any] = {}
    for name in samples:
        prediction = load_any_prediction(sources[name])
        loaded[name] = prediction
        layouts[name] = prediction.layout

    # Validate the requested resolution up front, so a level that only exists in the
    # *other* prediction can be pointed at instead of failing with "available: 224".
    first = samples[0]
    try:
        choose_resolution(loaded[first], resolution)
    except ConfigError as exc:
        raise _resolution_hint(layout, prediction_dir, kind, first, resolution, exc) from exc
    loaded.clear()
    if all(layout == LAYOUT_LEGACY for layout in layouts.values()):
        notes.append(
            f"predictions in {source_dir} already use the legacy layout; "
            "code/plot.py reads them directly"
        )
        return prediction_dir, notes
    modern = [name for name, layout in layouts.items() if layout != LAYOUT_LEGACY]
    if modern and len(modern) != len(samples):
        raise ConfigError(
            "the prediction files mix the modern and legacy layouts: "
            + ", ".join(f"{n} ({layouts[n]})" for n in sorted(layouts)),
            hint=(
                f"Re-run predict_{kind} with a single --output-format so every sample "
                "uses the same layout."
            ),
        )

    view_root = legacy_view_root(layout, kind)
    view_pred = view_root / "pred"
    if dry_run:
        notes.append(
            f"would project {len(samples)} prediction(s) into {view_root} for code/plot.py"
        )
        return view_root, notes
    if force and view_root.exists():
        shutil.rmtree(view_root)
    view_pred.mkdir(parents=True, exist_ok=True)

    # A derived view must be invalidated when the *producer* changes, not only
    # when the source file does.  An mtime-only check happily reused a view built
    # by an older converter, which kept a fixed bug alive on disk.  The sidecar
    # records the converter version and what each entry was built from.
    meta_path = view_root / VIEW_META_FILENAME
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    stale_reason = ""
    if meta.get("view_format") != VIEW_FORMAT_VERSION:
        stale_reason = (
            "the legacy view was built by an older converter and cannot be trusted"
            if meta
            else "no build record next to the legacy view"
        )
    elif meta.get("kind") != kind:
        # The view root is per kind, so this only happens if a record was copied
        # or the directory was reused; do not mix low- and high-resolution data.
        stale_reason = (
            f"the legacy view records kind {meta.get('kind')!r}, not {kind!r}"
        )
    entries: Dict[str, Any] = dict(meta.get("samples") or {})
    if stale_reason:
        notes.append(f"{stale_reason}; rebuilding it")

    for name in samples:
        source = sources[name]
        # code/plot.py always opens '<sample>-pred.pickle', whatever produced it.
        target = view_pred / TARGET_FILENAME.format(sample=name)
        prediction = load_any_prediction(source)
        chosen = choose_resolution(prediction, resolution)
        for warning in prediction.warnings:
            notes.append(f"{name}: {warning}")
        recorded = entries.get(name) or {}
        stat = source.stat()
        reusable = (
            not stale_reason
            and target.is_file()
            # The resolution is a *parameter* of the projection: changing
            # --plot-resolution must rebuild, or the wrong level gets plotted.
            and recorded.get("resolution") == chosen
            and recorded.get("source_mtime") == stat.st_mtime
            and recorded.get("source_size") == stat.st_size
            and target.stat().st_mtime >= stat.st_mtime
        )
        if reusable:
            notes.append(f"{name}: legacy view up to date at {chosen} px")
            entries[name] = recorded
            continue
        write_legacy_view(prediction, target, chosen)
        entries[name] = {
            "resolution": chosen,
            "source_mtime": stat.st_mtime,
            "source_size": stat.st_size,
            "spots": int(len(prediction.exp_pred[chosen])),
        }
        notes.append(
            f"{name}: projected the {kind} {chosen} px level into the legacy view "
            f"({len(prediction.exp_pred[chosen])} spots) for code/plot.py"
        )

    meta_path.write_text(
        json.dumps(
            {
                "view_format": VIEW_FORMAT_VERSION,
                "built_by": f"aurora {__version__}",
                "kind": kind,
                "prediction_dir": str(prediction_dir),
                "samples": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return view_root, notes
