"""Reclaim disk space from a finished run, without touching its results.

A completed AURORA run keeps a lot of material that is only useful while the run
is being produced: per-superpatch caches, the derived plot views, segmentation
scratch, the rescaled slides and the multi-scale embeddings.  On a real slide
these are the bulk of the run directory.

Two rules shape this module:

1. **Results are never candidates.**  Everything under
   ``03_predictions/*/pred/`` and ``05_plots/`` -- along with the masks, the
   resolved inputs in ``config/``, the logs and ``run.json`` -- is kept.  There is
   no option to delete them.
2. **A run without results is not cleaned.**  Removing the embeddings and caches
   of a run that has not produced a prediction would throw away the expensive
   work and leave nothing behind, so that is refused unless explicitly forced.

The items fall into two tiers:

``scratch``
    Regenerated on demand and needed by nothing that already exists: derived
    plot views, per-superpatch caches, segmentation scratch, check figures, raw
    embeddings, the chunked-extraction temp directories.  Safe by default.

``recomputable``
    Expensive to rebuild, and *currently needed* by existing outputs:
    ``wsis_rescaled/`` is read again by ``code/plot.py`` whenever a figure is
    drawn, and ``emb_*/`` is read by a re-run of the prediction stages.  Removed
    only with ``deep=True``, after which plotting or re-predicting means running
    the earlier stages again.

Standard library only.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .errors import ConfigError
from .paths import (
    LEGACY_FEATURE_BUNDLE_DIRNAME,
    RunLayout,
)

#: Fields the two tiers are named by.
TIER_SCRATCH = "scratch"
TIER_RECOMPUTABLE = "recomputable"

#: Subdirectories of ``02_features/UNI_multiscale_patches`` that hold temporary
#: working files rather than embeddings.
FEATURE_SCRATCH_DIRNAMES: Tuple[str, ...] = ("strip_tmp", "HistoSweep_tmp")

#: Segmentation subdirectories that are scratch, not the tissue masks themselves.
SEGMENTATION_SCRATCH_SUBDIRS: Tuple[str, ...] = ("contours", "patches", "viz", "stitches")

#: Suffixes `predict_*` writes.  Used to decide whether a run produced results, so
#: a stray file in `pred/` cannot make an unfinished run look finished.
PREDICTION_SUFFIXES: Tuple[str, ...] = (".pickle", ".h5", ".npz.zst")


def _measure(path: Path) -> Tuple[int, int]:
    """``(bytes, file count)`` for a file or directory tree."""
    if path.is_file():
        try:
            return path.stat().st_size, 1
        except OSError:
            return 0, 0
    total = 0
    files = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        try:
            total += child.stat().st_size
        except OSError:
            continue
        files += 1
    return total, files


def human_bytes(size: float) -> str:
    """``1536`` -> ``'1.5 KiB'``.  Uses binary units, as ``du`` does."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


@dataclass(frozen=True)
class CleanItem:
    """One thing the plan would remove."""

    path: Path
    tier: str
    what: str
    size_bytes: int
    files: int

    @property
    def size_text(self) -> str:
        return human_bytes(self.size_bytes)


@dataclass
class CleanPlan:
    """What a cleanup would do, and why it leaves everything else alone.

    :attr:`items` is exactly what :func:`apply_cleanup` removes.  The
    recomputable tier is only *in* the plan when ``deep`` is set; otherwise it is
    listed in :attr:`deferred` so the report can say what ``--deep`` would add
    without any risk of it being deleted.
    """

    layout: RunLayout
    deep: bool
    items: List[CleanItem] = field(default_factory=list)
    deferred: List[CleanItem] = field(default_factory=list)
    kept: List[Tuple[Path, str]] = field(default_factory=list)
    results: List[Path] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.items)

    @property
    def total_files(self) -> int:
        return sum(item.files for item in self.items)

    @property
    def has_results(self) -> bool:
        return bool(self.results)

    def by_tier(self, tier: str) -> List[CleanItem]:
        return [item for item in self.items if item.tier == tier]

    def describe(self) -> str:
        lines = [f"Cleanup plan for {self.layout.root}"]
        if self.has_results:
            lines.append(f"  results to keep ({len(self.results)}):")
            for path in self.results:
                lines.append(f"    {path.relative_to(self.layout.root).as_posix()}")
        else:
            lines.append("  results: NONE -- this run has produced no prediction yet")
        lines.append("")
        for tier, heading in (
            (TIER_SCRATCH, "scratch (safe: rebuilt on demand)"),
            (TIER_RECOMPUTABLE, "recomputable (only with --deep)"),
        ):
            shown = self.by_tier(tier)
            if not shown:
                continue
            lines.append(f"  {heading}:")
            for item in sorted(shown, key=lambda i: -i.size_bytes):
                rel = item.path.relative_to(self.layout.root).as_posix()
                lines.append(f"    {item.size_text:>10}  {rel}  ({item.what})")
        if self.deferred:
            skipped = sum(item.size_bytes for item in self.deferred)
            lines.append("")
            lines.append(
                f"  NOT in this plan: {len(self.deferred)} recomputable item(s), "
                f"{human_bytes(skipped)} -- pass --deep to include them."
            )
            lines.append(
                "  They are read again by code/plot.py (rescaled slides) and by the "
                "prediction stages (embeddings), so removing them means re-running "
                "rescale/features before plotting or re-predicting."
            )
        lines.append("")
        if not self.items and not self.deferred:
            lines.append("  nothing to clean: the intermediates are already gone")
        elif not self.items:
            lines.append("  nothing to clean at this level (see --deep below)")
        else:
            lines.append(
                f"  would free {human_bytes(self.total_bytes)} in {self.total_files} file(s)"
            )
        if self.kept:
            lines.append("  kept:")
            for path, why in self.kept:
                try:
                    rel = path.relative_to(self.layout.root).as_posix()
                except ValueError:
                    rel = str(path)
                lines.append(f"    {rel}  ({why})")
        return "\n".join(lines)


def _find_results(layout: RunLayout) -> List[Path]:
    """The prediction files that prove this run produced something."""
    found: List[Path] = []
    for kind in ("lowres", "highres"):
        directory = layout.prediction_output_dir(kind)
        if not directory.is_dir():
            continue
        found.extend(
            sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix in PREDICTION_SUFFIXES
            )
        )
    return found


def plan_cleanup(layout: RunLayout, deep: bool = False) -> CleanPlan:
    """Work out what can be removed from ``layout``.

    With ``deep=False`` the returned plan contains only the scratch tier, so
    applying it can never remove an embedding or a rescaled slide.
    """
    plan = CleanPlan(layout=layout, deep=deep)
    plan.results = _find_results(layout)
    scratch: List[CleanItem] = []
    recomputable: List[CleanItem] = []

    def add(path: Path, tier: str, what: str) -> None:
        if not path.exists():
            return
        size, files = _measure(path)
        if size == 0:
            # Nothing to reclaim: an empty directory, or one holding only
            # zero-byte files.  Listing it would just be noise.
            return
        item = CleanItem(path=path, tier=tier, what=what, size_bytes=size, files=files)
        (scratch if tier == TIER_SCRATCH else recomputable).append(item)

    # ---- scratch --------------------------------------------------------- #
    add(layout.tmp_dir, TIER_SCRATCH, "derived plot views and staging")
    for kind in ("lowres", "highres"):
        add(
            layout.prediction_cache_dir(kind),
            TIER_SCRATCH,
            f"{kind} per-superpatch cache",
        )
    for sub in SEGMENTATION_SCRATCH_SUBDIRS:
        add(
            layout.segmentation_dir / sub,
            TIER_SCRATCH,
            "segmentation scratch",
        )
    add(layout.check_figure_dir, TIER_SCRATCH, "feature check figures")
    for name in FEATURE_SCRATCH_DIRNAMES:
        add(layout.feature_bundle_dir / name, TIER_SCRATCH, "extraction temp directory")
    # `<sample>_raw.npz` is written by the high-resolution extractor for extra
    # precision and is not read by any stage in this repository.
    if layout.feature_bundle_dir.is_dir():
        for raw in sorted(layout.feature_bundle_dir.glob("emb_*/*_raw.npz")):
            add(raw, TIER_SCRATCH, "raw (unaggregated) embeddings")

    # ---- recomputable ---------------------------------------------------- #
    add(
        layout.rescaled_dir,
        TIER_RECOMPUTABLE,
        "rescaled slides and tissue masks -- code/plot.py reads these to draw figures",
    )
    if layout.feature_bundle_dir.is_dir():
        for embeddings in sorted(layout.feature_bundle_dir.glob("emb_*")):
            if embeddings.is_dir():
                add(
                    embeddings,
                    TIER_RECOMPUTABLE,
                    "multi-scale embeddings -- the prediction stages read these",
                )
    add(layout.histosweep_dir, TIER_RECOMPUTABLE, "HistoSweep native outputs")

    # ---- explicitly kept ------------------------------------------------- #
    kept: List[Tuple[Path, str]] = [
        (layout.masks_dir, "tissue masks (segmentation result)"),
        (layout.prediction_output_dir("lowres"), "low-resolution predictions"),
        (layout.prediction_output_dir("highres"), "high-resolution predictions"),
        (layout.plots_dir, "figures"),
        (layout.config_dir, "resolved inputs, for provenance"),
        (layout.logs_dir, "stage logs"),
        (layout.run_json, "run state"),
        (layout.state_dir, "layout stamp"),
    ]
    if layout.checkpoints_dir.exists():
        kept.append((layout.checkpoints_dir, "trained checkpoints and assets"))
    plan.kept = kept

    # The recomputable tier is only part of the plan when --deep was asked for.
    plan.items = scratch + (recomputable if deep else [])
    plan.deferred = [] if deep else recomputable
    return plan


def apply_cleanup(plan: CleanPlan, force: bool = False) -> int:
    """Remove the planned items and return the bytes freed.

    Refuses to touch the ``recomputable`` tier when the run has no results: at
    that point the intermediates are the only thing of value, and the expensive
    stages would have to run again.
    """
    if not plan.has_results and not force:
        raise ConfigError(
            f"{plan.layout.root} has produced no prediction yet, so its intermediate "
            "files are all that exists",
            hint=(
                "Finish the run first, or pass --force if the run really is abandoned.\n"
                "To see what would be removed without deleting anything, run the same "
                "command without --apply."
            ),
        )

    freed = 0
    for item in plan.items:
        try:
            if item.path.is_dir():
                shutil.rmtree(item.path)
            else:
                item.path.unlink()
        except OSError as exc:
            raise ConfigError(
                f"could not remove {item.path}: {exc}",
                hint=(
                    "Something may still be using it (an editor, a running stage, or a "
                    "sandbox denying delete). Re-run after it is released."
                ),
            ) from exc
        freed += item.size_bytes
    return freed


def find_runs(project: Path | str, run_root: Optional[Path | str] = None) -> List[Path]:
    """Every run directory under ``<project>/aurora_runs``, plus ``run_root``."""
    from .paths import RUNS_DIRNAME

    found: List[Path] = []
    if run_root is not None:
        root = Path(run_root).expanduser().resolve()
        if root.is_dir():
            found.append(root)
    base = Path(project).expanduser().resolve() / RUNS_DIRNAME
    if base.is_dir():
        found.extend(
            sorted(child for child in base.iterdir() if child.is_dir() and child not in found)
        )
    return found
