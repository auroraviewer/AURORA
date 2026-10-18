"""Per-sample artefact naming, and the one place the code base disagrees with it.

The convention
--------------
Every per-sample artefact in AURORA is keyed by ``Sample_name``::

    <rescaled>/<Sample_name>_rescaled.tif
    <features>/emb_224/<Sample_name>.npz
    <predictions>/<Sample_name>-pred.pickle
    <predictions>/<Sample_name>-highres-pred.h5

The exception
-------------
``code/create_patches_fp.py`` names its tissue masks after the *slide file* it
found in ``--source``::

    slide_id, _ = os.path.splitext(slide)          # slide = a file name
    WSI_object.saveSegmentation(os.path.join(mask_save_dir, slide_id + '.geojson'))

When ``Image_name`` differs from ``Sample_name`` -- the usual TCGA case, where
the GDC appends a file UUID, e.g. ``TCGA-44-5645-01B-04-BS4.76ED5128-….svs`` for
sample ``TCGA-44-5645-01B-04-BS4`` -- the mask lands under the long image name
while everything downstream looks for the short sample name.  ``code/rescale.py``
asks for the image-file spelling, because it derives the mask name from the
``Image_name`` column of the sheet it is given.

The reconciliation
------------------
After segmentation this module makes sure **both** spellings exist, by creating a
hard link (never a copy of anything large, never a deletion, never a rename):

* if only ``<image_stem>.geojson`` exists, link it to ``<Sample_name>.geojson``,
  the canonical name the pipeline declares and reports;
* if only ``<Sample_name>.geojson`` exists, link it to ``<image_stem>.geojson``,
  the name ``code/rescale.py`` looks up.

The two names share one inode, so editing either changes both, and no existing
file is ever overwritten.

Standard library only.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .paths import RunLayout
from .samples import Sample, SampleSheet

#: File suffixes ``code/create_patches_fp.py`` writes for each slide.
SEGMENTATION_SUFFIXES: Tuple[str, ...] = (".geojson", ".png")

#: Low-resolution prediction file (one name for both pickle layouts).
LOWRES_SUFFIX = "-pred.pickle"

#: High-resolution prediction files.  Which one is written depends on the two
#: output flags, exactly as in ``code/predict_highres.py``:
#:
#: ==============  ================  ======================
#: ``--uint8``     ``--compress``    file
#: ==============  ================  ======================
#: off             (either)          ``-highres-pred.pickle``
#: on              off               ``-highres-pred.h5``
#: on              on                ``-highres-pred.npz.zst``
#: ==============  ================  ======================
HIGHRES_SUFFIXES: Dict[str, str] = {
    "pickle": "-highres-pred.pickle",
    "h5": "-highres-pred.h5",
    "zst": "-highres-pred.npz.zst",
}


def lowres_output_path(out_dir: Path, sample: str) -> Path:
    """The file ``predict_lowres`` writes for ``sample``.

    One name for both pickle layouts (``--output-format modern`` and
    ``legacy``), so this needs no flags.
    """
    return Path(out_dir) / f"{sample}{LOWRES_SUFFIX}"


def highres_output_kind(uint8_output: bool, compress_output: bool) -> str:
    """``'pickle'`` / ``'h5'`` / ``'zst'`` for the two output flags."""
    if not uint8_output:
        return "pickle"
    return "zst" if compress_output else "h5"


def highres_output_path(
    out_dir: Path, sample: str, uint8_output: bool, compress_output: bool
) -> Path:
    """The file ``predict_highres`` writes for ``sample``.

    Single source of truth: both the writer and the stage's declared output use
    this, so the two can never disagree about which of the three files the flags
    select.
    """
    return Path(out_dir) / f"{sample}{HIGHRES_SUFFIXES[highres_output_kind(uint8_output, compress_output)]}"


def highres_output_candidates(out_dir: Path, sample: str) -> List[Path]:
    """Every file ``predict_highres`` could have written for ``sample``."""
    return [Path(out_dir) / f"{sample}{suffix}" for suffix in HIGHRES_SUFFIXES.values()]


#: ``run_HistoSweep.py --mask-suffix`` default.  The mask it writes is consumed
#: by ``predict_highres``, so the name has one definition here.
HISTOSWEEP_MASK_SUFFIX = "HistoSweep"


def histosweep_mask_path(rescaled_dir: Path, sample: str) -> Path:
    """``<rescaled_dir>/<sample>_HistoSweep_mask_rescaled.tif``.

    HistoSweep sweeps the rescaled H&E image and writes a 0/1 tissue mask next to
    it, aligned pixel-for-pixel with ``<sample>_rescaled.tif``.
    """
    return Path(rescaled_dir) / f"{sample}_{HISTOSWEEP_MASK_SUFFIX}_mask_rescaled.tif"


@dataclass(frozen=True)
class SampleNaming:
    """The spellings a single sample can be referred to by."""

    sample_name: str
    image_name: str
    image_stem: str
    extension: str

    @property
    def names_differ(self) -> bool:
        return self.sample_name != self.image_stem


def naming_for(sample: Sample) -> SampleNaming:
    """Resolve the naming variants of one sample-sheet row.

    ``image_stem`` uses ``os.path.splitext``, which is exactly what
    ``code/rescale.py`` does with ``'.'.join(image_file.split('.')[:-1])``.
    """
    image = sample.image_name or sample.sample_name
    stem, extension = os.path.splitext(image)
    return SampleNaming(
        sample_name=sample.sample_name,
        image_name=image,
        image_stem=stem or sample.sample_name,
        extension=extension,
    )


def naming_map(sheet: SampleSheet) -> Dict[str, SampleNaming]:
    """``Sample_name -> SampleNaming`` for a whole sheet."""
    return {sample.sample_name: naming_for(sample) for sample in sheet}


def segmentation_artifacts(naming: SampleNaming, masks_dir: Path, suffix: str = ".geojson") -> Tuple[Path, Path]:
    """Return ``(canonical, image_named)`` paths for one segmentation artefact."""
    canonical = masks_dir / f"{naming.sample_name}{suffix}"
    image_named = masks_dir / f"{naming.image_stem}{suffix}"
    return canonical, image_named


def _link(source: Path, destination: Path) -> Optional[str]:
    """Hard-link ``source`` to ``destination``; return a note or ``None``."""
    if destination.exists() or not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return f"linked {source.name} <-> {destination.name} (one inode, no extra storage)"
    except OSError:
        # Different file system, or a file system without hard links.  Masks are
        # small GeoJSON/PNG files, so a copy is acceptable here.
        try:
            destination.write_bytes(source.read_bytes())
            return f"copied {source.name} -> {destination.name} (hard links unavailable)"
        except OSError as exc:  # pragma: no cover - permissions
            return f"could not derive {destination.name} from {source.name}: {exc}"


def reconcile_segmentation_names(
    layout: RunLayout,
    sheet: SampleSheet,
    suffixes: Sequence[str] = SEGMENTATION_SUFFIXES,
) -> List[str]:
    """Make sure each sample has its mask under both the sample and image name.

    Idempotent.  Returns human-readable notes describing what was linked, for the
    stage log.  Does nothing when the two names agree, which is the common case.
    """
    notes: List[str] = []
    for sample in sheet:
        naming = naming_for(sample)
        if not naming.names_differ:
            continue
        for suffix in suffixes:
            canonical, image_named = segmentation_artifacts(naming, layout.masks_dir, suffix)
            if canonical.exists() and image_named.exists():
                continue
            for source, destination in ((image_named, canonical), (canonical, image_named)):
                if source.exists() and not destination.exists():
                    note = _link(source, destination)
                    if note:
                        notes.append(note)
                    break
    return notes


def segmentation_output_groups(layout: RunLayout, sheet: SampleSheet) -> List[List[Path]]:
    """Acceptable GeoJSON mask paths per sample, canonical first.

    Used by the stage gate: a sample counts as segmented when *either* spelling
    is on disk, so masks produced before the naming was unified are still
    recognised.  The canonical (``Sample_name``) path is what gets recorded.
    """
    groups: List[List[Path]] = []
    for sample in sheet:
        naming = naming_for(sample)
        canonical, image_named = segmentation_artifacts(naming, layout.masks_dir)
        groups.append([canonical] if canonical == image_named else [canonical, image_named])
    return groups


def rescue_hint(layout: RunLayout, sheet: SampleSheet) -> str:
    """A concrete diagnosis for a missing segmentation mask."""
    differing = [n for n in (naming_for(s) for s in sheet) if n.names_differ]
    lines: List[str] = []
    if differing:
        example = differing[0]
        lines.append(
            "This sheet names the slide and the sample differently, for example:\n"
            f"    Image_name  = {example.image_name}\n"
            f"    Sample_name = {example.sample_name}\n"
            "AURORA keys every per-sample artefact by Sample_name; "
            "code/create_patches_fp.py writes its masks named after the slide file, "
            "so the pipeline links the two spellings together after segmentation. "
            "Neither spelling exists here, which means the segmentation step found no "
            "slide to process."
        )
    else:
        lines.append(
            "Image_name and Sample_name are identical for every row, so the mask should "
            "have been written directly.  The segmentation step found no slide to process."
        )
    lines.append(
        "Check that:\n"
        f"  * the slides really are in the --image-path folder (an 'Image_folder' column is "
        "needed when each slide sits in its own sub-directory),\n"
        "  * the sheet is not pointing at a directory of per-slide folders without it,\n"
        "  * the images have an extension create_patches_fp.py recognises (.svs, .tif, ...)."
    )
    lines.append(
        "The stage log lists every 'processing <slide>' line and the final count. "
        f"List what was produced with:\n    ls {layout.masks_dir}"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Resolving the slide file a sheet row refers to
# --------------------------------------------------------------------------- #


@dataclass
class ResolvedImage:
    """Outcome of looking for one sample's slide on disk."""

    sample_name: str
    declared: str
    #: File name to use in the sheet handed to the legacy scripts.
    resolved_name: str
    #: Absolute path of the file that was found, if any.
    path: Optional[Path] = None
    #: Candidate paths that were tried, for error messages.
    tried: List[Path] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def changed(self) -> bool:
        return self.resolved_name != self.declared


def image_candidates(image_root: Path, naming: SampleNaming, image_folder: str = "") -> List[Path]:
    """Paths that could hold this sample's slide, most specific first.

    The declared ``Image_name`` wins; if it is not on disk we also accept the
    ``Sample_name`` spelling with the same extension, which is what a user gets
    after renaming GDC files to their sample barcodes.
    """
    candidates: List[Path] = []
    declared = Path(naming.image_name)
    if declared.is_absolute():
        candidates.append(declared)
    else:
        candidates.append(image_root / declared)
        if image_folder:
            candidates.append(image_root / image_folder / declared.name)
    sample_named = f"{naming.sample_name}{naming.extension}"
    if sample_named != declared.name:
        candidates.append(image_root / sample_named)
        if image_folder:
            candidates.append(image_root / image_folder / sample_named)
    unique: List[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def resolve_image(
    image_root: Path,
    sample: Sample,
    naming: Optional[SampleNaming] = None,
) -> ResolvedImage:
    """Find the slide file for one sample sheet row."""
    naming = naming or naming_for(sample)
    candidates = image_candidates(image_root, naming, sample.image_folder)
    for candidate in candidates:
        if candidate.is_file():
            # The legacy scripts join this value onto their --image-path, so the
            # resolved value is always the bare file name.
            return ResolvedImage(
                sample_name=sample.sample_name,
                declared=naming.image_name,
                resolved_name=candidate.name,
                path=candidate,
                tried=candidates,
            )
    return ResolvedImage(
        sample_name=sample.sample_name,
        declared=naming.image_name,
        resolved_name=naming.image_name,
        path=None,
        tried=candidates,
    )


def resolve_sheet_images(
    sheet: SampleSheet,
    image_root: Path,
) -> Tuple[SampleSheet, List[ResolvedImage], List[str]]:
    """Return ``(resolved_sheet, resolutions, notes)``.

    The resolved sheet is identical to ``sheet`` except that ``Image_name`` is
    set to the file name that actually exists on disk.  That single change makes
    ``code/rescale.py`` build both a valid image path *and* a mask path that
    matches the mask the segmentation stage wrote.
    """
    resolutions: List[ResolvedImage] = []
    notes: List[str] = []
    samples: List[Sample] = []
    for sample in sheet:
        resolution = resolve_image(image_root, sample)
        resolutions.append(resolution)
        if resolution.found and resolution.changed:
            notes.append(
                f"{sample.sample_name}: the declared Image_name {resolution.declared!r} is not on "
                f"disk; using {resolution.resolved_name!r} instead (found next to the other slides)"
            )
            samples.append(dataclasses.replace(sample, image_name=resolution.resolved_name))
        else:
            samples.append(sample)
    resolved_sheet = SampleSheet(
        path=None,
        samples=samples,
        columns=list(sheet.columns),
        delimiter=sheet.delimiter,
        kind=sheet.kind,
        warnings=list(sheet.warnings),
    )
    return resolved_sheet, resolutions, notes


def missing_image_hint(image_root: Path, resolutions: Sequence[ResolvedImage]) -> str:
    """Explain which slides could not be found and where they were looked for."""
    missing = [r for r in resolutions if not r.found]
    lines = [
        f"{len(missing)} of {len(resolutions)} slide(s) named by the sample sheet were not found.",
        f"The slides are looked up under: {image_root}",
        "",
    ]
    for resolution in missing[:5]:
        lines.append(f"  {resolution.sample_name}  (Image_name = {resolution.declared})")
        lines.append("    tried:")
        for candidate in resolution.tried:
            lines.append(f"      {candidate}")
    if len(missing) > 5:
        lines.append(f"  (+{len(missing) - 5} more)")
    lines.append("")
    lines.append(
        "Fix the Image_name column, or put the slides where --image-path points. "
        "If each slide sits in its own sub-directory (TCGA layout), add an "
        "'Image_folder' column naming those sub-directories."
    )
    return "\n".join(lines)
