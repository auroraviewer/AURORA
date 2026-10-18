"""Sample-sheet parsing and validation.

AURORA drives every stage from a single ``metadata`` CSV (the *sample sheet*).
The old code accepted slightly different columns per stage, which is one of the
reasons runs were hard to reproduce.  This module defines one canonical shape
and validates it once, up front, with errors that name the offending row.

Required columns
----------------
``Image_name``
    File name of the whole-slide image, including its extension.
``Sample_name``
    Unique key used to index every intermediate and output file.  Replicates are
    rejected because they would silently overwrite each other.

Optional columns
----------------
``Bulk_sample_name``
    Column of the bulk expression matrix to use (defaults to ``Sample_name``).
``Image_folder``
    Sub-directory of the WSI folder that holds the image (TCGA-style layout).
``Pixel_size_um``
    Micrometres per pixel, required for non-``.svs`` images.

Detection of training sheets
----------------------------
A *training* sheet lists Visium samples instead, with columns
``Sample``/``Dir_name``/``Image_name`` (see ``generate_sample_list.py``).  When
``--kind training`` is requested, ``Sample``/``Dir_name`` are renamed onto
``Sample_name``/``Image_folder`` so the rest of the pipeline sees one shape.

Standard library only.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .errors import SampleSheetError, format_error_list

#: Canonical column names.
COL_IMAGE = "Image_name"
COL_SAMPLE = "Sample_name"
COL_BULK = "Bulk_sample_name"
COL_FOLDER = "Image_folder"
COL_PIXEL_SIZE = "Pixel_size_um"

REQUIRED_INFERENCE_COLUMNS: Tuple[str, ...] = (COL_IMAGE, COL_SAMPLE)

#: Aliases accepted on input and mapped onto the canonical names.
COLUMN_ALIASES: Dict[str, str] = {
    "image_name": COL_IMAGE,
    "imagename": COL_IMAGE,
    "image": COL_IMAGE,
    "file name": COL_IMAGE,
    "file_name": COL_IMAGE,
    "filename": COL_IMAGE,
    "sample_name": COL_SAMPLE,
    "samplename": COL_SAMPLE,
    "sample": COL_SAMPLE,
    "sample id": COL_SAMPLE,
    "sample_id": COL_SAMPLE,
    "bulk_sample_name": COL_BULK,
    "bulk": COL_BULK,
    "image_folder": COL_FOLDER,
    "dir_name": COL_FOLDER,
    "dirname": COL_FOLDER,
    "folder": COL_FOLDER,
    "pixel_size_um": COL_PIXEL_SIZE,
    "pixel_size": COL_PIXEL_SIZE,
}

#: Columns of a Visium training sheet produced by ``generate_sample_list.py``.
TRAINING_COLUMNS: Tuple[str, ...] = ("Sample", "Dir_name", "Image_name")


@dataclass
class Sample:
    """One row of a sample sheet."""

    sample_name: str
    image_name: str = ""
    bulk_sample_name: str = ""
    image_folder: str = ""
    pixel_size_um: Optional[float] = None
    row_index: int = 0
    raw: Dict[str, str] = field(default_factory=dict)

    @property
    def image_stem(self) -> str:
        return os.path.splitext(self.image_name)[0] if self.image_name else self.sample_name

    def relative_image_path(self, image_root: str = "") -> str:
        """Image path relative to the WSI root, honouring ``Image_folder``."""
        parts = [p for p in (image_root, self.image_folder, self.image_name) if p]
        return "/".join(parts)


@dataclass
class SampleSheet:
    """A parsed, validated sample sheet."""

    path: Optional[Path]
    samples: List[Sample] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    delimiter: str = ","
    kind: str = "inference"
    warnings: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @property
    def sample_names(self) -> List[str]:
        return [s.sample_name for s in self.samples]

    @property
    def image_names(self) -> List[str]:
        return [s.image_name for s in self.samples if s.image_name]

    @property
    def has_folders(self) -> bool:
        return any(s.image_folder for s in self.samples)

    @property
    def has_pixel_sizes(self) -> bool:
        return any(s.pixel_size_um is not None for s in self.samples)

    @property
    def bulk_sample_names(self) -> List[str]:
        return [s.bulk_sample_name or s.sample_name for s in self.samples]

    def by_name(self, sample_name: str) -> Sample:
        for sample in self.samples:
            if sample.sample_name == sample_name:
                return sample
        raise KeyError(sample_name)

    def with_names(self, names: Sequence[str]) -> "SampleSheet":
        """A copy restricted to ``names`` (order preserved, unknown names rejected)."""
        wanted = list(names)
        index = {s.sample_name: s for s in self.samples}
        missing = [n for n in wanted if n not in index]
        if missing:
            raise SampleSheetError(
                f"unknown sample name(s) requested: {', '.join(missing)}",
                hint=f"Available samples: {', '.join(self.sample_names)}",
            )
        return SampleSheet(
            path=self.path,
            samples=[index[n] for n in wanted],
            columns=list(self.columns),
            delimiter=self.delimiter,
            kind=self.kind,
            warnings=list(self.warnings),
        )

    def fingerprint(self) -> str:
        """Stable hash of the sample set, used by the stage gate."""
        import hashlib

        digest = hashlib.sha256()
        for sample in self.samples:
            digest.update(
                "\x1f".join(
                    [
                        sample.sample_name,
                        sample.image_name,
                        sample.bulk_sample_name,
                        sample.image_folder,
                        "" if sample.pixel_size_um is None else f"{sample.pixel_size_um:g}",
                    ]
                ).encode("utf-8")
            )
            digest.update(b"\x1e")
        return "sha256:" + digest.hexdigest()

    # -- optional-column policy -------------------------------------------- #
    @property
    def has_bulk_names(self) -> bool:
        return any(s.bulk_sample_name for s in self.samples)

    def output_columns(self) -> List[str]:
        """Columns :func:`write_sample_sheet` will emit, in order.

        ``Image_name`` and ``Sample_name`` always come first, in that order,
        because ``code/extract_features_highres.py`` is given
        ``--sample_name_col 1``.  An optional column is only emitted when it
        actually carries information: ``code/rescale.py`` chooses the
        per-slide-subdirectory layout with ``'Image_folder' in meta.columns`` and
        ``code/predict.py`` does the same for ``Bulk_sample_name``, so an
        all-empty column makes them take the wrong branch -- with
        ``Image_folder`` that means joining a literal ``nan`` into every path.
        """
        columns = [COL_IMAGE, COL_SAMPLE]
        if self.has_bulk_names:
            columns.append(COL_BULK)
        if self.has_folders:
            columns.append(COL_FOLDER)
        if self.has_pixel_sizes:
            columns.append(COL_PIXEL_SIZE)
        return columns

    def to_rows(self) -> List[Dict[str, str]]:
        """Rows for :func:`write_sample_sheet`, with defaults for partial columns.

        A partially filled optional column cannot simply be left blank, because
        ``pandas.read_csv`` reads a blank field as ``NaN``: ``Image_folder`` gets
        ``.`` (a no-op path component, so the joined path stays correct) and
        ``Bulk_sample_name`` falls back to ``Sample_name``, which is the
        documented default.
        """
        columns = self.output_columns()
        rows: List[Dict[str, str]] = []
        for sample in self.samples:
            row = {COL_IMAGE: sample.image_name, COL_SAMPLE: sample.sample_name}
            if COL_BULK in columns:
                row[COL_BULK] = sample.bulk_sample_name or sample.sample_name
            if COL_FOLDER in columns:
                row[COL_FOLDER] = sample.image_folder or "."
            if COL_PIXEL_SIZE in columns:
                row[COL_PIXEL_SIZE] = (
                    "" if sample.pixel_size_um is None else f"{sample.pixel_size_um:g}"
                )
            rows.append(row)
        return rows

    def describe(self) -> str:
        lines = [
            f"Sample sheet: {self.path or '(in memory)'}",
            f"  kind      : {self.kind}",
            f"  samples   : {len(self.samples)}",
            f"  columns   : {', '.join(self.columns)}",
        ]
        if self.warnings:
            lines.append("  warnings  :")
            lines.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(lines)


def _canonical_column(name: str) -> str:
    stripped = name.strip()
    if stripped in (COL_IMAGE, COL_SAMPLE, COL_BULK, COL_FOLDER, COL_PIXEL_SIZE):
        return stripped
    return COLUMN_ALIASES.get(stripped.lower(), stripped)


def sniff_delimiter(header_line: str) -> str:
    """Pick the delimiter of the header line (``,`` or tab)."""
    return "\t" if header_line.count("\t") > header_line.count(",") else ","


def read_sample_sheet(
    path: os.PathLike | str,
    kind: str = "inference",
    *,
    require_image: Optional[bool] = None,
) -> SampleSheet:
    """Read and validate a sample sheet.

    Parameters
    ----------
    path:
        ``.csv`` or ``.tsv`` file.
    kind:
        ``'inference'`` (default) or ``'training'``.  Training sheets may use the
        ``Sample``/``Dir_name`` spelling produced by ``generate_sample_list.py``.
    require_image:
        Force the presence (``True``) or absence (``False``) of ``Image_name``.
        Defaults to ``True`` for inference sheets and ``False`` for training
        sheets that only list sample directories.
    """
    path = Path(path)
    if not path.is_file():
        raise SampleSheetError(
            f"sample sheet {path} does not exist",
            hint=(
                "Pass an existing CSV with --samples. See examples/Inference_samples.csv "
                "for the expected columns."
            ),
        )
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if not lines:
        raise SampleSheetError(f"sample sheet {path} is empty")
    delimiter = sniff_delimiter(lines[0])
    reader = csv.DictReader(lines, delimiter=delimiter)
    if reader.fieldnames is None:
        raise SampleSheetError(f"sample sheet {path} has no header row")

    rename = {name: _canonical_column(name) for name in reader.fieldnames}
    columns = [rename[name] for name in reader.fieldnames]

    if kind == "training" and COL_SAMPLE not in columns and "Sample" in columns:
        # generate_sample_list.py writes Sample/Dir_name/Image_name; they are
        # already canonical after aliasing, this is just a safety net.
        pass

    if require_image is None:
        require_image = kind != "training"

    missing = [c for c in REQUIRED_INFERENCE_COLUMNS if c not in columns]
    if missing and not (kind == "training" and COL_IMAGE in missing):
        raise SampleSheetError(
            f"sample sheet {path} is missing required column(s): {', '.join(missing)}",
            hint=(
                "Required columns are 'Image_name' and 'Sample_name'. "
                f"Found: {', '.join(reader.fieldnames)}\n"
                "For training sheets produced by generate_sample_list.py the columns are "
                "Sample/Dir_name/Image_name, which are accepted as aliases."
            ),
        )

    samples: List[Sample] = []
    warnings: List[str] = []
    seen: Dict[str, int] = {}
    for offset, raw_row in enumerate(reader, start=2):  # row 1 is the header
        row = {rename[k]: (v if v is not None else "") for k, v in raw_row.items() if k is not None}
        sample_name = (row.get(COL_SAMPLE) or "").strip()
        image_name = (row.get(COL_IMAGE) or "").strip()
        if not sample_name and not image_name:
            continue  # blank line
        if not sample_name:
            sample_name = os.path.splitext(image_name)[0]
            warnings.append(f"row {offset}: 'Sample_name' was empty, using the image stem {sample_name!r}")
        if sample_name in seen:
            raise SampleSheetError(
                f"duplicate Sample_name {sample_name!r} in {path} (rows {seen[sample_name]} and {offset})",
                hint=(
                    "Sample_name indexes every intermediate and output file, so replicates are "
                    "not allowed. Make the values unique, or split the sheet into two runs."
                ),
            )
        seen[sample_name] = offset
        if require_image and not image_name:
            raise SampleSheetError(
                f"row {offset} of {path} has no '{COL_IMAGE}' value (sample {sample_name!r})",
                hint="Every inference sample must name the whole-slide image file.",
            )
        pixel_size_raw = (row.get(COL_PIXEL_SIZE) or "").strip()
        pixel_size: Optional[float] = None
        if pixel_size_raw:
            try:
                pixel_size = float(pixel_size_raw)
            except ValueError as exc:
                raise SampleSheetError(
                    f"row {offset} of {path} has a non-numeric Pixel_size_um {pixel_size_raw!r}"
                ) from exc
        samples.append(
            Sample(
                sample_name=sample_name,
                image_name=image_name,
                bulk_sample_name=(row.get(COL_BULK) or "").strip(),
                image_folder=(row.get(COL_FOLDER) or "").strip(),
                pixel_size_um=pixel_size,
                row_index=offset,
                raw=row,
            )
        )

    if not samples:
        raise SampleSheetError(
            f"sample sheet {path} contains no data rows",
            hint="Add at least one row below the header.",
        )

    if require_image:
        non_svs = [s.sample_name for s in samples if s.image_name and not s.image_name.lower().endswith(".svs")]
        if non_svs and not any(s.pixel_size_um is not None for s in samples if s.sample_name in non_svs):
            warnings.append(
                "non-.svs images without a Pixel_size_um value: "
                + ", ".join(non_svs)
                + " (add the column, otherwise pixel size cannot be read from the file header)"
            )

    return SampleSheet(
        path=path,
        samples=samples,
        columns=[c for c in columns if c],
        delimiter=delimiter,
        kind=kind,
        warnings=warnings,
    )


def write_sample_sheet(sheet: SampleSheet, destination: os.PathLike | str) -> Path:
    """Write a sheet with only the columns that carry information.

    ``Image_name`` and ``Sample_name`` are always first, in that order, because
    ``code/extract_features_highres.py`` is given ``--sample_name_col 1``.
    Optional columns are dropped when every row is empty, and completed with a
    safe default when only some rows are empty -- see
    :meth:`SampleSheet.output_columns` and :meth:`SampleSheet.to_rows` for why
    that matters to ``code/rescale.py`` and ``code/predict.py``.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sheet.output_columns())
        writer.writeheader()
        for row in sheet.to_rows():
            writer.writerow(row)
    return destination


def sample_sheet_from_names(
    names: Sequence[str],
    images: Optional[Mapping[str, str]] = None,
    kind: str = "inference",
) -> SampleSheet:
    """Build an in-memory sheet from a plain list of sample names."""
    images = images or {}
    return SampleSheet(
        path=None,
        samples=[
            Sample(sample_name=name, image_name=images.get(name, ""), row_index=i + 1)
            for i, name in enumerate(names)
        ],
        columns=[COL_IMAGE, COL_SAMPLE],
        kind=kind,
    )


def format_sheet_problems(sheet: SampleSheet, problems: Iterable[str]) -> str:
    """Helper used by callers that accumulate several sheet problems."""
    return f"{sheet.path or '(in memory)'}:\n" + format_error_list(problems)
