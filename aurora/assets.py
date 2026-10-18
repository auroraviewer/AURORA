"""Reading the small data assets that ship inside a combined checkpoint.

The combined checkpoint is self-contained: the gene list, the bulk
normalisation factors and the cell-type names live in its ``assets/`` folder, so
a user never has to keep three loose CSV files next to the ``.pth``.

The readers below reproduce the exact parsing semantics of the original
pipeline (``pd.read_csv`` with and without ``header=None``) using the standard
library only, so they work before NumPy/pandas are imported:

* ``LUAD_gene.csv`` has a header and the genes live in the ``Genes`` column.
* ``LUAD_cell_types.csv`` has **no** header -- one cell type per line.
* ``LUAD_gene_normalize_factor.csv`` has a header, the first column is the gene
  name and the column named ``0`` holds the factor.

Nothing here imports a heavy third-party package.
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .errors import ConfigError

BytesLike = Union[bytes, bytearray, str, os.PathLike]


def _read_text(source: BytesLike) -> str:
    """Accept raw bytes, a path string, a ``Path`` or already-decoded text."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source).decode("utf-8-sig")
    if isinstance(source, Path) or (isinstance(source, str) and ("\n" not in source and "\r" not in source)):
        path = Path(source)
        if path.is_file():
            return path.read_text(encoding="utf-8-sig")
    if isinstance(source, str):
        return source
    raise ConfigError(f"cannot read text asset from {type(source).__name__}")


def _rows(text: str, delimiter: Optional[str] = None) -> List[List[str]]:
    if delimiter is None:
        first = text.splitlines()[0] if text.splitlines() else ""
        delimiter = "\t" if first.count("\t") > first.count(",") else ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    return [row for row in reader if row and any(cell.strip() for cell in row)]


HEADER_TOKENS = {"genes", "gene", "gene_name", "genename", "cell_type", "celltype", "cell_types"}


def read_gene_list(source: BytesLike, column: Optional[str] = None) -> List[str]:
    """Read the ordered gene list.

    A header row is detected by looking for a known token; the genes are taken
    from ``column`` when given, otherwise from the ``Genes`` column when the
    header has one, otherwise from the first column.
    """
    rows = _rows(_read_text(source))
    if not rows:
        raise ConfigError("the gene list asset is empty")
    header = [cell.strip() for cell in rows[0]]
    has_header = any(cell.lower() in HEADER_TOKENS for cell in header)
    if has_header:
        if column is not None:
            try:
                index = header.index(column)
            except ValueError as exc:
                raise ConfigError(
                    f"gene list has no column {column!r} (columns: {', '.join(header)})"
                ) from exc
        else:
            index = next(
                (i for i, cell in enumerate(header) if cell.lower() in ("genes", "gene", "gene_name")),
                0,
            )
            if header[index].strip() == "" and len(header) > 1:
                index = 1
        body = rows[1:]
    else:
        index = 0
        body = rows
    genes = [row[index].strip() for row in body if len(row) > index and row[index].strip()]
    if not genes:
        raise ConfigError("the gene list asset contains no gene names")
    duplicates = len(genes) - len(set(genes))
    if duplicates:
        # The original code tolerated duplicates; keep the order but say so.
        seen = set()
        unique: List[str] = []
        for gene in genes:
            if gene not in seen:
                seen.add(gene)
                unique.append(gene)
        genes = unique
    return genes


def read_cell_types(source: BytesLike, column: Optional[str] = None) -> List[str]:
    """Read the ordered cell-type names.

    The shipped ``*_cell_types.csv`` files have no header, so a header is only
    assumed when the first row is not a plausible cell-type name (i.e. it looks
    like a column title).
    """
    rows = _rows(_read_text(source))
    if not rows:
        raise ConfigError("the cell-type asset is empty")
    header = [cell.strip() for cell in rows[0]]
    looks_like_header = len(header) == 1 and header[0].lower() in HEADER_TOKENS
    if column is not None or looks_like_header:
        if looks_like_header:
            body = rows[1:]
            index = 0
            if column is not None:
                try:
                    index = header.index(column)
                except ValueError:
                    index = 0
        else:
            index = 0
            body = rows
    else:
        index = 0
        body = rows
    cell_types = [row[index].strip() for row in body if len(row) > index and row[index].strip()]
    if not cell_types:
        raise ConfigError("the cell-type asset contains no names")
    return cell_types


def read_norm_factors(source: BytesLike, genes: Sequence[str]) -> List[float]:
    """Read per-gene bulk normalisation factors, aligned to ``genes``.

    Mirrors ``pd.read_csv(..., index_col=0).loc[gene_list, '0'].values``: the
    first column is the index, the column named ``0`` holds the values, and a
    gene missing from the table raises a KeyError-equivalent error.

    Only the requested genes are parsed.  The shipped table contains a couple of
    rows with an empty value (``DAZ2``, ``HNRNPCL3``) which the original code
    silently turned into ``NaN`` -- they are not part of any gene list, and
    parsing the whole column eagerly would reject a perfectly good table.
    """
    rows = _rows(_read_text(source))
    if not rows:
        raise ConfigError("the normalisation-factor asset is empty")
    header = [cell.strip() for cell in rows[0]]
    if "0" in header:
        value_index = header.index("0")
    elif len(header) >= 2:
        value_index = 1
    else:
        raise ConfigError(
            "the normalisation-factor asset needs at least two columns "
            f"(gene name and value); found {header}"
        )
    table: Dict[str, str] = {}
    for row in rows[1:]:
        if len(row) <= value_index:
            continue
        key = row[0].strip()
        if key and key not in table:
            table[key] = row[value_index].strip()

    missing = [gene for gene in genes if gene not in table]
    if missing:
        shown = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise ConfigError(
            f"the normalisation-factor asset has no entry for {len(missing)} gene(s): {shown}{more}",
            hint=(
                "The factor table must cover every gene of the gene list. This is the same "
                "check the original code performed with DataFrame.loc[gene_list, '0']."
            ),
        )
    values: List[float] = []
    for gene in genes:
        raw = table[gene]
        try:
            values.append(float(raw))
        except ValueError as exc:
            raise ConfigError(
                f"the normalisation factor for gene {gene!r} is not a number: {raw!r}",
                hint=(
                    "The original pipeline produced NaN for such rows and fed it to the model, "
                    "which poisons the prediction. Fill in a value or remove the gene from the "
                    "gene list."
                ),
            ) from exc
    return values


def read_params(source: BytesLike) -> Dict[str, object]:
    """Read a ``model_parameter_*.json`` asset."""
    import json

    text = _read_text(source)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"could not parse the model parameter JSON asset: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("the model parameter asset must be a JSON object")
    return data
