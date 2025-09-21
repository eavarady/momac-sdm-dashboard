from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence
import re

import pandas as pd


def to_csv_bytes(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    index: bool = False,
    encoding: str = "utf-8",
    na_rep: str | None = None,
    float_format: str | None = None,
) -> bytes:
    """Serialize a DataFrame to CSV bytes with consistent defaults.

    Parameters
    ----------
    df : DataFrame
        The source data.
    columns : optional
        Column order/selection to apply before export.
    rename : optional
        Mapping of {old: new} column names applied before export.
    index : bool
        Include the DataFrame index. Defaults to False.
    encoding : str
        Output encoding. Defaults to UTF-8.
    na_rep : optional
        String representation for NaN/None.
    float_format : optional
        Format string for floating point numbers (e.g., '%.3f').

    Returns
    -------
    bytes
        The CSV content encoded to bytes.
    """
    out = df.copy()
    # Apply column selection first (so rename maps original column keys)
    if columns:
        # keep only columns that exist to avoid KeyError
        cols = [c for c in columns if c in out.columns]
        out = out.loc[:, cols]
    if rename:
        out = out.rename(columns=dict(rename))
    csv_str = out.to_csv(index=index, na_rep=na_rep, float_format=float_format)
    return csv_str.encode(encoding)


def safe_filename(name: str, ext: str = "csv") -> str:
    """Create a filesystem-friendly filename with the given extension."""
    # collapse to ascii-ish slug, keep alnum, dash, underscore, dot, space
    slug = re.sub(r"[^A-Za-z0-9\-_. ]+", "_", name).strip().strip("._ ")
    slug = re.sub(r"[\s]+", "_", slug)
    if not slug:
        slug = "export"
    return f"{slug}.{ext.lstrip('.')}"


def export_tables(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Write multiple DataFrames to CSV files in a directory.

    Existing files are overwritten. Creates the directory if needed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / safe_filename(name, "csv"), index=False)


__all__ = ["to_csv_bytes", "safe_filename", "export_tables"]
