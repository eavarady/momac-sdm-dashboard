from __future__ import annotations

from io import BytesIO
from typing import Mapping, Sequence

import pandas as pd


def to_excel_bytes(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    index: bool = False,
    sheet_name: str = "Sheet1",
    na_rep: str | None = None,
    float_format: str | None = None,
    datetime_format: str = "yyyy-mm-dd hh:mm:ss",
) -> bytes:
    """Serialize a DataFrame to XLSX bytes with consistent defaults.

    Mirrors to_csv_bytes semantics: apply column selection before rename and
    return a bytes payload suitable for Streamlit's download_button.
    """
    out = df.copy()
    # Apply column selection first (so rename maps original column keys)
    if columns:
        cols = [c for c in columns if c in out.columns]
        out = out.loc[:, cols]
    if rename:
        out = out.rename(columns=dict(rename))

    buf = BytesIO()
    # Use openpyxl engine (present in requirements.txt)
    with pd.ExcelWriter(
        buf,
        engine="openpyxl",
        datetime_format=datetime_format,
        date_format=datetime_format,
    ) as writer:
        out.to_excel(
            writer,
            sheet_name=sheet_name,
            index=index,
            na_rep=na_rep,
            float_format=float_format,
        )
    return buf.getvalue()


__all__ = ["to_excel_bytes"]
