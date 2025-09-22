from io import BytesIO

import pandas as pd
from openpyxl import load_workbook

from export.excel_exporter import to_excel_bytes


def _read_xlsx(bytes_data: bytes) -> pd.DataFrame:
    wb = load_workbook(filename=BytesIO(bytes_data))
    ws = wb.active
    data = ws.values
    rows = list(data)
    headers = rows[0]
    values = rows[1:]
    return pd.DataFrame(values, columns=headers)


def test_to_excel_bytes_basic_roundtrip():
    df = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
    b = to_excel_bytes(df, index=False, sheet_name="Test")
    out = _read_xlsx(b)
    assert list(out.columns) == ["a", "b"]
    assert out.shape == (2, 2)
    assert float(out.loc[0, "b"]) == 3.5


def test_to_excel_bytes_columns_and_rename():
    df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
    b = to_excel_bytes(
        df,
        columns=["z", "x"],
        rename={"z": "Zed", "x": "Ex"},
        index=False,
        sheet_name="Renamed",
    )
    out = _read_xlsx(b)
    assert list(out.columns) == ["Zed", "Ex"]
    assert out.iloc[0].tolist() == [3, 1]
