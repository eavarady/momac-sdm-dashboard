import io
import pandas as pd
import pytest

from export.csv_exporter import to_csv_bytes, safe_filename


def test_to_csv_bytes_basic(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3.14, None], "c": ["x", "y"]})
    b = to_csv_bytes(df)
    assert isinstance(b, (bytes, bytearray))
    s = b.decode("utf-8")
    # header and two rows
    assert "a,b,c" in s
    assert s.count("\n") >= 3


def test_to_csv_bytes_columns_and_rename():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    b = to_csv_bytes(df, columns=["c", "a"], rename={"a": "A"}, index=False)
    s = b.decode("utf-8")
    assert "c,A" in s
    # only one row + header
    assert s.count("\n") == 2


def test_safe_filename_edgecases():
    assert safe_filename("Top 3 Bottlenecks by WIP").endswith(".csv")
    fn = safe_filename("weird/.. name*?", ext="csv")
    assert "/" not in fn and ".." in fn or fn.count("_") >= 1
