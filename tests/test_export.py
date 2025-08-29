import pandas as pd
from pathlib import Path
from src.export.csv_exporter import export_tables


def test_export(tmp_path: Path):
    tables = {"t1": pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])}
    out = tmp_path / "out"
    export_tables(tables, out)
    assert (out / "t1.csv").exists()
