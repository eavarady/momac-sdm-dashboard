from pathlib import Path
import pandas as pd
from typing import Dict

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def read_csv_tables() -> Dict[str, pd.DataFrame]:
    tables = {}
    for name in [
        "machines",
        "production_lines",
        "products",
        "operators",
        "process_steps",
        "production_log",
        "machine_metrics",
        "quality_checks",
    ]:
        path = DATA_DIR / f"{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path)
        else:
            tables[name] = pd.DataFrame()
    return tables
