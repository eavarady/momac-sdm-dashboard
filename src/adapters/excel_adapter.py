from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Mapping, Optional
import pandas as pd
from schema.validate import validate_dataframe, check_uniques_and_fks

_LAST_LOAD_STATS: Dict[str, Dict[str, Any]] = {}

TABLES_ORDER = [
    "machines",
    "production_lines",
    "products",
    "operators",
    "process_steps",
    "production_log",
    "machine_metrics",
    "quality_checks",
]


def read_excel_tables(
    xlsx_path: str | Path,
    title_map: Optional[Mapping[str, str]] = None,
    skiprows: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Strict read: fast-fail on missing sheets/columns or normalization errors.
    title_map can remap expected table name -> actual sheet title.
    """
    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    _LAST_LOAD_STATS.clear()
    tables: Dict[str, pd.DataFrame] = {}

    # Discover sheets once (for friendly errors)
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
        available = set(xls.sheet_names)
    except Exception as e:
        raise RuntimeError(f"Failed to open Excel workbook: {e}")

    for name in TABLES_ORDER:
        title = (title_map or {}).get(name, name)
        if title not in available:
            _LAST_LOAD_STATS[name] = {
                "rows_read": 0,
                "rows_valid": 0,
                "rows_dropped": 0,
                "error": f"Missing worksheet '{title}' for table '{name}'",
            }
            raise ValueError(f"Missing worksheet '{title}' for table '{name}'")

        try:
            # Read as strings; let schema coerce types
            raw = pd.read_excel(
                path, sheet_name=title, engine="openpyxl", dtype=str, skiprows=skiprows
            )
            # Drop completely empty rows
            raw = raw.dropna(how="all").reset_index(drop=True)
            # If process_steps sheet omits requires_machine, default to True for backward compatibility
            if name == "process_steps" and "requires_machine" not in raw.columns:
                raw["requires_machine"] = True
            rows_read = len(raw)

            df = validate_dataframe(raw, name)
            rows_valid = len(df)
            _LAST_LOAD_STATS[name] = {
                "rows_read": rows_read,
                "rows_valid": rows_valid,
                "rows_dropped": 0,
            }
            tables[name] = df
        except Exception as e:
            _LAST_LOAD_STATS[name] = {
                "rows_read": int(raw.shape[0]) if "raw" in locals() else 0,
                "rows_valid": 0,
                "rows_dropped": int(raw.shape[0]) if "raw" in locals() else 0,
                "error": str(e),
            }
            raise

    # Cross-table validation
    fk_errs = check_uniques_and_fks(tables)
    if fk_errs:
        _LAST_LOAD_STATS["__cross_table__"] = {"errors": fk_errs}
        details = "\n".join(fk_errs)
        raise ValueError(f"Cross-table validation failed:\n{details}")

    return tables


def get_last_load_stats() -> Dict[str, Dict[str, Any]]:
    return dict(_LAST_LOAD_STATS)
