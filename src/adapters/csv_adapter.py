from pathlib import Path
import pandas as pd
from typing import Dict, Any
from schema.validate import validate_dataframe, check_uniques_and_fks

# In-memory stats for last load
# Note: values include ints and optional error strings under key 'error'.
_LAST_LOAD_STATS: Dict[str, Dict[str, Any]] = {}

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _normalize_production_log(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, "production_log")


def _normalize_process_steps(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, "process_steps")

def _normalize_machine_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, "machine_metrics")


def _normalize_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, "quality_checks")


def _normalize_dimension_table(
    df: pd.DataFrame, required: list[str], pk: str
) -> pd.DataFrame:
    # Map pk to table name so we validate with the right model
    table_name = {
        "machine_id": "machines",
        "line_id": "production_lines",
        "product_id": "products",
        "operator_id": "operators",
    }.get(pk)
    if table_name is None:
        raise ValueError(f"Unknown dimension pk={pk}; cannot select schema")

    norm = validate_dataframe(df, table_name)

    # Ensure required columns exist and order them for downstream code
    missing = [c for c in required if c not in norm.columns]
    for c in missing:
        norm[c] = ""  # fill any missing optional columns
    return norm[required].copy()


def read_csv_tables() -> Dict[str, pd.DataFrame]:
    tables = {}
    _LAST_LOAD_STATS.clear()
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
            raw = pd.read_csv(path)
            rows_read = len(raw)
            try:
                df = raw
                if name == "production_log":
                    df = _normalize_production_log(df)
                elif name == "process_steps":
                    df = _normalize_process_steps(df)
                elif name == "machine_metrics":
                    df = _normalize_machine_metrics(df)
                elif name == "quality_checks":
                    df = _normalize_quality_checks(df)
                elif name == "machines":
                    df = _normalize_dimension_table(
                        df, ["machine_id", "line_id", "type", "status"], pk="machine_id"
                    )
                elif name == "production_lines":
                    df = _normalize_dimension_table(
                        df, ["line_id", "name", "shift"], pk="line_id"
                    )
                elif name == "products":
                    df = _normalize_dimension_table(
                        df,
                        ["product_id", "name", "category", "spec_version"],
                        pk="product_id",
                    )
                elif name == "operators":
                    df = _normalize_dimension_table(
                        df, ["operator_id", "name", "role"], pk="operator_id"
                    )
                rows_valid = len(df)
                _LAST_LOAD_STATS[name] = {
                    "rows_read": rows_read,
                    "rows_valid": rows_valid,
                    "rows_dropped": 0,
                }
                tables[name] = df
            except Exception as e:
                # Record failure stats and re-raise to enforce fast-fail at the table level
                _LAST_LOAD_STATS[name] = {
                    "rows_read": rows_read,
                    "rows_valid": 0,
                    "rows_dropped": rows_read,  # marker of invalid rows since we don't keep any
                    "error": str(e),
                }
                raise
        else:
            _LAST_LOAD_STATS[name] = {
                "rows_read": 0,
                "rows_valid": 0,
                "rows_dropped": 0,
            }
            tables[name] = pd.DataFrame()

    # Cross-table validation (unique keys, foreign keys, workflow deps)
    fk_errs = check_uniques_and_fks(tables)
    if fk_errs:
        # record for UI troubleshooting (non-breaking to your existing stats shape)
        _LAST_LOAD_STATS["__cross_table__"] = {"errors": fk_errs}
        details = "\n".join(fk_errs)
        raise ValueError(f"Cross-table validation failed:\n{details}")
    return tables


def get_last_load_stats() -> Dict[str, Dict[str, Any]]:
    """Return counts of rows read/kept/dropped for the most recent read_csv_tables() call.
    On validation failure, stats for the offending table include an 'error' string and rows_valid=0.
    """
    return dict(_LAST_LOAD_STATS)
