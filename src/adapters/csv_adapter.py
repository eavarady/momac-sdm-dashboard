from pathlib import Path
import pandas as pd
from typing import Dict, Any

# In-memory stats for last load
# Note: values include ints and optional error strings under key 'error'.
_LAST_LOAD_STATS: Dict[str, Dict[str, Any]] = {}

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _normalize_production_log(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    # Required columns must exist (legacy 'timestamp' kept for now)
    required_cols = ["timestamp", "start_time", "quantity", "status"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"production_log missing required columns: {missing}")

    # Parse timestamps from ISO 8601 Z to tz-aware UTC; mark bad as NaT then fail if any
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["start_time"] = pd.to_datetime(out.get("start_time"), utc=True, errors="coerce")
    if "end_time" in out.columns:
        out["end_time"] = pd.to_datetime(out["end_time"], utc=True, errors="coerce")
    else:
        # Back-compat: if no end_time column, create it empty
        out["end_time"] = pd.NaT

    # Coerce quantity to numeric and validate > 0 for all rows
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")

    # Normalize status to lowercase for consistent comparisons
    if "status" in out.columns:
        out["status"] = (
            out["status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(
                {
                    "completed": "complete",
                    "inprogress": "in_progress",
                }
            )
        )

    # Validate fields
    bad_ts = out["timestamp"].isna()  # legacy column must parse
    bad_start = out["start_time"].isna()
    bad_qty_nan = out["quantity"].isna()
    bad_qty_nonpos = out["quantity"] <= 0
    allowed_status = {"in_progress", "complete"}
    bad_status = ~out["status"].isin(allowed_status)
    # Status-specific rules
    is_complete = out["status"] == "complete"
    is_inprog = out["status"] == "in_progress"
    # For complete: end_time required and >= start_time
    bad_end_missing_for_complete = is_complete & out["end_time"].isna()
    bad_end_before_start = is_complete & (out["end_time"] < out["start_time"])
    # For in_progress: end_time must be NaT (not filled yet)
    bad_end_present_for_inprog = is_inprog & out["end_time"].notna()

    invalid_mask = (
        bad_ts
        | bad_start
        | bad_qty_nan
        | bad_qty_nonpos
        | bad_status
        | bad_end_missing_for_complete
        | bad_end_before_start
        | bad_end_present_for_inprog
    )
    if invalid_mask.any():
        counts = {
            "bad_timestamp": int(bad_ts.sum()),
            "bad_start_time": int(bad_start.sum()),
            "bad_quantity_nan": int(bad_qty_nan.sum()),
            "bad_quantity_nonpos": int(bad_qty_nonpos.sum()),
            "bad_status": int(bad_status.sum()),
            "bad_end_missing_for_complete": int(bad_end_missing_for_complete.sum()),
            "bad_end_before_start": int(bad_end_before_start.sum()),
            "bad_end_present_for_in_progress": int(bad_end_present_for_inprog.sum()),
        }
        raise ValueError(f"production_log validation failed: {counts}")
    return out


def _normalize_process_steps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    # Basic required columns to uniquely identify a step
    required_cols = ["product_id", "step_id"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"process_steps missing required columns: {missing}")
    # Strip whitespace from id/name-like fields
    for col in [
        "product_id",
        "step_id",
        "step_name",
        "assigned_machine",
        "dependency_step_id",
    ]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    # Empty dependency -> NaN
    if "dependency_step_id" in out.columns:
        out.loc[
            out["dependency_step_id"].isin(["", "nan", "None"]), "dependency_step_id"
        ] = pd.NA
    # estimated_time in hours -> nullable Int64
    if "estimated_time" in out.columns:
        out["estimated_time"] = (
            pd.to_numeric(out["estimated_time"], errors="coerce")
            .round()
            .astype("Int64")
        )
    # assigned_operators normalize comma-separated list spacing
    if "assigned_operators" in out.columns:
        out["assigned_operators"] = (
            out["assigned_operators"]
            .fillna("")
            .astype(str)
            .apply(
                lambda s: ",".join([p for p in [x.strip() for x in s.split(",")] if p])
            )
        )
    # Validate non-empty keys
    if (
        out["product_id"].isna().any()
        or (out["product_id"].astype(str).str.strip() == "").any()
    ):
        raise ValueError("process_steps has empty product_id values")
    if (
        out["step_id"].isna().any()
        or (out["step_id"].astype(str).str.strip() == "").any()
    ):
        raise ValueError("process_steps has empty step_id values")
    return out


def _normalize_machine_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    required_cols = ["timestamp", "machine_id", "metric_type", "metric_value"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"machine_metrics missing required columns: {missing}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["metric_value"] = pd.to_numeric(out["metric_value"], errors="coerce")
    out["metric_type"] = out["metric_type"].astype(str).str.strip().str.lower()
    # Validate
    bad_ts = out["timestamp"].isna()
    bad_val = out["metric_value"].isna()
    bad_type = out["metric_type"].astype(str).str.strip() == ""
    invalid_mask = bad_ts | bad_val | bad_type
    if invalid_mask.any():
        counts = {
            "bad_timestamp": int(bad_ts.sum()),
            "bad_metric_value": int(bad_val.sum()),
            "bad_metric_type": int(bad_type.sum()),
        }
        raise ValueError(f"machine_metrics validation failed: {counts}")
    return out


def _normalize_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    required_cols = ["timestamp", "product_id", "check_type", "result", "inspector_id"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"quality_checks missing required columns: {missing}")

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    # Normalize results to {pass, fail}
    out["result"] = (
        out["result"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "passed": "pass",
                "ok": "pass",
                "fail": "fail",
                "failed": "fail",
            }
        )
    )

    bad_ts = out["timestamp"].isna()
    bad_result = ~out["result"].isin({"pass", "fail"})
    invalid_mask = bad_ts | bad_result
    if invalid_mask.any():
        counts = {
            "bad_timestamp": int(bad_ts.sum()),
            "bad_result": int(bad_result.sum()),
        }
        raise ValueError(f"quality_checks validation failed: {counts}")
    return out


def _normalize_dimension_table(
    df: pd.DataFrame, required: list[str], pk: str
) -> pd.DataFrame:
    """Strict validation for dimension tables: trim strings; fail if missing required columns,
    any empty required values, or duplicate PKs. No row drops."""
    if df.empty:
        return df
    out = df.copy()
    # Ensure required columns exist
    if not set(required).issubset(out.columns):
        missing = [c for c in required if c not in out.columns]
        raise ValueError(
            f"dimension table missing required columns {missing} (pk={pk})"
        )
    # Trim whitespace for object dtypes on required columns
    for col in required:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()
    # Validate non-null and non-empty required fields
    empties = {}
    for col in required:
        nulls = int(out[col].isna().sum())
        blanks = int((out[col].astype(str).str.strip() == "").sum())
        if nulls or blanks:
            empties[col] = {"nulls": nulls, "blanks": blanks}
    if empties:
        raise ValueError(
            f"dimension table invalid required values (pk={pk}): {empties}"
        )
    # Duplicate PKs
    dup_count = int(out.duplicated(subset=[pk], keep=False).sum())
    if dup_count:
        raise ValueError(
            f"dimension table has duplicate primary keys for {pk}: {dup_count} duplicates"
        )
    return out


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
    return tables


def get_last_load_stats() -> Dict[str, Dict[str, Any]]:
    """Return counts of rows read/kept/dropped for the most recent read_csv_tables() call.
    On validation failure, stats for the offending table include an 'error' string and rows_valid=0.
    """
    return dict(_LAST_LOAD_STATS)
