import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, timedelta


def compute_throughput(production_log: pd.DataFrame) -> float:
    # Throughput is the total quantity produced divided by the total time taken
    # TODO: Implement timestamp conversion to local time
    # production_log["timestamp"] = production_log["timestamp"].dt.tz_convert("local_tz")
    # If there are no valid timestamps, return 0.0
    if production_log.empty:
        return 0.0
    df = production_log.copy()

    # Parse timestamps (strings like "2025-06-29T12:34:38Z" → UTC datetimes)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Clean quantities
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["timestamp", "quantity"])
    df = df[df["quantity"] > 0]
    if df.empty:
        return 0.0

    # Use only completed items for throughput (sensible definition)
    produced = df[df["status"].astype(str).str.lower() == "complete"] if "status" in df.columns else df
    if produced.empty:
        return 0.0

    # Time window across the (cleaned) dataset
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    total_time = t_max - t_min
    if total_time.total_seconds() <= 0:
        return 0.0

    return float(produced["quantity"].sum()) / total_time.total_seconds()




# TODO: Implement drill-down and roll-up functionality for time periods


def compute_wip(production_log: pd.DataFrame) -> int:
    # WIP is the total quantity of items currently in progress
    if production_log.empty:
        return 0
    df = production_log.copy()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["quantity"])
    df = df[df["quantity"] > 0]
    if df.empty or "status" not in df.columns:
        return 0
    df["status"] = df["status"].astype(str).str.lower()
    in_progress = df[df["status"] == "in_progress"]
    return int(in_progress["quantity"].sum())


# TODO: Implement on-time rate calculation
def compute_on_time_rate(
    process_steps: pd.DataFrame, production_log: pd.DataFrame
) -> float:
    """Placeholder. Implement on-time rate later."""
    raise NotImplementedError("compute_on_time_rate is not implemented yet")


def compute_all_kpis(tables: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    kpis = {}
    kpis["throughput"] = compute_throughput(
        tables.get("production_log", pd.DataFrame())
    )
    kpis["wip"] = compute_wip(tables.get("production_log", pd.DataFrame()))
    try:
        kpis["on_time_rate"] = compute_on_time_rate(
            tables.get("process_steps", pd.DataFrame()),
            tables.get("production_log", pd.DataFrame()),
        )
    # on_time_rate is not implemented yet
    except NotImplementedError:
        kpis["on_time_rate"] = np.nan
    return kpis
