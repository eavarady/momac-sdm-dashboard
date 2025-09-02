import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, timedelta


def compute_throughput(production_log: pd.DataFrame) -> float:
    # Assumes adapter normalized timestamps, quantity, and status
    if production_log.empty:
        return 0.0
    df = production_log
    produced = df[df["status"] == "complete"] if "status" in df.columns else df
    if produced.empty:
        return 0.0
    # Use completed events' span for the denominator
    t_min = produced["timestamp"].min()
    t_max = produced["timestamp"].max()
    total_time = t_max - t_min
    if total_time.total_seconds() <= 0:
        return 0.0
    return float(produced["quantity"].sum()) / total_time.total_seconds()


# TODO: Implement drill-down and roll-up functionality for time periods


def compute_wip(production_log: pd.DataFrame) -> int:
    # Assumes adapter normalized quantity/status
    if production_log.empty or "status" not in production_log.columns:
        return 0
    in_progress = production_log[production_log["status"] == "in_progress"]
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
