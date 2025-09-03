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
    raise NotImplementedError("on_time_rate is not implemented yet")


def compute_schedule_efficiency(
    process_steps: pd.DataFrame, production_log: pd.DataFrame
) -> float:
    if process_steps.empty or production_log.empty:
        return 0.0
    # Calculate total planned time, multiply by 3600 to convert hours to seconds
    total_planned_time = process_steps["estimated_time"].sum() * 3600
    if total_planned_time == 0:
        return 0.0
    # Calculate total actual time in seconds
    completed = production_log[production_log["status"] == "complete"]
    if completed.empty:
        return 0.0
    t_min = completed["timestamp"].min()
    t_max = completed["timestamp"].max()
    total_actual_time = (t_max - t_min).total_seconds()
    if total_actual_time <= 0:
        return 0.0
    return total_planned_time / total_actual_time


def compute_all_kpis(tables: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    kpis = {}
    kpis["throughput"] = compute_throughput(
        tables.get("production_log", pd.DataFrame())
    )
    kpis["wip"] = compute_wip(tables.get("production_log", pd.DataFrame()))
    kpis["schedule_efficiency"] = compute_schedule_efficiency(
        tables.get("process_steps", pd.DataFrame()),
        tables.get("production_log", pd.DataFrame()),
    )
    # try:
    #    kpis["on_time_rate"] = compute_on_time_rate(
    #        tables.get("process_steps", pd.DataFrame()),
    #        tables.get("production_log", pd.DataFrame()),
    #    )
    # on_time_rate is not implemented yet
    # except NotImplementedError:
    #    kpis["on_time_rate"] = np.nan
    return kpis
