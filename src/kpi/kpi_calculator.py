import pandas as pd
from typing import Dict

# Controller: import KPI implementations from small modules
from .throughput import compute_throughput
from .wip import compute_wip
from .schedule_efficiency import compute_schedule_efficiency
from .on_time import compute_on_time_rate
from .manpower import compute_manpower_utilization
from .labor_efficiency import compute_labor_efficiency
from .cycle_time import (
    compute_avg_cycle_time_mean,
    compute_avg_cycle_time_median,
)


def compute_all_kpis(tables: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Orchestrator that computes and aggregates KPI values from provided tables.

    This file is intentionally small — individual KPI implementations live in
    dedicated modules under `src/kpi/` so they are easier to test and maintain.
    """
    kpis = {}
    kpis["throughput"] = compute_throughput(
        tables.get("production_log", pd.DataFrame())
    )
    kpis["wip"] = compute_wip(tables.get("production_log", pd.DataFrame()))
    kpis["schedule_efficiency"] = compute_schedule_efficiency(
        tables.get("process_steps", pd.DataFrame()),
        tables.get("production_log", pd.DataFrame()),
    )
    kpis["on_time_rate"] = compute_on_time_rate(
        tables.get("process_steps", pd.DataFrame()),
        tables.get("production_log", pd.DataFrame()),
    )

    labor_log = tables.get("labor_activities", pd.DataFrame())
    if not (labor_log is None or labor_log.empty):
        period_start = pd.to_datetime(labor_log["start_time"]).min()
        period_end = pd.to_datetime(labor_log.get("end_time")).max()
        if pd.isna(period_start) or pd.isna(period_end):
            period_end = pd.Timestamp.utcnow().tz_localize("UTC")
            period_start = period_end - pd.Timedelta(days=1)
        manpower = compute_manpower_utilization(labor_log, period_start, period_end)
        kpis["manpower_utilization_overall"] = manpower.get("overall", 0.0)
        kpis["manpower_by_operator"] = manpower.get("by_operator", {})
        kpis["manpower_by_line"] = manpower.get("by_line", {})
    else:
        kpis["manpower_utilization_overall"] = 0.0
        kpis["manpower_by_operator"] = {}
        kpis["manpower_by_line"] = {}

    labor_eff = compute_labor_efficiency(
        labor_log, tables.get("process_steps", pd.DataFrame())
    )
    kpis["labor_efficiency_overall"] = labor_eff.get("overall", 0.0)
    kpis["labor_eff_by_operator"] = labor_eff.get("by_operator", {})
    kpis["labor_eff_by_line"] = labor_eff.get("by_line", {})

    kpis["avg_cycle_time_mean"] = compute_avg_cycle_time_mean(
        tables.get("production_log", pd.DataFrame())
    )
    kpis["avg_cycle_time_median"] = compute_avg_cycle_time_median(
        tables.get("production_log", pd.DataFrame())
    )
    kpis["avg_cycle_time"] = kpis["avg_cycle_time_mean"]

    return kpis


def compute_avg_cycle_time_mean(production_log: pd.DataFrame) -> float:
    """Average cycle time (mean) in hours for completed events."""
    if production_log is None or production_log.empty:
        return 0.0
    df = production_log.copy()
    if not {"start_time", "end_time"}.issubset(df.columns):
        return 0.0
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "complete"]
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    if df.empty:
        return 0.0
    avg_seconds = (df["end_time"] - df["start_time"]).dt.total_seconds().mean()
    return float(avg_seconds / 3600.0)


def compute_avg_cycle_time_median(production_log: pd.DataFrame) -> float:
    """Typical cycle time (median) in hours for completed events."""
    if production_log is None or production_log.empty:
        return 0.0
    df = production_log.copy()
    if not {"start_time", "end_time"}.issubset(df.columns):
        return 0.0
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "complete"]
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    if df.empty:
        return 0.0
    med_seconds = (df["end_time"] - df["start_time"]).dt.total_seconds().median()
    return float(med_seconds / 3600.0)
