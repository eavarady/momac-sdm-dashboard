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
    # Total time in ZULU time, from the first to the last timestamp of the data set
    total_time = production_log["timestamp"].max() - production_log["timestamp"].min()
    # If the total time is zero, return 0.0 to avoid division by zero
    if total_time == timedelta(0):
        return 0.0
    # Compute throughput as quantity produced per second
    return production_log["quantity"].sum() / total_time.total_seconds()


# TODO: Implement drill-down and roll-up functionality for time periods


def compute_wip(production_log: pd.DataFrame) -> int:
    # WIP is the total quantity of items currently in progress
    if production_log.empty:
        return 0
    # Count the number of items in progress
    in_progress = production_log[production_log["status"] == "in_progress"]
    # Sum the quantities of items in progress
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
