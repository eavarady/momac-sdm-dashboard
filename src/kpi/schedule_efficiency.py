import pandas as pd


def compute_schedule_efficiency(
    process_steps: pd.DataFrame, production_log: pd.DataFrame
) -> float:
    if (
        process_steps is None
        or process_steps.empty
        or production_log is None
        or production_log.empty
    ):
        return 0.0
    # Calculate total planned time, multiply by 3600 to convert hours to seconds
    total_planned_time = process_steps["estimated_time"].sum() * 3600
    if total_planned_time == 0:
        return 0.0
    # Calculate total actual time in seconds
    completed = production_log[production_log["status"] == "complete"]
    if completed.empty:
        return 0.0
    # Use completion times for actual elapsed window
    if "end_time" not in completed.columns:
        return 0.0
    t_min = completed["end_time"].min()
    t_max = completed["end_time"].max()
    total_actual_time = (t_max - t_min).total_seconds()
    if total_actual_time <= 0:
        return 0.0
    return total_planned_time / total_actual_time
