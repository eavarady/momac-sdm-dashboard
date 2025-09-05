import pandas as pd
from typing import Dict
from workflow.dag import planned_finish_offsets
from utils.helpers import (
    apply_tolerance_seconds,
    tolerance_extra_seconds,
    weighted_mean,
)


def compute_throughput(production_log: pd.DataFrame) -> float:
    """Compute throughput (units/sec) from normalized production_log.

    Assumes data layer normalizes status (e.g., 'completed' -> 'complete') and timestamps (UTC).
    Uses end_time of completed items as the production completion window.
    """
    if production_log.empty:
        return 0.0
    df = production_log
    if "end_time" in df.columns and "quantity" in df.columns:
        produced = df[df["status"] == "complete"] if "status" in df.columns else df
        produced = produced[produced["end_time"].notna()]
        if not produced.empty:
            t_min = produced["end_time"].min()
            t_max = produced["end_time"].max()
            total_time = t_max - t_min
            if pd.notna(t_min) and pd.notna(t_max) and total_time.total_seconds() > 0:
                return float(produced["quantity"].sum()) / total_time.total_seconds()
    # Fallback: no valid timing window -> total quantity
    return float(df["quantity"].sum()) if "quantity" in df.columns else 0.0


# TODO: Implement drill-down and roll-up functionality for time periods


def compute_wip(production_log: pd.DataFrame) -> int:
    """Sum of quantities for in-progress rows from normalized production_log."""
    if production_log.empty or "status" not in production_log.columns:
        return 0
    in_progress = production_log[production_log["status"] == "in_progress"]
    if in_progress.empty:
        return 0
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
    # Use completion times for actual elapsed window
    if "end_time" not in completed.columns:
        return 0.0
    t_min = completed["end_time"].min()
    t_max = completed["end_time"].max()
    total_actual_time = (t_max - t_min).total_seconds()
    if total_actual_time <= 0:
        return 0.0
    return total_planned_time / total_actual_time


def _planned_finish_offsets(process_steps: pd.DataFrame) -> pd.DataFrame | None:
    # Backward-compatible wrapper; delegate to workflow.dag
    return planned_finish_offsets(process_steps)


# Measures execution efficiency of the step itself
# Ignores when the step started, upstream delays, idle/wait time
# Rule: (end_time − start_time) <= estimated_time (+ tolerance)
def _on_time_rate_duration(
    process_steps: pd.DataFrame,
    production_log: pd.DataFrame,
    tolerance_seconds: int,
    tolerance_pct: float,
    include_in_progress_as_late: bool,
) -> float:
    df = production_log.copy()
    if df.empty:
        return 0.0

    # Completed durations
    completed = df[df["status"] == "complete"] if "status" in df.columns else df
    if not completed.empty:
        if "end_time" not in completed.columns or "start_time" not in completed.columns:
            return 0.0
        completed["actual_sec"] = (
            (
                pd.to_datetime(completed["end_time"])
                - pd.to_datetime(completed["start_time"])
            )
            .dt.total_seconds()
            .clip(lower=0)
        )
    else:
        completed = pd.DataFrame(columns=df.columns.tolist() + ["actual_sec"])

    # Optional in-progress as late (use now as provisional end)
    if include_in_progress_as_late and "status" in df.columns:
        in_prog = df[df["status"] == "in_progress"].copy()
        if not in_prog.empty:
            now = pd.Timestamp.utcnow().tz_localize("UTC")
            in_prog["actual_sec"] = (
                (now - pd.to_datetime(in_prog["start_time"]))
                .dt.total_seconds()
                .clip(lower=0)
            )
    else:
        in_prog = pd.DataFrame(columns=df.columns.tolist() + ["actual_sec"])

    relevant = pd.concat([completed, in_prog], ignore_index=True)
    if relevant.empty:
        return 0.0

    steps = process_steps[["product_id", "step_id", "estimated_time"]].copy()
    steps["planned_sec"] = (
        pd.to_numeric(steps["estimated_time"], errors="coerce").fillna(0.0) * 3600.0
    )

    merged = relevant.merge(
        steps[["product_id", "step_id", "planned_sec"]],
        on=["product_id", "step_id"],
        how="left",
    )
    merged["planned_sec"] = merged["planned_sec"].fillna(0.0)
    merged = merged[merged["planned_sec"] > 0].copy()
    if merged.empty:
        return 0.0

    merged["threshold_sec"] = apply_tolerance_seconds(
        merged["planned_sec"], tolerance_seconds, tolerance_pct
    )
    merged["on_time"] = merged["actual_sec"] <= merged["threshold_sec"]

    # Quantity-weighted average
    weights = merged["quantity"] if "quantity" in merged.columns else None
    return weighted_mean(merged["on_time"], weights)


# Measure schedule adherence within a run/batch using dependency chains
# Uses Kahn's algorithm to validate DAG and compute planned finish offsets
# If run_id or dependency_step_id is missing, falls back to duration-based method
# Rule: end_time <= anchor_start + cumulative_estimates_along_path (+ tolerance).
def _on_time_rate_dependency(
    process_steps: pd.DataFrame,
    production_log: pd.DataFrame,
    tolerance_seconds: int,
    tolerance_pct: float,
    include_in_progress_as_late: bool,
) -> float:
    if "run_id" not in production_log.columns:
        return _on_time_rate_duration(
            process_steps,
            production_log,
            tolerance_seconds,
            tolerance_pct,
            include_in_progress_as_late,
        )
    offsets = _planned_finish_offsets(process_steps)
    if offsets is None:
        return _on_time_rate_duration(
            process_steps,
            production_log,
            tolerance_seconds,
            tolerance_pct,
            include_in_progress_as_late,
        )
    if production_log.empty:
        return 0.0

    df = production_log.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = (
        pd.to_datetime(df["end_time"]) if "end_time" in df.columns else pd.NaT
    )

    # Optionally treat in-progress as late using "now" as provisional end
    if include_in_progress_as_late and "status" in df.columns:
        now = pd.Timestamp.utcnow().tz_localize("UTC")
        df.loc[df["status"] == "in_progress", "end_time"] = now

    agg = df.groupby(["product_id", "run_id", "step_id"], as_index=False).agg(
        start_time=("start_time", "min"),
        end_time=("end_time", "max"),
        quantity=("quantity", "sum"),
        any_complete=(
            "status",
            lambda s: (s == "complete").any() if s.notna().any() else True,
        ),
    )
    agg = (
        agg[agg["end_time"].notna()]
        if include_in_progress_as_late
        else agg[agg["any_complete"] & agg["end_time"].notna()]
    )
    if agg.empty:
        return 0.0

    anchors = (
        agg.groupby(["product_id", "run_id"], as_index=False)["start_time"]
        .min()
        .rename(columns={"start_time": "anchor_start"})
    )

    pl = agg.merge(anchors, on=["product_id", "run_id"], how="left").merge(
        offsets, on=["product_id", "step_id"], how="left"
    )
    pl = pl[pl["planned_finish_offset_sec"].notna()].copy()
    if pl.empty:
        return 0.0

    pl["planned_finish_ts"] = pl["anchor_start"] + pd.to_timedelta(
        pl["planned_finish_offset_sec"], unit="s"
    )

    # Add only the tolerance part to the planned timestamp
    tol_sec = tolerance_extra_seconds(
        pl["planned_finish_offset_sec"], tolerance_seconds, tolerance_pct
    )
    pl["threshold_ts"] = pl["planned_finish_ts"] + pd.to_timedelta(tol_sec, unit="s")

    pl["on_time"] = pd.to_datetime(pl["end_time"]) <= pd.to_datetime(pl["threshold_ts"])

    # Quantity-weighted average
    weights = pl["quantity"] if "quantity" in pl.columns else None
    return weighted_mean(pl["on_time"], weights)


def compute_on_time_rate(
    process_steps: pd.DataFrame,
    production_log: pd.DataFrame,
    tolerance_seconds: int = 0,
    tolerance_pct: float = 0.0,
    include_in_progress_as_late: bool = False,
    mode: str = "auto",  # "auto" | "duration" | "dependency"
) -> float:
    """
    On-time = actual end_time <= inferred due time.

    - duration: compare (end - start) to estimated_time (+ tolerance)
    - dependency: infer due time = anchor_start + cumulative estimated_time along dependency chain
      (per product_id and run_id), then compare end_time to that due time (+ tolerance)
    - auto: pick dependency if both dependency_step_id and run_id exist and DAG is valid; else duration
    """
    if production_log.empty or process_steps.empty:
        return 0.0
    has_deps = "dependency_step_id" in process_steps.columns
    has_run = "run_id" in production_log.columns

    if mode == "dependency" and not has_deps:
        mode = "duration"

    if mode == "dependency" or (mode == "auto" and has_deps and has_run):
        return _on_time_rate_dependency(
            process_steps,
            production_log,
            tolerance_seconds,
            tolerance_pct,
            include_in_progress_as_late,
        )
    return _on_time_rate_duration(
        process_steps,
        production_log,
        tolerance_seconds,
        tolerance_pct,
        include_in_progress_as_late,
    )


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
    kpis["on_time_rate"] = compute_on_time_rate(
        tables.get("process_steps", pd.DataFrame()),
        tables.get("production_log", pd.DataFrame()),
    )
    return kpis
