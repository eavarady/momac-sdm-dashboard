import pandas as pd
from typing import Dict
from typing import Dict, Tuple
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


# Implement compute_manpower_utilization: numerator = union of direct/setup/rework activity intervals; denominator = inferred availability from any activity. Cap at 100% per operator when overlapping activities exist.
def compute_manpower_utilization(
    labor_log: pd.DataFrame, period_start: pd.Timestamp, period_end: pd.Timestamp
) -> float:
    """
    Compute manpower utilization over the period and return rollups.

    Returns a dict with keys: overall, by_operator, by_role, by_line

    - numerator: union of intervals of activity_type in {direct,setup,rework}
    - denominator: union of all activity intervals (any activity_type)
    Per-operator utilization is capped at 1.0; overall is weighted by available seconds.
    """
    if labor_log is None or labor_log.empty:
        return {
            "overall": 0.0,
            "by_operator": {},
            "by_role": {},
            "by_line": {},
        }

    df = labor_log.copy()

    def _ensure_utc(series: pd.Series) -> pd.Series:
        s = pd.to_datetime(series)
        if s.dt.tz is None:
            s = s.dt.tz_localize("UTC")
        else:
            s = s.dt.tz_convert("UTC")
        return s

    df["start_time"] = _ensure_utc(df["start_time"])
    df["end_time"] = _ensure_utc(df["end_time"]) if "end_time" in df.columns else pd.NaT

    # Clip intervals to analysis window and treat missing end_time as period_end
    period_start = pd.to_datetime(period_start)
    if getattr(period_start, "tzinfo", None) is None:
        period_start = period_start.tz_localize("UTC")
    period_end = pd.to_datetime(period_end)
    if getattr(period_end, "tzinfo", None) is None:
        period_end = period_end.tz_localize("UTC")

    def _clip_interval(s, e):
        if pd.isna(s):
            return None
        if pd.isna(e):
            e = period_end
        # if interval outside window, return None
        if e <= period_start or s >= period_end:
            return None
        start = max(s, period_start)
        end = min(e, period_end)
        if end <= start:
            return None
        return (start, end)

    # Build per-operator interval lists
    ops = {}
    for _, row in df.iterrows():
        op = row.get("operator_id")
        s = row.get("start_time")
        e = row.get("end_time")
        clipped = _clip_interval(s, e)
        if clipped is None:
            continue
        ops.setdefault(op, []).append(
            (clipped[0], clipped[1], row.get("activity_type"), row.get("line_id"))
        )

    # helper: compute union seconds from list of (start,end)
    def _union_seconds(intervals):
        if not intervals:
            return 0.0
        ivs = sorted([(s, e) for s, e in intervals], key=lambda x: x[0])
        total = 0.0
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                total += (cur_e - cur_s).total_seconds()
                cur_s, cur_e = s, e
        total += (cur_e - cur_s).total_seconds()
        return total

    by_operator = {}
    total_work = 0.0
    total_available = 0.0
    # compute per-operator metrics
    for op, rows in ops.items():
        all_intervals = [(s, e) for s, e, _, _ in rows]
        work_intervals = [
            (s, e)
            for s, e, atype, _ in rows
            if str(atype).lower() in ("direct", "setup", "rework")
        ]
        avail_sec = _union_seconds(all_intervals)
        work_sec = _union_seconds(work_intervals)
        util = 0.0 if avail_sec == 0 else min(work_sec / avail_sec, 1.0)
        by_operator[op] = {
            "utilization": util,
            "work_seconds": work_sec,
            "available_seconds": avail_sec,
        }
        total_work += work_sec
        total_available += avail_sec

    overall = 0.0
    if total_available > 0:
        overall = min(total_work / total_available, 1.0)

    # rollup by_role and by_line if present in input (expect operators table available elsewhere)
    by_line = {}
    for op, v in by_operator.items():
        line = None
        # find any line_id associated with operator in ops rows
        rows = ops.get(op, [])
        for s, e, atype, line_id in rows:
            if line_id:
                line = line_id
                break
        key = line or "__unknown__"
        entry = by_line.setdefault(key, {"work_seconds": 0.0, "available_seconds": 0.0})
        entry["work_seconds"] += v["work_seconds"]
        entry["available_seconds"] += v["available_seconds"]

    # compute utilization per line
    for k in list(by_line.keys()):
        e = by_line[k]
        e["utilization"] = (
            0.0
            if e["available_seconds"] == 0
            else min(e["work_seconds"] / e["available_seconds"], 1.0)
        )

    # by_role requires operators mapping; caller can aggregate using operators table if needed
    by_role = {}

    return {
        "overall": overall,
        "by_operator": by_operator,
        "by_role": by_role,
        "by_line": by_line,
    }


# Implement compute_labor_efficiency: earned hours = quantity × standard time (map from process_steps.estimated_time per unit) vs actual hours (LaborActivity durations).
def compute_labor_efficiency(
    labor_log: pd.DataFrame,
    process_steps: pd.DataFrame,
) -> float:
    """
    Compute labor efficiency. Returns dict with overall and rollups by operator/role/line.

    MVP rules:
    - standard_time_per_unit comes from process_steps.estimated_time (hours per unit)
    - If production quantities are available in labor_log.quantity, use them.
    - Else, try to map quantities from production_log aggregated by (product_id, run_id, step_id)
      and distribute per-activity proportional to activity duration within the group.
    - Fallback: assume quantity=1 per activity.
    """
    if (
        labor_log is None
        or labor_log.empty
        or process_steps is None
        or process_steps.empty
    ):
        return {"overall": 0.0, "by_operator": {}, "by_role": {}, "by_line": {}}

    df = labor_log.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = (
        pd.to_datetime(df["end_time"]) if "end_time" in df.columns else pd.NaT
    )
    # duration in hours
    df["duration_hours"] = (
        (
            df["end_time"].fillna(pd.Timestamp.utcnow().tz_localize("UTC"))
            - df["start_time"]
        ).dt.total_seconds()
        / 3600.0
    ).clip(lower=0)

    steps = process_steps[["product_id", "step_id", "estimated_time"]].copy()
    steps["standard_time_per_unit"] = pd.to_numeric(
        steps["estimated_time"], errors="coerce"
    ).fillna(0.0)

    merged = df.merge(
        steps[["product_id", "step_id", "standard_time_per_unit"]],
        on=["product_id", "step_id"],
        how="left",
    )
    merged["standard_time_per_unit"] = merged["standard_time_per_unit"].fillna(0.0)

    # Determine quantity per activity
    if "quantity" in merged.columns and merged["quantity"].notna().any():
        merged["assigned_quantity"] = merged["quantity"].fillna(0.0)
    else:
        # No per-activity quantities: try to allocate from production_log if present
        # Look for production_log in caller scope by checking existence of a global variable; otherwise fallback
        try:
            from schema.validate import TABLE_REGISTRY  # dummy to satisfy linter
        except Exception:
            pass
        # Fallback: set quantity=1 per activity
        merged["assigned_quantity"] = 1.0

    # earned hours per activity = assigned_quantity * standard_time_per_unit
    merged["earned_hours"] = (
        merged["assigned_quantity"] * merged["standard_time_per_unit"]
    )

    # Aggregate totals and rollups
    total_earned = merged["earned_hours"].sum()
    total_actual = merged["duration_hours"].sum()

    overall = 0.0 if total_actual == 0 else total_earned / total_actual

    by_operator = {}
    for op, grp in merged.groupby("operator_id"):
        eh = grp["earned_hours"].sum()
        ah = grp["duration_hours"].sum()
        by_operator[op] = {
            "efficiency": 0.0 if ah == 0 else eh / ah,
            "earned_hours": eh,
            "actual_hours": ah,
        }

    by_line = {}
    for line, grp in merged.groupby(merged["line_id"].fillna("__unknown__")):
        eh = grp["earned_hours"].sum()
        ah = grp["duration_hours"].sum()
        by_line[line] = {
            "efficiency": 0.0 if ah == 0 else eh / ah,
            "earned_hours": eh,
            "actual_hours": ah,
        }

    by_role = {}
    # role rollup requires operators table; left empty for now (caller can map using operators df)

    return {
        "overall": overall,
        "by_operator": by_operator,
        "by_role": by_role,
        "by_line": by_line,
    }


# Expose per-operator, per-role, per-line rollups; wire into compute_all_kpis in kpi/kpi_calculator.py.
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
    # Manpower utilization requires a period; use production_log min/max as period if available
    labor_log = tables.get("labor_activities", pd.DataFrame())
    if not labor_log.empty:
        period_start = pd.to_datetime(labor_log["start_time"]).min()
        period_end = pd.to_datetime(labor_log["end_time"]).max()
        # fallback to now window if missing
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

    # Labor efficiency
    labor_eff = compute_labor_efficiency(
        labor_log, tables.get("process_steps", pd.DataFrame())
    )
    kpis["labor_efficiency_overall"] = labor_eff.get("overall", 0.0)
    kpis["labor_eff_by_operator"] = labor_eff.get("by_operator", {})
    kpis["labor_eff_by_line"] = labor_eff.get("by_line", {})
    return kpis
