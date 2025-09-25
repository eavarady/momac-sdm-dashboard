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
    """Compute schedule efficiency as (planned_seconds) / (actual_seconds).

    This implementation merges the planned estimate per (product_id, step_id)
    into the completed rows of the production log and computes quantity-weighted
    totals. The previous implementation used a global end_time window which
    produced a very large denominator across historical data and yielded tiny
    ratios; this per-row approach keeps planned/actual paired and comparable.

    Returns 0.0 when there is insufficient data.
    """
    # use only completed rows (or all rows if status is missing)
    completed = (
        production_log[production_log["status"] == "complete"]
        if "status" in production_log.columns
        else production_log
    )
    if completed.empty:
        return 0.0

    # Prepare planned seconds per (product_id, step_id)
    steps = process_steps[["product_id", "step_id", "estimated_time"]].copy()
    steps["planned_sec"] = (
        pd.to_numeric(steps["estimated_time"], errors="coerce").fillna(0.0) * 3600.0
    )

    merged = completed.merge(
        steps[["product_id", "step_id", "planned_sec"]],
        on=["product_id", "step_id"],
        how="left",
    )
    merged["planned_sec"] = merged["planned_sec"].fillna(0.0)
    # remove rows without a planned estimate
    merged = merged[merged["planned_sec"] > 0].copy()
    if merged.empty:
        return 0.0

    # compute actual duration per row where possible
    if not {"start_time", "end_time"}.issubset(merged.columns):
        return 0.0
    merged["start_time"] = pd.to_datetime(
        merged["start_time"], utc=True, errors="coerce"
    )
    merged["end_time"] = pd.to_datetime(merged["end_time"], utc=True, errors="coerce")
    merged = merged.dropna(
        subset=["start_time", "end_time"]
    )  # require both for a duration
    if merged.empty:
        return 0.0

    merged["actual_sec"] = (
        (merged["end_time"] - merged["start_time"]).dt.total_seconds().clip(lower=0)
    )

    # weight by quantity when available
    if "quantity" in merged.columns and merged["quantity"].notna().any():
        merged["qty"] = merged["quantity"].fillna(0).astype(float)
    else:
        merged["qty"] = 1.0

    planned_total = (merged["planned_sec"] * merged["qty"]).sum()
    actual_total = (merged["actual_sec"] * merged["qty"]).sum()
    if actual_total <= 0:
        return 0.0
    return float(planned_total / actual_total)
