import pandas as pd


def compute_labor_efficiency(
    labor_log: pd.DataFrame, process_steps: pd.DataFrame
) -> dict:
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
    now = pd.Timestamp.utcnow()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    df["duration_hours"] = (
        (df["end_time"].fillna(now) - df["start_time"]).dt.total_seconds() / 3600.0
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

    if "quantity" in merged.columns and merged["quantity"].notna().any():
        merged["assigned_quantity"] = merged["quantity"].fillna(0.0)
    else:
        merged["assigned_quantity"] = 1.0

    merged["earned_hours"] = (
        merged["assigned_quantity"] * merged["standard_time_per_unit"]
    )

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

    return {
        "overall": overall,
        "by_operator": by_operator,
        "by_role": by_role,
        "by_line": by_line,
    }
