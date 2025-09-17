from __future__ import annotations
import pandas as pd


def _normalize_status_col(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        return df
    out = df.copy()
    out["status"] = (
        out["status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"completed": "complete"})
    )
    return out


def per_step_progress(
    process_steps: pd.DataFrame,
    production_log: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-step progress as quantity-complete / (quantity-complete + quantity-in_progress).

    Returns DataFrame with:
      product_id, step_id[, step_name], complete_qty, in_progress_qty, progress (0..1)
    """
    cols = ["product_id", "step_id"] + (
        ["step_name"] if "step_name" in process_steps.columns else []
    )
    base = process_steps.loc[:, cols].drop_duplicates().copy()
    if base.empty:
        return pd.DataFrame(
            columns=cols + ["complete_qty", "in_progress_qty", "progress"]
        )

    log = production_log.copy()
    if not log.empty:
        log = _normalize_status_col(log)
        if "quantity" in log.columns:
            log["quantity"] = pd.to_numeric(log["quantity"], errors="coerce").fillna(0)
        else:
            log["quantity"] = 0

        agg = (
            log.groupby(["product_id", "step_id", "status"], dropna=False)["quantity"]
            .sum()
            .reset_index()
        )
        pivot = agg.pivot_table(
            index=["product_id", "step_id"],
            columns="status",
            values="quantity",
            fill_value=0,
            aggfunc="sum",
        ).reset_index()
        for col in ("complete", "in_progress"):
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot.rename(
            columns={"complete": "complete_qty", "in_progress": "in_progress_qty"}
        )
    else:
        pivot = base[["product_id", "step_id"]].copy()
        pivot["complete_qty"] = 0
        pivot["in_progress_qty"] = 0

    merged = base.merge(pivot, on=["product_id", "step_id"], how="left")
    merged["complete_qty"] = merged["complete_qty"].fillna(0)
    merged["in_progress_qty"] = merged["in_progress_qty"].fillna(0)

    denom = merged["complete_qty"] + merged["in_progress_qty"]
    merged["progress"] = 0.0
    nonzero = denom > 0
    merged.loc[nonzero, "progress"] = (
        merged.loc[nonzero, "complete_qty"] / denom.loc[nonzero]
    ).clip(0, 1)

    ordered_cols = ["product_id", "step_id"]
    if "step_name" in merged.columns:
        ordered_cols.append("step_name")
    cols_tail = ["complete_qty", "in_progress_qty"]
    cols_tail.append("progress")
    ordered_cols += cols_tail
    return merged.loc[:, [c for c in ordered_cols if c in merged.columns]]


def overall_progress_by_product(step_progress: pd.DataFrame) -> pd.DataFrame:
    """
    Average per-step progress across all steps of each product.
    Contract: Always returns DataFrame with columns [product_id, overall_progress].
    - Empty input -> empty DataFrame with those columns.
    - Missing 'progress' column -> raises ValueError (fast fail).
    - Progress values clipped into [0.0, 1.0].
    """
    required_cols = {"product_id", "progress"}
    if not required_cols.issubset(step_progress.columns):
        raise ValueError(
            f"overall_progress_by_product: missing required columns {required_cols - set(step_progress.columns)}"
        )
    if step_progress.empty:
        return pd.DataFrame(columns=["product_id", "overall_progress"])
    grp = step_progress.groupby("product_id", as_index=False)["progress"].mean()
    grp["progress"] = pd.to_numeric(grp["progress"], errors="coerce").fillna(0.0)
    grp["progress"] = grp["progress"].clip(lower=0.0, upper=1.0)
    grp = grp.rename(columns={"progress": "overall_progress"})
    return grp[["product_id", "overall_progress"]]


def per_run_progress(
    process_steps: pd.DataFrame,
    production_log: pd.DataFrame,
    runs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Mode-agnostic per-run KPIs derived from event grain.
    Returns one row per run with:
      - progress_qty: qty_out(run)/planned_qty using terminal steps (clip 0..1)
      - progress_steps: steps_completed/total_steps
      - progress: alias of progress_qty if planned_qty>0 else progress_steps
      - planned_qty (from runs/targets) and optional execution_mode
    """
    if production_log.empty or process_steps.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "run_id",
                "planned_qty",
                "execution_mode",
                "progress_qty",
                "progress_steps",
                "progress",
            ]
        )

    log = _normalize_status_col(production_log.copy())
    log["quantity"] = pd.to_numeric(log.get("quantity", 0), errors="coerce").fillna(0)

    # Determine terminal steps per product (no dependents)
    steps = process_steps[["product_id", "step_id", "dependency_step_id"]].copy()
    dependents = steps.dropna(subset=["dependency_step_id"])[
        "dependency_step_id"
    ].astype(str)
    steps["is_terminal"] = ~steps["step_id"].astype(str).isin(dependents.astype(str))

    # Completed quantity at terminal steps per run
    term = steps.loc[steps["is_terminal"], ["product_id", "step_id"]]
    log_c = log[log["status"] == "complete"]
    qty_out = (
        log_c.merge(term, on=["product_id", "step_id"], how="inner")
        .groupby(["product_id", "run_id"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "qty_out"})
    )

    # Steps completion progress per run: cross with all steps to include unstarted
    all_steps = process_steps[["product_id", "step_id"]].drop_duplicates()
    step_done = (
        log[log["status"] == "complete"]
        .groupby(["product_id", "run_id", "step_id"], as_index=False)["quantity"]
        .sum()
    )
    step_done["done"] = (step_done["quantity"] > 0).astype(int)
    per_run_step = (
        all_steps.merge(
            step_done[["product_id", "run_id", "step_id", "done"]],
            on=["product_id", "step_id"],
            how="left",
        )
        .assign(done=lambda d: d["done"].fillna(0))
        .groupby(["product_id", "run_id"], as_index=False)["done"]
        .mean()
        .rename(columns={"done": "progress_steps"})
    )

    # Planned quantity
    planned = None
    if (
        runs is not None
        and not runs.empty
        and {"run_id", "planned_qty"}.issubset(runs.columns)
    ):
        planned = runs[["run_id", "planned_qty", "execution_mode"]].copy()

    base = per_run_step.merge(qty_out, on=["product_id", "run_id"], how="left")
    base["qty_out"] = pd.to_numeric(base.get("qty_out", 0), errors="coerce").fillna(0)

    if planned is not None:
        base = base.merge(planned, on="run_id", how="left")
        base["planned_qty"] = pd.to_numeric(
            base.get("planned_qty", 0), errors="coerce"
        ).fillna(0)
        has_plan = base["planned_qty"] > 0
        base["progress_qty"] = 0.0
        base.loc[has_plan, "progress_qty"] = (
            base.loc[has_plan, "qty_out"] / base.loc[has_plan, "planned_qty"]
        ).clip(0, 1)
        base["progress"] = (
            base["progress_qty"].fillna(base["progress_steps"]).clip(0, 1)
        )
    else:
        base["planned_qty"] = 0
        base["execution_mode"] = None
        base["progress_qty"] = None
        base["progress"] = base["progress_steps"].clip(0, 1)

    out_cols = [
        "product_id",
        "run_id",
        "planned_qty",
        "execution_mode",
        "progress_qty",
        "progress_steps",
        "progress",
    ]
    return (
        base.loc[:, [c for c in out_cols if c in base.columns]]
        .sort_values(["product_id", "run_id"])
        .reset_index(drop=True)
    )
