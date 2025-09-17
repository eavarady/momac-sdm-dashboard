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
      - progress: for planned runs, use progress_qty if planned_qty > 1, else progress_steps (unit runs)
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

    # Step universe and totals per product (fixed denominator)
    steps_pp = process_steps[["product_id", "step_id"]].drop_duplicates()
    totals = (
        steps_pp.groupby("product_id", as_index=False)["step_id"]
        .nunique()
        .rename(columns={"step_id": "total_steps"})
    )

    # Run keys we care about (from log; optionally union runs table if product_id present)
    run_keys = (
        log[["product_id", "run_id"]].dropna(subset=["run_id"]).drop_duplicates().copy()
    )
    if runs is not None and not runs.empty and "product_id" in runs.columns:
        rk = (
            runs[["product_id", "run_id"]]
            .dropna(subset=["product_id", "run_id"])
            .drop_duplicates()
        )
        if not rk.empty:
            run_keys = (
                pd.concat([run_keys, rk], ignore_index=True)
                .drop_duplicates()
                .reset_index(drop=True)
            )

    if run_keys.empty:
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

    # Completed steps per run (numerator)
    completed = (
        log.loc[log["status"] == "complete"]
        .dropna(subset=["run_id"])
        .groupby(["product_id", "run_id", "step_id"], as_index=False)["quantity"]
        .sum()
    )
    completed["done"] = (completed["quantity"] > 0).astype(int)
    steps_completed = (
        completed.groupby(["product_id", "run_id"], as_index=False)["done"]
        .sum()
        .rename(columns={"done": "steps_completed"})
    )

    # progress_steps = steps_completed / total_steps (clip 0..1)
    per_run_step = run_keys.merge(
        steps_completed, on=["product_id", "run_id"], how="left"
    ).merge(totals, on="product_id", how="left")
    per_run_step["steps_completed"] = per_run_step["steps_completed"].fillna(0)
    per_run_step["total_steps"] = pd.to_numeric(
        per_run_step["total_steps"], errors="coerce"
    ).fillna(0)
    per_run_step["progress_steps"] = 0.0
    valid_den = per_run_step["total_steps"] > 0
    per_run_step.loc[valid_den, "progress_steps"] = (
        per_run_step.loc[valid_den, "steps_completed"]
        / per_run_step.loc[valid_den, "total_steps"]
    ).clip(0, 1)

    # Determine terminal steps per product (no dependents)
    steps = process_steps[["product_id", "step_id", "dependency_step_id"]].copy()
    dependents = steps.dropna(subset=["dependency_step_id"])[
        "dependency_step_id"
    ].astype(str)
    steps["is_terminal"] = ~steps["step_id"].astype(str).isin(dependents.astype(str))

    # Completed output qty at terminal steps per run
    term = steps.loc[steps["is_terminal"], ["product_id", "step_id"]]
    log_c = log[log["status"] == "complete"]
    qty_out = (
        log_c.merge(term, on=["product_id", "step_id"], how="inner")
        .groupby(["product_id", "run_id"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "qty_out"})
    )

    base = (
        per_run_step.merge(qty_out, on=["product_id", "run_id"], how="left")
        .rename(columns={"qty_out": "qty_out"})
        .copy()
    )
    base["qty_out"] = pd.to_numeric(base.get("qty_out", 0), errors="coerce").fillna(0)

    # Planned quantity (optional)
    planned = None
    if (
        runs is not None
        and not runs.empty
        and {"run_id", "planned_qty"}.issubset(runs.columns)
    ):
        cols = ["run_id", "planned_qty"]
        if "execution_mode" in runs.columns:
            cols.append("execution_mode")
        planned = runs[cols].copy()

    if planned is not None:
        base = base.merge(planned, on="run_id", how="left")
        base["planned_qty"] = pd.to_numeric(
            base.get("planned_qty", 0), errors="coerce"
        ).fillna(0)

        has_plan = base["planned_qty"] > 0
        base["progress_qty"] = None
        base.loc[has_plan, "progress_qty"] = (
            base.loc[has_plan, "qty_out"] / base.loc[has_plan, "planned_qty"]
        ).clip(0, 1)

        # Default to step-based; for batch runs (planned_qty > 1), use qty-based
        base["progress_steps"] = base["progress_steps"].clip(0, 1)
        base["progress"] = base["progress_steps"]
        use_qty = base["planned_qty"] > 1
        base.loc[use_qty & base["progress_qty"].notna(), "progress"] = base.loc[
            use_qty & base["progress_qty"].notna(), "progress_qty"
        ]

        # If qty indicates completion, force progress to 1.0 regardless of mode
        qty_done = base["progress_qty"].fillna(0) >= 1.0
        base.loc[qty_done, "progress"] = 1.0
        base["progress"] = base["progress"].clip(0, 1)
    else:
        base["planned_qty"] = 0
        if "execution_mode" not in base.columns:
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
