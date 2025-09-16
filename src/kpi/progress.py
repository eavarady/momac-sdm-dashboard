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
    targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute per-step progress as quantity-complete / (quantity-complete + quantity-in_progress).

    Returns DataFrame with:
      product_id, step_id[, step_name], complete_qty, in_progress_qty, progress (0..1)
    """
    # Base universe of steps (so steps with no activity still appear as 0%)
    cols = ["product_id", "step_id"] + (
        ["step_name"] if "step_name" in process_steps.columns else []
    )
    base = process_steps.loc[:, cols].drop_duplicates().copy()
    if base.empty:
        # Nothing to compute
        return pd.DataFrame(
            columns=cols + ["complete_qty", "in_progress_qty", "progress"]
        )

    # Aggregate production log by status
    log = production_log.copy()
    if not log.empty:
        log = _normalize_status_col(log)
        # make sure quantity is numeric
        if "quantity" in log.columns:
            log["quantity"] = pd.to_numeric(log["quantity"], errors="coerce").fillna(0)
        else:
            log["quantity"] = 0

        agg = (
            log.groupby(["product_id", "step_id", "status"], dropna=False)["quantity"]
            .sum()
            .reset_index()
        )
        # Pivot statuses to columns we care about
        pivot = agg.pivot_table(
            index=["product_id", "step_id"],
            columns="status",
            values="quantity",
            fill_value=0,
            aggfunc="sum",
        ).reset_index()
        # Ensure both columns exist
        for col in ("complete", "in_progress"):
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot.rename(
            columns={"complete": "complete_qty", "in_progress": "in_progress_qty"}
        )
    else:
        # No log -> zeros for all steps
        pivot = base[["product_id", "step_id"]].copy()
        pivot["complete_qty"] = 0
        pivot["in_progress_qty"] = 0

    # Join base steps with the aggregated quantities
    merged = base.merge(pivot, on=["product_id", "step_id"], how="left")
    merged["complete_qty"] = merged["complete_qty"].fillna(0)
    merged["in_progress_qty"] = merged["in_progress_qty"].fillna(0)

    denom = merged["complete_qty"] + merged["in_progress_qty"]
    merged["progress"] = 0.0
    nonzero = denom > 0
    merged.loc[nonzero, "progress"] = (
        merged.loc[nonzero, "complete_qty"] / denom.loc[nonzero]
    ).clip(0, 1)

    # If targets provided, override per-step progress where a positive target exists
    if targets is not None and not targets.empty:
        # For backward compatibility, only apply overrides when step-level targets are present.
        if {"product_id", "step_id", "target_qty"}.issubset(targets.columns):
            t = targets[["product_id", "step_id", "target_qty"]].copy()
            t["target_qty"] = pd.to_numeric(t["target_qty"], errors="coerce").fillna(0)
            merged = merged.merge(t, on=["product_id", "step_id"], how="left")

            has_target = merged["target_qty"].fillna(0) > 0
            merged.loc[has_target, "progress"] = (
                merged.loc[has_target, "complete_qty"]
                / merged.loc[has_target, "target_qty"]
            ).clip(0, 1)

    # Order columns nicely
    ordered_cols = ["product_id", "step_id"]
    if "step_name" in merged.columns:
        ordered_cols.append("step_name")
    # If target_qty is present, show it before progress
    cols_tail = ["complete_qty", "in_progress_qty"]
    if "target_qty" in merged.columns:
        cols_tail.append("target_qty")
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

    # Compute mean progress per product
    grp = step_progress.groupby("product_id", as_index=False)["progress"].mean()
    # Ensure numeric and clip only the progress column (not the whole DataFrame)
    grp["progress"] = pd.to_numeric(grp["progress"], errors="coerce").fillna(0.0)
    grp["progress"] = grp["progress"].clip(lower=0.0, upper=1.0)
    grp = grp.rename(columns={"progress": "overall_progress"})

    return grp[["product_id", "overall_progress"]]


def per_step_progress_by_run(
    process_steps: pd.DataFrame,
    production_log: pd.DataFrame,
    targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute per-step progress per run.
    Returns DataFrame with:
      product_id, run_id, step_id[, step_name], complete_qty, in_progress_qty, progress (0..1)

    Notes:
    - Targets are currently ignored unless step-level targets are provided; run-level targets
      should be applied at run roll-up time, not per-step, to avoid ambiguity.
    """
    cols = ["product_id", "step_id"] + (
        ["step_name"] if "step_name" in process_steps.columns else []
    )
    base = process_steps.loc[:, cols].drop_duplicates().copy()
    if base.empty or production_log.empty:
        out_cols = (
            ["product_id", "run_id", "step_id"]
            + (["step_name"] if "step_name" in base.columns else [])
            + ["complete_qty", "in_progress_qty", "progress"]
        )
        return pd.DataFrame(columns=out_cols)

    log = _normalize_status_col(production_log.copy())
    if "quantity" in log.columns:
        log["quantity"] = pd.to_numeric(log["quantity"], errors="coerce").fillna(0)
    else:
        log["quantity"] = 0

    # Identify runs present
    runs = (
        log[["product_id", "run_id"]]
        .dropna(subset=["run_id"])  # only runs with an id
        .drop_duplicates()
    )
    if runs.empty:
        # No run ids -> fall back to empty
        out_cols = (
            ["product_id", "run_id", "step_id"]
            + (["step_name"] if "step_name" in base.columns else [])
            + ["complete_qty", "in_progress_qty", "progress"]
        )
        return pd.DataFrame(columns=out_cols)

    # Cross runs with steps to ensure unstarted steps appear
    cross = runs.merge(base, on="product_id", how="left")

    agg = (
        log.groupby(["product_id", "run_id", "step_id", "status"], dropna=False)[
            "quantity"
        ]
        .sum()
        .reset_index()
    )
    pivot = agg.pivot_table(
        index=["product_id", "run_id", "step_id"],
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

    merged = cross.merge(pivot, on=["product_id", "run_id", "step_id"], how="left")
    merged["complete_qty"] = merged["complete_qty"].fillna(0)
    merged["in_progress_qty"] = merged["in_progress_qty"].fillna(0)

    denom = merged["complete_qty"] + merged["in_progress_qty"]
    merged["progress"] = 0.0
    nz = denom > 0
    merged.loc[nz, "progress"] = (merged.loc[nz, "complete_qty"] / denom.loc[nz]).clip(
        0, 1
    )

    # Ignore run-level targets here; show target_qty only if step-level targets provided
    if targets is not None and not targets.empty:
        if {"product_id", "step_id", "target_qty"}.issubset(targets.columns):
            t = targets[["product_id", "step_id", "target_qty"]].copy()
            t["target_qty"] = pd.to_numeric(t["target_qty"], errors="coerce").fillna(0)
            merged = merged.merge(t, on=["product_id", "step_id"], how="left")

    ordered_cols = ["product_id", "run_id", "step_id"]
    if "step_name" in merged.columns:
        ordered_cols.insert(2, "step_name")
    tail = ["complete_qty", "in_progress_qty"]
    if "target_qty" in merged.columns:
        tail.append("target_qty")
    tail.append("progress")
    ordered_cols += tail
    return merged.loc[:, [c for c in ordered_cols if c in merged.columns]]
