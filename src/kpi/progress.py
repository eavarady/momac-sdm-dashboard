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

    if targets is not None and not targets.empty:
        if {"product_id", "step_id", "target_qty"}.issubset(targets.columns):
            t = targets[["product_id", "step_id", "target_qty"]].copy()
            t["target_qty"] = pd.to_numeric(t["target_qty"], errors="coerce").fillna(0)
            merged = merged.merge(t, on=["product_id", "step_id"], how="left")
            has_target = merged["target_qty"].fillna(0) > 0
            merged.loc[has_target, "progress"] = (
                merged.loc[has_target, "complete_qty"]
                / merged.loc[has_target, "target_qty"]
            ).clip(0, 1)

    ordered_cols = ["product_id", "step_id"]
    if "step_name" in merged.columns:
        ordered_cols.append("step_name")
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
    grp = step_progress.groupby("product_id", as_index=False)["progress"].mean()
    grp["progress"] = pd.to_numeric(grp["progress"], errors="coerce").fillna(0.0)
    grp["progress"] = grp["progress"].clip(lower=0.0, upper=1.0)
    grp = grp.rename(columns={"progress": "overall_progress"})
    return grp[["product_id", "overall_progress"]]


def per_run_progress(
    process_steps: pd.DataFrame,
    production_log: pd.DataFrame,
    targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute overall progress per run (one row per run_id), using:
      - Per-step quantities from production_log grouped by (product_id, run_id, step_id)
      - Run-level target_qty when provided (preferred)
      - Otherwise, per-run progress is computed as mean of step progresses

    Returns DataFrame: [product_id, run_id, progress] (0..1),
    and includes target_qty if supplied.
    """
    if production_log.empty:
        return pd.DataFrame(columns=["product_id", "run_id", "progress"])

    log = _normalize_status_col(production_log.copy())
    if "quantity" in log.columns:
        log["quantity"] = pd.to_numeric(log["quantity"], errors="coerce").fillna(0)
    else:
        log["quantity"] = 0

    agg = (
        log.groupby(["product_id", "run_id", "step_id", "status"], dropna=False)[
            "quantity"
        ]
        .sum()
        .reset_index()
    )
    if agg.empty:
        return pd.DataFrame(columns=["product_id", "run_id", "progress"])

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

    denom = pivot["complete_qty"] + pivot["in_progress_qty"]
    step_prog = pivot.copy()
    step_prog["step_progress"] = 0.0
    nz = denom > 0
    step_prog.loc[nz, "step_progress"] = (
        step_prog.loc[nz, "complete_qty"] / denom.loc[nz]
    ).clip(0, 1)
    run_prog = (
        step_prog.groupby(["product_id", "run_id"], as_index=False)["step_progress"]
        .mean()
        .rename(columns={"step_progress": "progress"})
    )

    if (
        targets is not None
        and not targets.empty
        and {"run_id", "target_qty"}.issubset(targets.columns)
    ):
        t = targets[["run_id", "target_qty"]].copy()
        t["target_qty"] = pd.to_numeric(t["target_qty"], errors="coerce").fillna(0)
        run_prog = run_prog.merge(t, on="run_id", how="left")

        total_complete = pivot.groupby(["product_id", "run_id"], as_index=False)[
            "complete_qty"
        ].sum()
        run_prog = run_prog.merge(
            total_complete, on=["product_id", "run_id"], how="left"
        )
        has_t = run_prog["target_qty"].fillna(0) > 0
        run_prog.loc[has_t, "progress"] = (
            run_prog.loc[has_t, "complete_qty"] / run_prog.loc[has_t, "target_qty"]
        ).clip(0, 1)
        run_prog.drop(columns=["complete_qty"], inplace=True)

    cols = ["product_id", "run_id", "progress"]
    if "target_qty" in run_prog.columns:
        cols.append("target_qty")
    return run_prog.loc[:, cols]
