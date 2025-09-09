import pandas as pd
from typing import Optional


def detect_bottleneck(
    process_steps: pd.DataFrame,  # not required for v1, kept for future enrichment
    production_log: pd.DataFrame,
) -> Optional[str]:
    """
    Heuristic v1: Pick the step with the highest WIP (work-in-progress).

    Definition:
      WIP(step) = sum(quantity) over rows where status == "in_progress", grouped by step_id.

    Returns:
      - step_id (str) of the bottleneck if any in-progress work exists
      - None if there is no in-progress work or data is missing

    Tie-breaks:
      - If multiple steps have the same WIP, choose lexicographically smallest step_id.
    """

    # 0) Fast exits for empty inputs
    if production_log is None or production_log.empty:
        return None

    # 1) Ensure required columns are present; if not, bail out safely
    required_cols = {"step_id", "status", "quantity"}
    if not required_cols.issubset(production_log.columns):
        return None

    # 2) Work on a copy so we don't mutate the caller's DataFrame
    df = production_log.copy()

    # 3) Normalize types and values
    #    - Coerce quantity to numeric; invalid values become NaN
    #    - Drop rows with NaN, zero or negative quantities (they don't contribute to WIP)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["quantity"])
    df = df[df["quantity"] > 0]

    # 4) Normalize status and filter to only in-progress rows
    #    - Lowercase so we tolerate "In_Progress", "IN_PROGRESS", etc.
    df["status"] = df["status"].astype(str).str.lower()
    wip_rows = df[df["status"] == "in_progress"].copy()
    if wip_rows.empty:
        return None

    # 5) Normalize step_id to string for predictable grouping & tie-breaks
    wip_rows.loc[:, "step_id"] = wip_rows["step_id"].astype(str)

    # 6) Group by step_id and sum quantities to compute WIP per step
    wip_by_step = (
        wip_rows.groupby("step_id", dropna=False)["quantity"]
        .sum()
        .reset_index(name="wip")
    )
    if wip_by_step.empty:
        return None

    # 7) Sort to pick the bottleneck
    #    - Highest WIP first
    #    - Tie-break: lexicographically smallest step_id
    wip_by_step = wip_by_step.sort_values(
        by=["wip", "step_id"], ascending=[False, True], kind="mergesort"
    )

    top = wip_by_step.iloc[0]
    # Defensive guard: if top.wip somehow ended up 0 after filtering
    if top["wip"] <= 0:
        return None

    # 8) Return only the step_id for v1 (simple API)
    return str(top["step_id"])


def top_bottlenecks(
    production_log: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Return the top-N bottlenecks by WIP (work-in-progress).

    WIP(step) = sum(quantity) over rows where status == "in_progress".

    Args:
        production_log: DataFrame with at least columns [step_id, status, quantity]
        top_n: number of top rows to return (default=3)

    Returns:
        DataFrame with columns ["step_id", "total_wip"], sorted by WIP desc then step_id.
    """
    if production_log is None or production_log.empty:
        return pd.DataFrame(columns=["step_id", "total_wip"])

    df = production_log.copy()
    if not {"step_id", "status", "quantity"}.issubset(df.columns):
        return pd.DataFrame(columns=["step_id", "total_wip"])

    # Clean quantities
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["quantity"])
    df = df[df["quantity"] > 0]

    # Filter WIP
    df["status"] = df["status"].astype(str).str.lower()
    wip = df[df["status"] == "in_progress"].copy()
    if wip.empty:
        return pd.DataFrame(columns=["step_id", "total_wip"])

    wip.loc[:, "step_id"] = wip["step_id"].astype(str)

    # Aggregate and sort
    agg = (
        wip.groupby("step_id")["quantity"].sum().reset_index(name="total_wip")
    )
    agg = agg.sort_values(by=["total_wip", "step_id"], ascending=[False, True])

    return agg.head(top_n).reset_index(drop=True)