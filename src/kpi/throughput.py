import pandas as pd


def compute_throughput(production_log: pd.DataFrame) -> float:
    """Compute throughput (units/sec) from normalized production_log.

    Assumes data layer normalizes status (e.g., 'completed' -> 'complete') and timestamps (UTC).
    Uses end_time of completed items as the production completion window.
    """
    if production_log is None or production_log.empty:
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
