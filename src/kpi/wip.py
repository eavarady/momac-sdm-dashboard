import pandas as pd


def compute_wip(production_log: pd.DataFrame) -> int:
    """Sum of quantities for in-progress rows from normalized production_log."""
    if (
        production_log is None
        or production_log.empty
        or "status" not in production_log.columns
    ):
        return 0
    in_progress = production_log[production_log["status"] == "in_progress"]
    if in_progress.empty:
        return 0
    return int(in_progress["quantity"].sum())
