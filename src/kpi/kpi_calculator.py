import pandas as pd
from typing import Dict


def compute_throughput(production_log: pd.DataFrame) -> float:
    """Placeholder. Implement throughput calculation later."""
    raise NotImplementedError("compute_throughput is not implemented yet")


def compute_wip(production_log: pd.DataFrame) -> int:
    """Placeholder. Implement WIP calculation later."""
    raise NotImplementedError("compute_wip is not implemented yet")


def compute_on_time_rate(
    process_steps: pd.DataFrame, production_log: pd.DataFrame
) -> float:
    """Placeholder. Implement on-time rate later."""
    raise NotImplementedError("compute_on_time_rate is not implemented yet")


def compute_all_kpis(tables: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Dashboard-safe placeholder values until KPI logic is implemented."""
    return {"throughput": 0.0, "wip": 0, "on_time_rate": 0.0}
