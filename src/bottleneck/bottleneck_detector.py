import pandas as pd
from typing import Optional


def detect_bottleneck(
    process_steps: pd.DataFrame, production_log: pd.DataFrame
) -> Optional[str]:
    """Placeholder. Implement bottleneck detection later."""
    if process_steps.empty or production_log.empty:
        return None
    return None
