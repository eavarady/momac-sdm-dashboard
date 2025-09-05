from typing import Optional
import pandas as pd


def safe_int(value, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def apply_tolerance_seconds(
    planned_sec: pd.Series | float,
    tolerance_seconds: int = 0,
    tolerance_pct: float = 0.0,
):
    """
    Returns planned_sec * (1 + tolerance_pct) + tolerance_seconds.
    Works with scalars or pandas Series.
    """
    return planned_sec * (1.0 + float(tolerance_pct)) + float(tolerance_seconds)


def tolerance_extra_seconds(
    planned_sec: pd.Series | float,
    tolerance_seconds: int = 0,
    tolerance_pct: float = 0.0,
):
    """
    Returns only the extra tolerance amount in seconds:
    planned_sec * tolerance_pct + tolerance_seconds.
    Useful when you already have a planned timestamp and need to add tolerance.
    """
    return planned_sec * float(tolerance_pct) + float(tolerance_seconds)


def weighted_mean(values: pd.Series, weights: Optional[pd.Series] = None) -> float:
    """
    Quantity-weighted average with safe fallbacks.
    - values: numeric or boolean Series (booleans are coerced to float).
    - weights: optional numeric Series; if missing/invalid/zero-sum -> unweighted mean.
    """
    if values is None or len(values) == 0:
        return 0.0
    v = pd.Series(values).astype(float)
    if weights is None:
        return float(v.mean())
    w = pd.Series(weights).fillna(0).astype(float)
    total = float(w.sum())
    if total <= 0:
        return float(v.mean())
    return float((v * w).sum() / total)
