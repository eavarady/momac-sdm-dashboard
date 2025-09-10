"""Reusable, model-agnostic time-series helpers.

This module intentionally avoids heavy ML dependencies (Prophet, sklearn, etc.)
so it can be imported by multiple forecasting backends without causing large
import costs or circular dependency risks.
"""

from __future__ import annotations

from typing import Sequence, Optional, Iterable, Dict
import pandas as pd
import numpy as np


def aggregate_duration_series(
    df: pd.DataFrame,
    *,
    freq: str = "D",
    metric: str = "mean",
    status_complete_values: tuple[str, ...] = ("complete",),
    require_positive: bool = True,
    min_points: int = 1,
) -> pd.DataFrame:
    """Turn (start_time, end_time [, status]) rows into a regular (ds, y) series.

    Parameters mirror the original implementation (moved from time_series.py).
    See original docstring for detailed semantics.
    """
    required = {"start_time", "end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[list(required) + (["status"] if "status" in df.columns else [])].copy()

    for col in ("start_time", "end_time"):
        work[col] = pd.to_datetime(work[col], errors="coerce", utc=True)

    work = work.dropna(subset=["start_time", "end_time"])
    if "status" in work.columns:
        work = work[work["status"].isin(status_complete_values)]

    if work.empty:
        raise ValueError("No completed rows after filtering & datetime parsing")

    work["duration_h"] = (work["end_time"] - work["start_time"]).dt.total_seconds() / 3600.0
    if require_positive:
        work = work[work["duration_h"] > 0]
    if len(work) < min_points:
        raise ValueError(
            f"Insufficient completed rows for aggregation (have {len(work)}, need >= {min_points})"
        )

    agg_map: Dict[str, str] = {"mean": "mean", "median": "median", "sum": "sum", "count": "count"}
    if metric not in agg_map:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(agg_map)}")

    events = work[["end_time", "duration_h"]].rename(columns={"end_time": "timestamp"})
    events = events.set_index("timestamp").sort_index()
    if metric == "count":
        agg_series = events["duration_h"].resample(freq).count()
    else:
        agg_series = getattr(events["duration_h"].resample(freq), agg_map[metric])()
    agg_series = agg_series.dropna()
    if agg_series.empty:
        raise ValueError("Aggregation produced an empty series")

    ds = agg_series.index
    if getattr(ds, "tz", None) is not None:
        ds = ds.tz_convert("UTC").tz_localize(None)
    out = pd.DataFrame({"ds": ds, "y": agg_series.values})
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["y"])
    if out.empty:
        raise ValueError("Resulting aggregated series is empty after numeric coercion")
    return out


def _infer_future_index(last_ds: pd.Timestamp, horizon: int, work_ds: pd.Series):
    """Infer future timestamps using robust date_range logic.

    Mirrors implementation moved from time_series.py.
    """
    freq = pd.infer_freq(work_ds)
    if freq is None:
        diffs = work_ds.sort_values().diff().dropna()
        if not diffs.empty:
            delta = diffs.median()
        else:
            delta = pd.Timedelta(days=1)
        future_idx = pd.date_range(start=last_ds + delta, periods=horizon, freq=delta)
        return list(future_idx)
    future_idx = pd.date_range(start=last_ds, periods=horizon + 1, freq=freq)[1:]
    return list(future_idx)


def build_forecast_frame(
    ds: Sequence[pd.Timestamp],
    yhat: Sequence[float],
    *,
    model_label: str,
    y: Optional[Sequence[float]] = None,
    intervals: Optional[dict] = None,
) -> pd.DataFrame:
    """Assemble a standardized forecast DataFrame.

    intervals dict may contain 'yhat_lower' and 'yhat_upper'. If absent, they
    default to yhat. Ensures ds dtype is datetime64[ns] and numeric coercion
    for yhat (and y when provided).
    """
    if len(ds) != len(yhat):
        raise ValueError("Length of ds and yhat must match")

    frame = pd.DataFrame({"ds": pd.to_datetime(list(ds), utc=False), "yhat": yhat})
    frame["yhat"] = pd.to_numeric(frame["yhat"], errors="coerce")
    if y is not None:
        if len(y) != len(ds):
            raise ValueError("Length of y must match ds when provided")
        frame["y"] = pd.to_numeric(list(y), errors="coerce")
    if intervals is not None:
        lower = intervals.get("yhat_lower", frame["yhat"])  # type: ignore[arg-type]
        upper = intervals.get("yhat_upper", frame["yhat"])  # type: ignore[arg-type]
    else:
        lower = frame["yhat"]
        upper = frame["yhat"]
    frame["yhat_lower"] = pd.to_numeric(lower, errors="coerce")
    frame["yhat_upper"] = pd.to_numeric(upper, errors="coerce")
    frame["model"] = model_label
    return frame
