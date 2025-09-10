from __future__ import annotations

from typing import Optional
import pandas as pd

from .ts_utils import (
    aggregate_duration_series as _agg_series,
    _infer_future_index as _infer_idx,
    build_forecast_frame,
)


def linear_forecast(
    df: pd.DataFrame,
    *,
    horizon: int = 90,
    require_non_negative: bool = True,
    min_rows: int = 2,
    use_aggregation: bool = True,
    agg_freq: str = "D",
    agg_metric: str = "mean",
    adapt_horizon: bool = True,
    horizon_multiplier: float = 1.0,
    output_path: str = "time_series_forecasted_data.csv",
) -> pd.DataFrame:
    """Placeholder linear model forecast.

    Currently: echoes historical y as yhat and holds last value for future.
    TODO: Replace with actual linear regression fit/predict (e.g., Ordinary
    Least Squares on index vs y) retaining identical output schema.
    """
    if {"ds", "y"}.issubset(df.columns):
        work = df[["ds", "y"]].copy()
        work["ds"] = pd.to_datetime(work["ds"], errors="coerce", utc=True)
    elif {"start_time", "end_time"}.issubset(df.columns):
        if use_aggregation:
            work = _agg_series(df, freq=agg_freq, metric=agg_metric)
            work["ds"] = pd.to_datetime(work["ds"], errors="coerce", utc=True)
        else:
            raw = df[["start_time", "end_time"]].copy()
            raw["start_time"] = pd.to_datetime(raw["start_time"], errors="coerce", utc=True)
            raw["end_time"] = pd.to_datetime(raw["end_time"], errors="coerce", utc=True)
            raw = raw.dropna(subset=["start_time", "end_time"])
            durations = (raw["end_time"] - raw["start_time"]).dt.total_seconds() / 3600.0
            work = pd.DataFrame({"ds": raw["end_time"], "y": durations})
    else:
        raise ValueError("DataFrame must contain either ('ds','y') or ('start_time','end_time') columns")

    work = work.dropna(subset=["ds", "y"])
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work = work.dropna(subset=["y"]).sort_values("ds").drop_duplicates(subset=["ds"])
    if getattr(work["ds"].dt, "tz", None) is not None:
        work["ds"] = work["ds"].dt.tz_convert("UTC").dt.tz_localize(None)
    if require_non_negative:
        work = work[work["y"] >= 0]
    if len(work) < min_rows:
        raise ValueError(f"Not enough rows for forecasting (have {len(work)}, need >= {min_rows})")

    if adapt_horizon:
        span_days = max(1, (work["ds"].max() - work["ds"].min()).days + 1)
        max_allowed = max(1, int(span_days * horizon_multiplier))
        effective_horizon = min(horizon, max_allowed)
    else:
        effective_horizon = horizon

    future_ds = _infer_idx(work["ds"].max(), effective_horizon, work["ds"]) if effective_horizon > 0 else []

    # Placeholder "model": flat hold of last value into future
    last_val = float(work["y"].iloc[-1])
    hist_yhat = list(work["y"].astype(float))
    future_yhat = [last_val] * effective_horizon
    combined_ds = list(work["ds"]) + future_ds
    combined_yhat = hist_yhat + future_yhat

    forecast = build_forecast_frame(combined_ds, combined_yhat, model_label="linear-placeholder")
    # Merge historical y
    forecast = forecast.merge(work[["ds", "y"]], on="ds", how="left")
    forecast.to_csv(output_path, index=False)
    return forecast
