from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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
    output_path: str = "linear_forecasted_data.csv",
) -> pd.DataFrame:
    """
    Linear regression forecast using sklearn (y ~ a + b * t).

    Inputs:
      - Either (ds, y) series OR raw logs with (start_time, end_time), which will be
        aggregated to (ds, y) using the same ts_utils helpers as Prophet.

    Output:
      - A DataFrame with columns: ds, yhat, yhat_lower, yhat_upper, model, (and y for history),
        written to `output_path` for the dashboard to consume.
    """
    # prep to (ds, y)
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
            if raw.empty:
                raise ValueError("No valid (start_time, end_time) rows after parsing datetimes")
            durations = (raw["end_time"] - raw["start_time"]).dt.total_seconds() / 3600.0
            work = pd.DataFrame({"ds": raw["end_time"], "y": durations})
    else:
        raise ValueError("DataFrame must contain either ('ds','y') or ('start_time','end_time') columns")

    # clean & sort
    work = (
        work.dropna(subset=["ds", "y"])
            .assign(y=lambda d: pd.to_numeric(d["y"], errors="coerce"))
            .dropna(subset=["y"])
            .sort_values("ds")
            .drop_duplicates(subset=["ds"])
    )
    # strip timezone for sklearn / plotting consistency
    if getattr(work["ds"].dt, "tz", None) is not None:
        work["ds"] = work["ds"].dt.tz_convert("UTC").dt.tz_localize(None)

    if require_non_negative:
        work = work[work["y"] >= 0]

    if len(work) < min_rows:
        raise ValueError(f"Not enough rows for forecasting (have {len(work)}, need >= {min_rows})")

    # effective horizon (same rule as Prophet path)
    if adapt_horizon:
        span_days = max(1, (work["ds"].max() - work["ds"].min()).days + 1)
        max_allowed = max(1, int(span_days * horizon_multiplier))
        effective_horizon = min(horizon, max_allowed)
    else:
        effective_horizon = horizon

    # fit OLS on time index
    n = len(work)
    t_hist = np.arange(n, dtype=float).reshape(-1, 1)
    y_hist = work["y"].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(t_hist, y_hist)

    # predict for history (smooth fitted line) and future (extrapolation)
    hist_yhat = model.predict(t_hist).astype(float)

    if effective_horizon > 0:
        future_ds = _infer_idx(work["ds"].max(), effective_horizon, work["ds"])
        t_fut = np.arange(n, n + effective_horizon, dtype=float).reshape(-1, 1)
        future_yhat = model.predict(t_fut).astype(float)
    else:
        future_ds = []
        future_yhat = np.array([], dtype=float)

    # simple uncertainty bands from residual stddev (homoskedastic assumption)
    if n > 2:
        resid = y_hist - hist_yhat
        sigma = float(resid.std(ddof=1))
    else:
        sigma = 0.0

    # concatenate series
    combined_ds = list(work["ds"]) + list(future_ds)
    combined_yhat = np.concatenate([hist_yhat, future_yhat])
    if require_non_negative:
        combined_yhat = np.clip(combined_yhat, 0.0, None)

    # 95% pseudo-intervals (constant sigma)
    if sigma > 0:
        intervals = {
            "yhat_lower": combined_yhat - 1.96 * sigma,
            "yhat_upper": combined_yhat + 1.96 * sigma,
        }
        if require_non_negative:
            intervals["yhat_lower"] = np.clip(intervals["yhat_lower"], 0.0, None)
    else:
        intervals = {"yhat_lower": combined_yhat, "yhat_upper": combined_yhat}

    # build standardized forecast frame and merge historic y
    forecast = build_forecast_frame(
        combined_ds,
        combined_yhat,
        model_label="linear-regression",
        intervals=intervals,
    ).merge(work[["ds", "y"]], on="ds", how="left")

    forecast.to_csv(output_path, index=False)
    return forecast