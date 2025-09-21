from adapters import csv_adapter as adapter  # placeholder (not yet used)
import pandas as pd
import prophet as pf
from prophet.models import StanBackendEnum
from .ts_utils import (
    aggregate_duration_series,
    _infer_future_index,
    build_forecast_frame,
)

# Opt-in to pandas future behavior to avoid silent downcasting in this module too
pd.set_option('future.no_silent_downcasting', True)


def _baseline_forecast(
    work: pd.DataFrame,
    horizon: int,
    strategy: str,
) -> pd.DataFrame:
    if strategy not in {"mean", "linear"}:
        raise ValueError("baseline_strategy must be one of {'mean','linear'}")
    future_ds = _infer_future_index(work["ds"].max(), horizon, work["ds"])
    if strategy == "mean":
        base_value = float(work["y"].mean())
        hist_pred = [base_value] * len(work)
        future_pred = [base_value] * horizon
        label = "baseline-mean"
    else:  # linear
        if len(work) == 1:
            slope = 0.0
        else:
            slope = (work["y"].iloc[-1] - work["y"].iloc[0]) / max(1, len(work) - 1)
        hist_pred = [work["y"].iloc[0] + slope * i for i in range(len(work))]
        future_pred = [work["y"].iloc[-1] + slope * (i + 1) for i in range(horizon)]
        label = "baseline-linear"
    combined_ds = list(work["ds"]) + future_ds
    combined_yhat = hist_pred + future_pred
    forecast = build_forecast_frame(combined_ds, combined_yhat, model_label=label)
    forecast = forecast.merge(work[["ds", "y"]], on="ds", how="left")
    return forecast


def time_series_forecast(
    df: pd.DataFrame,
    *,
    horizon: int = 90,
    require_non_negative: bool = True,
    min_rows: int = 2,
    min_prophet_points: int = 20,
    baseline_strategy: str = "mean",
    use_aggregation: bool = True,
    agg_freq: str = "D",
    agg_metric: str = "mean",
    adapt_horizon: bool = True,
    horizon_multiplier: float = 1.0,
    fallback_on_failure: bool = True,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Forecast durations.

    Data sources:
      - If df has (ds, y): used directly.
      - Else if df has (start_time, end_time): durations derived. If use_aggregation=True
        these are aggregated (daily by default) for a regular series.

    Fallback logic:
      1. If after prep len(work) < min_prophet_points -> baseline forecast (mean or linear).
      2. If Prophet fit/predict raises -> baseline (if fallback_on_failure True).

    Adaptive horizon:
      If adapt_horizon and history span (days) * horizon_multiplier < requested horizon,
      horizon is reduced accordingly (min 1).

    Returns forecast DataFrame with columns at least ds, yhat, yhat_lower,
    yhat_upper, model, and historic y where available.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Derive or accept ds,y
    if {"ds", "y"}.issubset(df.columns):
        work = df[["ds", "y"]].copy()
        work["ds"] = pd.to_datetime(work["ds"], errors="coerce", utc=True)
    elif {"start_time", "end_time"}.issubset(df.columns):
        if use_aggregation:
            work = aggregate_duration_series(
                df,
                freq=agg_freq,
                metric=agg_metric,
            )
        else:
            raw = df[["start_time", "end_time"]].copy()
            raw["start_time"] = pd.to_datetime(
                raw["start_time"], errors="coerce", utc=True
            )
            raw["end_time"] = pd.to_datetime(raw["end_time"], errors="coerce", utc=True)
            raw = raw.dropna(subset=["start_time", "end_time"])
            if raw.empty:
                raise ValueError(
                    "No valid (start_time, end_time) rows after parsing datetimes"
                )
            durations = (
                raw["end_time"] - raw["start_time"]
            ).dt.total_seconds() / 3600.0
            work = pd.DataFrame({"ds": raw["end_time"], "y": durations})
    else:
        raise ValueError(
            "DataFrame must contain either ('ds','y') or ('start_time','end_time') columns"
        )

    # Clean
    work = work.dropna(subset=["ds", "y"])
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work = work.dropna(subset=["y"])

    if getattr(work["ds"].dt, "tz", None) is not None:
        work["ds"] = work["ds"].dt.tz_convert("UTC").dt.tz_localize(None)

    if not pd.api.types.is_datetime64_any_dtype(work["ds"]):
        raise ValueError("'ds' column could not be converted to datetime")

    if require_non_negative:
        work = work[work["y"] >= 0]

    work = work.sort_values("ds").drop_duplicates(subset=["ds"])
    if len(work) < min_rows:
        raise ValueError(
            f"Not enough rows for forecasting (have {len(work)}, need >= {min_rows})"
        )

    # Adaptive horizon reduction (applied before baseline / prophet decision)
    if adapt_horizon:
        span_days = max(1, (work["ds"].max() - work["ds"].min()).days + 1)
        max_allowed = max(1, int(span_days * horizon_multiplier))
        effective_horizon = min(horizon, max_allowed)
    else:
        effective_horizon = horizon

    # Pre-fit baseline decision (use effective_horizon)
    if len(work) < min_prophet_points:
        forecast = _baseline_forecast(work, effective_horizon, baseline_strategy)
        if output_path:
            forecast.to_csv(output_path, index=False)
        return forecast

    # Prophet training with failure fallback
    try:
        # Prefer PyStan backend on Windows to avoid some cmdstan issues; fallback if unavailable
        try:
            model = pf.Prophet(stan_backend=StanBackendEnum.PYSTAN)
        except Exception:
            model = pf.Prophet()
        model.fit(work)

        # Manually construct future dataframe to avoid internal deprecated Timestamp arithmetic
        if effective_horizon > 0:
            future_ds = _infer_future_index(work["ds"].max(), effective_horizon, work["ds"])
        else:
            future_ds = []
        future_full = pd.DataFrame({"ds": list(work["ds"]) + future_ds})

        forecast = model.predict(future_full)
        forecast["model"] = "prophet"
        # Merge original y for historical rows (future rows retain NaN y)
        forecast = forecast.merge(work[["ds", "y"]], on="ds", how="left")
    except Exception:
        if not fallback_on_failure:
            raise
        forecast = _baseline_forecast(work, effective_horizon, baseline_strategy)

    if output_path:
        forecast.to_csv(output_path, index=False)
    return forecast


#
