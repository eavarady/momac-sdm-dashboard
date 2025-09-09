from adapters import csv_adapter as adapter  # placeholder (not yet used)
import pandas as pd
import prophet as pf


def aggregate_duration_series(
    df: pd.DataFrame,
    *,
    freq: str = "D",
    metric: str = "mean",
    status_complete_values: tuple[str, ...] = ("complete",),
    require_positive: bool = True,
    min_points: int = 1,
) -> pd.DataFrame:
    """
    Aggregate completed run durations into a regular time series.

    Inputs:
      df: DataFrame with at least start_time, end_time. Optional 'status'.
      freq: Resample frequency (e.g. 'D' for daily, 'W' for weekly).
      metric: One of {'mean','median','sum','count'} to define y.
      status_complete_values: If 'status' column exists, keep only these values.
      require_positive: Drop non-positive durations (guard against inverted times).
      min_points: Minimum raw completed rows required before aggregation.

    Returns:
      DataFrame with columns:
        ds: period timestamp (period start)
        y: aggregated metric

    Notes:
      - Timestamps are normalized to naive UTC (Prophet requirement).
      - For 'count', y = number of completed runs per period.
      - For other metrics, durations are in hours.
    """
    required = {"start_time", "end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[list(required) + (["status"] if "status" in df.columns else [])].copy()

    # Parse datetimes (force UTC awareness, then strip later)
    for col in ("start_time", "end_time"):
        work[col] = pd.to_datetime(work[col], errors="coerce", utc=True)

    work = work.dropna(subset=["start_time", "end_time"])
    if "status" in work.columns:
        work = work[work["status"].isin(status_complete_values)]

    if work.empty:
        raise ValueError("No completed rows after filtering & datetime parsing")

    # Compute duration (hours)
    work["duration_h"] = (
        work["end_time"] - work["start_time"]
    ).dt.total_seconds() / 3600.0

    if require_positive:
        work = work[work["duration_h"] > 0]

    if len(work) < min_points:
        raise ValueError(
            f"Insufficient completed rows for aggregation (have {len(work)}, need >= {min_points})"
        )

    # Choose aggregation target
    agg_map: dict[str, str] = {
        "mean": "mean",
        "median": "median",
        "sum": "sum",
        "count": "count",
    }
    if metric not in agg_map:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(agg_map)}")

    # Build a base series with end_time as event timestamp
    events = work[["end_time", "duration_h"]].rename(columns={"end_time": "timestamp"})

    # Resample
    events = events.set_index("timestamp").sort_index()

    # For count, ignore duration values
    if metric == "count":
        agg_series = events["duration_h"].resample(freq).count()
    else:
        agg_series = getattr(events["duration_h"].resample(freq), agg_map[metric])()

    # Drop empty periods (NaN)
    agg_series = agg_series.dropna()

    if agg_series.empty:
        raise ValueError("Aggregation produced an empty series")

    # Prepare output DataFrame; convert index to naive UTC timestamps
    ds = agg_series.index
    if getattr(ds, "tz", None) is not None:
        ds = ds.tz_convert("UTC").tz_localize(None)

    out = pd.DataFrame({"ds": ds, "y": agg_series.values})

    # Enforce numeric y
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["y"])

    if out.empty:
        raise ValueError("Resulting aggregated series is empty after numeric coercion")

    return out


def _infer_future_index(last_ds: pd.Timestamp, horizon: int, work_ds: pd.Series):
    freq = pd.infer_freq(work_ds)
    if freq is None:
        diffs = work_ds.diff().dropna()
        if not diffs.empty:
            freq = diffs.median()  # Timedelta
        else:
            freq = pd.Timedelta(days=1)
    if isinstance(freq, pd.Timedelta):
        future_ds = [last_ds + (i + 1) * freq for i in range(horizon)]
    else:
        future_idx = pd.date_range(start=last_ds, periods=horizon + 1, freq=freq)[1:]
        future_ds = list(future_idx)
    return future_ds


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
    forecast = pd.DataFrame(
        {
            "ds": combined_ds,
            "yhat": combined_yhat,
            "yhat_lower": combined_yhat,
            "yhat_upper": combined_yhat,
            "model": label,
        }
    ).merge(work[["ds", "y"]], on="ds", how="left")
    return forecast


def time_series_forecast(
    df: pd.DataFrame,
    *,
    horizon: int = 365,
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
    output_path: str = "time_series_forecasted_data.csv",
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

    # Pre-fit baseline decision
    if len(work) < min_prophet_points:
        forecast = _baseline_forecast(work, horizon, baseline_strategy)
        forecast.to_csv(output_path, index=False)
        return forecast

    # Adaptive horizon reduction
    if adapt_horizon:
        span_days = max(1, (work["ds"].max() - work["ds"].min()).days + 1)
        max_allowed = max(1, int(span_days * horizon_multiplier))
        effective_horizon = min(horizon, max_allowed)
    else:
        effective_horizon = horizon

    # Prophet training with failure fallback
    try:
        model = pf.Prophet()  # (Could simplify config later if needed)
        model.fit(work)
        future = model.make_future_dataframe(periods=effective_horizon)
        forecast = model.predict(future)
        forecast["model"] = "prophet"
        # Merge original y for historical rows
        forecast = forecast.merge(work[["ds", "y"]], on="ds", how="left")
    except Exception:
        if not fallback_on_failure:
            raise
        forecast = _baseline_forecast(work, effective_horizon, baseline_strategy)

    forecast.to_csv(output_path, index=False)
    return forecast


#
