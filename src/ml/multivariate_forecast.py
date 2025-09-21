"""Multivariate regression forecasting with scenario support.

Core flow:
    1. Aggregate production log into target series (ds, y).
    2. Build historical feature matrix WITHOUT applying scenario assumptions.
    3. Validate each selected feature for adequacy (enough data & variation).
         If ANY selected feature is inadequate -> raise ForecastFeatureAdequacyError.
         (User must unselect that feature before proceeding.)
    4. Build future feature rows starting from last historic values, THEN apply
         user scenario assumptions ONLY to future periods.
    5. Fit Ridge (or LinearRegression if only one numeric feature) and forecast.

Adequacy rules (numeric): non_null >= 3, distinct >= 2, variance > 0 (eps 1e-9).
Categorical (shift_type): non_null >= 3, distinct >= 2.

Temporal feature engineering:
    * defect_rate_pct: per-period (fails/total * 100) using timestamps if present.

Scenario assumptions NEVER overwrite history; they only shape future rows.

If user selects zero features: we raise an error (UI catches and prompts user to select a variable). No silent mean fallback.


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LinearRegression

from .ts_utils import (
    aggregate_duration_series,
    _infer_future_index,
    build_forecast_frame,
)


SUPPORTED_FEATURES = {
    "defect_rate_pct",
}

MIN_NON_NULL = 3
MIN_DISTINCT = 2
VAR_EPS = 1e-9
COEF_INFLUENCE_EPS = 0.05  # threshold on standardized coef magnitude to warn
MIN_TRAIN_FOR_INFLUENCE = 10  # require at least this many rows to assess


class ForecastFeatureAdequacyError(Exception):
    """Raised when one or more selected features are inadequate for modeling.

    Attributes
    ----------
    details : dict
        Mapping feature -> reason string.
    """

    def __init__(self, details: Dict[str, str]):
        self.details = details
        msg = "Inadequate feature(s): " + ", ".join(f"{k} ({v})" for k, v in details.items())
        super().__init__(msg)


@dataclass
class MVConfig:
    agg_freq: str = "D"
    agg_metric: str = "mean"
    require_non_negative: bool = True
    min_rows: int = 5  # need at least this many periods after aggregation
    adapt_horizon: bool = True
    horizon_multiplier: float = 1.0
    model_label: str = "mv-ridge"


def _safe_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristic: return first column name that looks like a timestamp."""
    candidates = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ("time", "date", "ts", "timestamp"))
    ]
    for c in candidates:
        if pd.api.types.is_datetime64_any_dtype(df[c]) or pd.to_datetime(
            df[c], errors="coerce"
        ).notna().any():
            return c
    return None


def _static_or_resampled(df: pd.DataFrame, value_series: pd.Series, freq: str, ds_index: pd.DatetimeIndex) -> pd.Series:
    """If df has a timestamp column, resample value_series to freq over ds_index; else repeat static value."""
    ts_col = _find_timestamp_column(df)
    if ts_col is None:
        # static: repeat mean (or single) value across index
        if value_series.empty:
            return pd.Series([np.nan] * len(ds_index), index=ds_index)
        val = float(value_series.mean())
        return pd.Series([val] * len(ds_index), index=ds_index)
    tmp = df.copy()
    tmp[ts_col] = _safe_dt(tmp[ts_col])
    tmp = tmp.dropna(subset=[ts_col])
    if tmp.empty:
        val = float(value_series.mean()) if not value_series.empty else np.nan
        return pd.Series([val] * len(ds_index), index=ds_index)
    # Build a full period index covering ds_index span
    start, end = ds_index.min(), ds_index.max()
    full_range = pd.date_range(start=start, end=end, freq=freq)
    if full_range.empty:
        full_range = ds_index
    tmp = tmp.set_index(ts_col)
    # numeric encode value_series if categorical? For now assume already numeric / precomputed ratio.
    if value_series.name not in tmp.columns:
        # Provide a constant numeric value
        val = float(value_series.mean()) if not value_series.empty else np.nan
        return pd.Series([val] * len(ds_index), index=ds_index)
    res = tmp[value_series.name].resample(freq).mean()  # generic mean
    # Align to ds_index
    res = res.reindex(full_range).interpolate(limit_direction="both")
    return res.reindex(ds_index)


def _compute_feature_series(
    tables: Dict[str, pd.DataFrame],
    ds_index: pd.DatetimeIndex,
    freq: str,
    included: Sequence[str],
) -> pd.DataFrame:
    """Build raw historical feature series (no scenario overrides)."""
    feats: Dict[str, pd.Series] = {}

    # (operator_count, shift_type removed until temporal sources exist)

    # Defect rate % (temporal if timestamps exist)
    if "defect_rate_pct" in included:
        qc = tables.get("quality_checks", pd.DataFrame())
        if not qc.empty and "result" in qc.columns:
            ts_col = _find_timestamp_column(qc)
            if ts_col:
                qc = qc.copy()
                qc[ts_col] = _safe_dt(qc[ts_col])
                qc = qc.dropna(subset=[ts_col])
                if not qc.empty:
                    qc.set_index(ts_col, inplace=True)
                    # Normalize to tz-naive to match target ds_index (which is naive)
                    if getattr(qc.index, "tz", None) is not None:
                        qc.index = qc.index.tz_convert("UTC").tz_localize(None)
                    qc["is_fail"] = qc["result"].astype(str).str.lower().eq("fail")
                    grouped = qc.resample(freq).agg(total=("result", "count"), fails=("is_fail", "sum"))
                    grouped["defect_rate_pct"] = (grouped["fails"] / grouped["total"].clip(lower=1)) * 100.0
                    # Align to ds_index (ensure ds_index is DatetimeIndex and naive)
                    target_index = pd.DatetimeIndex(ds_index)
                    rate = grouped["defect_rate_pct"].reindex(target_index).astype(float)
                    rate = rate.ffill().bfill().infer_objects(copy=False)
                    feats["defect_rate_pct"] = rate
                else:
                    feats["defect_rate_pct"] = pd.Series([np.nan] * len(ds_index), index=ds_index)
            else:
                # No timestamp -> constant snapshot
                total = len(qc)
                fails = qc["result"].astype(str).str.lower().eq("fail").sum()
                defect_rate = (fails / max(1, total)) * 100.0
                feats["defect_rate_pct"] = pd.Series([defect_rate] * len(ds_index), index=ds_index)
        else:
            feats["defect_rate_pct"] = pd.Series([np.nan] * len(ds_index), index=ds_index)

    if not feats:
        return pd.DataFrame(index=ds_index)
    return pd.DataFrame(feats)


def _evaluate_feature_adequacy(feat_hist: pd.DataFrame, categorical: Sequence[str]) -> Tuple[bool, Dict[str, str], Dict[str, Dict[str, float]]]:
    """Evaluate adequacy for each feature.

    Returns
    -------
    ok : bool
        True if all features adequate.
    failures : dict
        feature -> reason.
    stats : dict
        feature -> {non_null, distinct, variance} (variance NaN for categorical).
    """
    failures: Dict[str, str] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for col in feat_hist.columns:
        series = feat_hist[col]
        non_null = int(series.notna().sum())
        distinct = int(series.dropna().nunique())
        if col in categorical:
            variance = float("nan")
            if non_null < MIN_NON_NULL:
                failures[col] = f"non_null={non_null} (<{MIN_NON_NULL})"
            elif distinct < MIN_DISTINCT:
                failures[col] = f"distinct={distinct} (<{MIN_DISTINCT})"
        else:
            # Numeric
            variance = float(series.astype(float).var(ddof=1)) if non_null > 1 else 0.0
            if non_null < MIN_NON_NULL:
                failures[col] = f"non_null={non_null} (<{MIN_NON_NULL})"
            elif distinct < MIN_DISTINCT:
                failures[col] = f"distinct={distinct} (<{MIN_DISTINCT})"
            elif abs(variance) <= VAR_EPS:
                failures[col] = f"variance≈0"
        stats[col] = {"non_null": float(non_null), "distinct": float(distinct), "variance": variance}
    return len(failures) == 0, failures, stats


def run_multivariate_forecast(
    tables: Dict[str, pd.DataFrame],
    scenario: Dict,
    *,
    output_path: str | None = None,
    config: MVConfig | None = None,
) -> pd.DataFrame:
    """Execute multivariate regression forecast based on scenario.

    Parameters
    ----------
    tables : dict[str, DataFrame]
        Raw loaded tables (production_log required for target derivation).
    scenario : dict
        Scenario dict from UI containing horizon, adapt_horizon, horizon_multiplier,
        aggregation {freq, metric}, included_variables, assumptions.
    output_path : str
        CSV path for dashboard consumption.
    config : MVConfig
        Optional override config.
    """
    if config is None:
        config = MVConfig(
            agg_freq=scenario["aggregation"]["freq"],
            agg_metric=scenario["aggregation"]["metric"],
            adapt_horizon=scenario.get("adapt_horizon", True),
            horizon_multiplier=scenario.get("horizon_multiplier", 1.0),
        )

    horizon = int(scenario.get("horizon", 30))
    included = [v for v in scenario.get("included_variables", []) if v in SUPPORTED_FEATURES]
    assumptions = scenario.get("assumptions", {})

    prod_log = tables.get("production_log", pd.DataFrame())
    if prod_log.empty:
        raise ValueError("production_log table required for multivariate forecast")

    # Build target series (ds, y)
    target = aggregate_duration_series(
        prod_log,
        freq=config.agg_freq,
        metric=config.agg_metric,
    )
    target = target.sort_values("ds").drop_duplicates("ds")
    if len(target) < config.min_rows:
        raise ValueError(f"Not enough aggregated periods (have {len(target)}, need >= {config.min_rows})")

    ds_hist = target["ds"].to_list()

    # Adaptive horizon
    if config.adapt_horizon:
        span_days = max(1, (target["ds"].max() - target["ds"].min()).days + 1)
        max_allowed = max(1, int(span_days * config.horizon_multiplier))
        eff_horizon = min(horizon, max_allowed)
    else:
        eff_horizon = horizon

    # If user selected zero features -> explicit error (UI will display guidance)
    if not included:
        raise ValueError("No scenario variables selected; select at least one scenario variable to run a multivariate scenario.")

    # Feature matrix for history (no scenario overrides)
    feat_hist = _compute_feature_series(tables, target["ds"], config.agg_freq, included)

    # If user selected features, enforce adequacy; else allow baseline later.
    if included:
        categorical: List[str] = []  # currently no categorical features active
        ok, failures, _stats = _evaluate_feature_adequacy(
            feat_hist[[c for c in feat_hist.columns if c in included]], categorical
        )
        if not ok:
            raise ForecastFeatureAdequacyError(failures)

    # Build future ds
    if eff_horizon > 0:
        future_ds = _infer_future_index(target["ds"].max(), eff_horizon, target["ds"])
    else:
        future_ds = []

    # Future features: replicate last historic then apply assumptions
    if future_ds:
        future_index = pd.DatetimeIndex(future_ds)
        if included and not feat_hist.empty:
            last_row = feat_hist.iloc[-1]
            data = {}
            for col in included:
                base_val = last_row[col] if col in last_row else np.nan
                override_val = assumptions.get(col, base_val)
                data[col] = [override_val] * len(future_index)
            feat_future = pd.DataFrame(data, index=future_index)
        else:
            feat_future = pd.DataFrame(index=future_index)
    else:
        feat_future = pd.DataFrame()

    # Combine features
    feature_full = pd.concat([feat_hist, feat_future]) if not feat_future.empty else feat_hist.copy()
    feature_full.index = pd.to_datetime(feature_full.index, utc=False)

    # Merge target y into combined frame
    merged = pd.DataFrame({"ds": pd.to_datetime(ds_hist + future_ds, utc=False)})
    merged = merged.merge(target, on="ds", how="left")  # y present only for history
    # Align features on ds
    feature_full["ds"] = feature_full.index
    merged = merged.merge(feature_full, on="ds", how="left")
    # Ensure chronological order then add explicit time index feature 't'
    merged = merged.sort_values("ds").reset_index(drop=True)
    merged["t"] = range(len(merged))

    # Determine feature columns: always include time index plus selected scenario features that are present
    feature_cols = ["t"] + [c for c in included if c in merged.columns]

    # Separate historical rows for fitting
    hist_mask = merged["y"].notna()
    train_df = merged.loc[hist_mask].copy()

    # Identify categorical / numeric
    categorical = []
    numeric = [c for c in feature_cols if c not in categorical]

    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    # Choose estimator: if multiple non-time driver features, use Ridge for stability.
    driver_features = [c for c in feature_cols if c != "t"]
    if len(driver_features) > 1:
        estimator = Ridge(alpha=1.0, random_state=42)
        model_label = "mv-ridge"
    else:
        estimator = LinearRegression()
        model_label = "mv-linear"

    pipe = Pipeline([
        ("prep", preprocessor),
        ("est", estimator),
    ])

    X_train = train_df[feature_cols]
    y_train = train_df["y"].astype(float)

    pipe.fit(X_train, y_train)
    # In-sample predictions (smoothed fit)
    yhat_hist = pipe.predict(X_train).tolist()

    # Collect standardized coefficients for non-time drivers (for user feedback)
    try:
        est = pipe.named_steps.get("est")
        coefs = getattr(est, "coef_", None)
        coef_map: Dict[str, float] = {}
        low_influence: List[str] = []
        if coefs is not None and len(X_train.columns) == len(coefs):
            coef_map = {name: float(coef) for name, coef in zip(X_train.columns, coefs)}
            # Consider only non-time drivers
            driver_features = [c for c in X_train.columns if c != "t"]
            if len(y_train) >= MIN_TRAIN_FOR_INFLUENCE and driver_features:
                low_influence = [
                    f for f in driver_features
                    if abs(coef_map.get(f, 0.0)) < COEF_INFLUENCE_EPS
                ]
        influence_meta = {
            "estimator": model_label,
            "n_train": int(len(y_train)),
            "standardized_coefficients": {k: v for k, v in coef_map.items() if k != "t"},
            "low_influence": low_influence,
            "threshold": COEF_INFLUENCE_EPS,
        }
    except Exception:
        influence_meta = None

    # Residual stddev for intervals
    if len(y_train) > 2:
        resid = y_train - yhat_hist
        sigma = float(pd.Series(resid).std(ddof=1))
    else:
        sigma = 0.0

    # Future predictions
    future_mask = merged["y"].isna()
    if future_mask.any():
        X_future = merged.loc[future_mask, feature_cols]
        yhat_future = pipe.predict(X_future).tolist()
    else:
        yhat_future = []

    combined_yhat = yhat_hist + yhat_future
    combined_ds = merged["ds"].tolist()
    # Assemble intervals
    if sigma > 0:
        yhat_arr = np.array(combined_yhat)
        intervals = {
            "yhat_lower": yhat_arr - 1.96 * sigma,
            "yhat_upper": yhat_arr + 1.96 * sigma,
        }
    else:
        intervals = None

    y_full = merged["y"].tolist()
    forecast = build_forecast_frame(
        combined_ds,
        combined_yhat,
        model_label=model_label,
        y=y_full,
        intervals=intervals,
    )
    if config.require_non_negative:
        forecast["yhat"] = forecast["yhat"].clip(lower=0.0)
        if intervals is not None:
            forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0.0)

    if output_path:
        forecast.to_csv(output_path, index=False)
        # Write sidecar meta JSON with influence diagnostics (if available)
        if influence_meta is not None:
            try:
                import json, os
                meta_path = os.path.splitext(output_path)[0] + "_meta.json"
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump(influence_meta, fh, indent=2)
            except Exception:
                pass
    return forecast


__all__ = ["run_multivariate_forecast", "ForecastFeatureAdequacyError"]
