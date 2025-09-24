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
    "shift_night_share",
    "avg_energy_consumption",
    "wip_proxy",
    "inspection_intensity",
}

# Friendly label -> internal feature key mapping (single source of truth)
FEATURE_LABELS = {
    "Defect rate %": "defect_rate_pct",
    "Night shift share": "shift_night_share",
    "Avg energy consumption": "avg_energy_consumption",
    "WIP proxy (fraction)": "wip_proxy",
    "Inspection coverage": "inspection_intensity",
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

    # shift_night_share: fraction of production events that occurred on 'night' shift
    if "shift_night_share" in included:
        pl = tables.get("production_log", pd.DataFrame())
        lines = tables.get("production_lines", pd.DataFrame())
        if (
            not pl.empty
            and "line_id" in pl.columns
            and not lines.empty
            and {"line_id", "shift"}.issubset(lines.columns)
        ):
            ts_col_pl = _find_timestamp_column(pl)
            if ts_col_pl:
                tmp = pl.copy()
                tmp[ts_col_pl] = _safe_dt(tmp[ts_col_pl])
                tmp = tmp.dropna(subset=[ts_col_pl])
                if not tmp.empty:
                    tmp = tmp.merge(
                        lines[["line_id", "shift"]].copy(), on="line_id", how="left"
                    )
                    tmp["shift_norm"] = tmp["shift"].astype(str).str.lower()
                    tmp.set_index(ts_col_pl, inplace=True)
                    if getattr(tmp.index, "tz", None) is not None:
                        tmp.index = tmp.index.tz_convert("UTC").tz_localize(None)
                    grp = tmp.resample(freq).agg(
                        total=("shift_norm", "count"),
                        night=("shift_norm", lambda s: (s == "night").sum()),
                    )
                    grp["shift_night_share"] = grp["night"] / grp["total"].clip(lower=1)
                    series = grp["shift_night_share"].reindex(pd.DatetimeIndex(ds_index))
                    series = series.ffill().bfill()
                    feats["shift_night_share"] = series.astype(float)
                else:
                    feats["shift_night_share"] = pd.Series([np.nan] * len(ds_index), index=ds_index)
            else:
                feats["shift_night_share"] = pd.Series([np.nan] * len(ds_index), index=ds_index)
        else:
            feats["shift_night_share"] = pd.Series([np.nan] * len(ds_index), index=ds_index)

    # avg_energy_consumption: mean energy_consumption metric per period across all machines
    if "avg_energy_consumption" in included:
        mm = tables.get("machine_metrics", pd.DataFrame())
        if not mm.empty and {"metric_type", "metric_value"}.issubset(mm.columns):
            ts_col_mm = _find_timestamp_column(mm)
            if ts_col_mm:
                tmpm = mm[mm["metric_type"].astype(str).str.lower() == "energy_consumption"].copy()
                if not tmpm.empty:
                    tmpm[ts_col_mm] = _safe_dt(tmpm[ts_col_mm])
                    tmpm = tmpm.dropna(subset=[ts_col_mm])
                    if not tmpm.empty:
                        tmpm.set_index(ts_col_mm, inplace=True)
                        if getattr(tmpm.index, "tz", None) is not None:
                            tmpm.index = tmpm.index.tz_convert("UTC").tz_localize(None)
                        grp = tmpm["metric_value"].resample(freq).mean()
                        series = grp.reindex(pd.DatetimeIndex(ds_index)).astype(float)
                        series = series.ffill().bfill()
                        feats["avg_energy_consumption"] = series
                    else:
                        feats["avg_energy_consumption"] = pd.Series([np.nan] * len(ds_index), index=ds_index)
                else:
                    feats["avg_energy_consumption"] = pd.Series([np.nan] * len(ds_index), index=ds_index)
            else:
                feats["avg_energy_consumption"] = pd.Series([np.nan] * len(ds_index), index=ds_index)
        else:
            feats["avg_energy_consumption"] = pd.Series([np.nan] * len(ds_index), index=ds_index)

    # wip_proxy: fraction of in-progress events during the period (in_progress / total)
    if "wip_proxy" in included:
        pl2 = tables.get("production_log", pd.DataFrame())
        if not pl2.empty and {"status"}.issubset(pl2.columns):
            ts_col_pl2 = _find_timestamp_column(pl2)
            if ts_col_pl2:
                tmp2 = pl2.copy()
                tmp2[ts_col_pl2] = _safe_dt(tmp2[ts_col_pl2])
                tmp2 = tmp2.dropna(subset=[ts_col_pl2])
                if not tmp2.empty:
                    tmp2["_status_norm"] = tmp2["status"].astype(str).str.lower()
                    tmp2.set_index(ts_col_pl2, inplace=True)
                    if getattr(tmp2.index, "tz", None) is not None:
                        tmp2.index = tmp2.index.tz_convert("UTC").tz_localize(None)
                    grp2 = tmp2.resample(freq).agg(
                        total=("_status_norm", "count"),
                        inprog=("_status_norm", lambda s: (s == "in_progress").sum()),
                    )
                    grp2["wip_proxy"] = grp2["inprog"] / grp2["total"].clip(lower=1)
                    series = grp2["wip_proxy"].reindex(pd.DatetimeIndex(ds_index))
                    series = series.fillna(0.0)
                    feats["wip_proxy"] = series.astype(float)
                else:
                    feats["wip_proxy"] = pd.Series([0.0] * len(ds_index), index=ds_index)
            else:
                feats["wip_proxy"] = pd.Series([0.0] * len(ds_index), index=ds_index)
        else:
            feats["wip_proxy"] = pd.Series([np.nan] * len(ds_index), index=ds_index)

    # inspection_intensity: inspection coverage = inspected_units / produced_units (0..1)
    if "inspection_intensity" in included:
        qc = tables.get("quality_checks", pd.DataFrame())
        pl_prod = tables.get("production_log", pd.DataFrame())
        ts_qc = _find_timestamp_column(qc) if not qc.empty else None
        ts_pl = _find_timestamp_column(pl_prod) if not pl_prod.empty else None
        if qc is not None and not qc.empty and ts_qc and pl_prod is not None and not pl_prod.empty and ts_pl:
            qcc = qc.copy()
            qcc[ts_qc] = _safe_dt(qcc[ts_qc])
            qcc = qcc.dropna(subset=[ts_qc])
            plc = pl_prod.copy()
            plc[ts_pl] = _safe_dt(plc[ts_pl])
            plc = plc.dropna(subset=[ts_pl])
            if not qcc.empty and not plc.empty:
                # Set indices
                qcc.set_index(ts_qc, inplace=True)
                plc.set_index(ts_pl, inplace=True)
                if getattr(qcc.index, "tz", None) is not None:
                    qcc.index = qcc.index.tz_convert("UTC").tz_localize(None)
                if getattr(plc.index, "tz", None) is not None:
                    plc.index = plc.index.tz_convert("UTC").tz_localize(None)
                # Determine inspected units: prefer unit_id, else run_id, else row count
                inspected_col = None
                for c in ("unit_id", "run_id"):
                    if c in qcc.columns:
                        inspected_col = c
                        break
                if inspected_col:
                    qc_grp = qcc.resample(freq).agg(inspected=(inspected_col, "nunique"))
                else:
                    qc_grp = qcc.resample(freq).agg(inspected=(qcc.columns[0], "count"))
                # Produced units: prefer completed events if status present, else total events
                if "status" in plc.columns:
                    plc["_status_norm"] = plc["status"].astype(str).str.lower()
                    plc["__complete_flag"] = (plc["_status_norm"] == "complete").astype(int)
                    prod_grp = plc.resample(freq).agg(produced=("__complete_flag", "sum"))
                else:
                    # count all rows
                    prod_grp = plc.resample(freq).size().to_frame(name="produced")
                coverage = qc_grp.join(prod_grp, how="outer")
                # Fill missing counts with 0 then compute coverage; when produced==0 define coverage=0 (avoid NaN propagation)
                coverage["inspected"] = coverage["inspected"].fillna(0).astype(float)
                coverage["produced"] = coverage["produced"].fillna(0).astype(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    cov_vals = np.where(
                        coverage["produced"] > 0,
                        coverage["inspected"] / coverage["produced"],
                        0.0,
                    )
                coverage["inspection_intensity"] = cov_vals
                # Clip to [0,1] just in case inspected > produced due to counting differences
                coverage["inspection_intensity"] = coverage["inspection_intensity"].clip(lower=0.0, upper=1.0)
                series = coverage["inspection_intensity"].reindex(pd.DatetimeIndex(ds_index)).fillna(0.0)
                feats["inspection_intensity"] = series.astype(float)
            else:
                feats["inspection_intensity"] = pd.Series([np.nan] * len(ds_index), index=ds_index)
        else:
            feats["inspection_intensity"] = pd.Series([np.nan] * len(ds_index), index=ds_index)

    if not feats:
        return pd.DataFrame(index=ds_index)
    return pd.DataFrame(feats)


def compute_feature_defaults(tables: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Compute baseline default values for each supported feature from raw tables.

    Returns
    -------
    dict
        Mapping internal feature key -> default numeric value.
    """
    defaults: Dict[str, float] = {}

    # Defect rate % overall snapshot (fails / total * 100)
    qc = tables.get("quality_checks", pd.DataFrame())
    if not qc.empty and "result" in qc.columns:
        total_q = len(qc)
        fails = qc["result"].astype(str).str.lower().eq("fail").sum()
        defaults["defect_rate_pct"] = (fails / max(1, total_q)) * 100.0
    else:
        defaults["defect_rate_pct"] = 5.0  # fallback heuristic

    # Night shift share (overall fraction of events on night shift)
    pl = tables.get("production_log", pd.DataFrame())
    lines = tables.get("production_lines", pd.DataFrame())
    if (
        not pl.empty
        and "line_id" in pl.columns
        and not lines.empty
        and {"line_id", "shift"}.issubset(lines.columns)
    ):
        ts_col = _find_timestamp_column(pl)
        tmp = pl.copy()
        if ts_col and ts_col in tmp.columns:
            tmp[ts_col] = _safe_dt(tmp[ts_col])
        if "line_id" in tmp.columns:
            tmp = tmp.merge(lines[["line_id", "shift"]].copy(), on="line_id", how="left")
        if not tmp.empty and "shift" in tmp.columns:
            shift_norm = tmp["shift"].astype(str).str.lower()
            defaults["shift_night_share"] = float((shift_norm == "night").sum() / max(1, len(shift_norm)))
        else:
            defaults["shift_night_share"] = 0.0
    else:
        defaults["shift_night_share"] = 0.0

    # Avg energy consumption (mean of energy_consumption metric)
    mm = tables.get("machine_metrics", pd.DataFrame())
    if not mm.empty and {"metric_type", "metric_value"}.issubset(mm.columns):
        energy_vals = mm[mm["metric_type"].astype(str).str.lower() == "energy_consumption"]["metric_value"].astype(float)
        defaults["avg_energy_consumption"] = float(round(energy_vals.mean(), 2)) if not energy_vals.empty else 0.0
    else:
        defaults["avg_energy_consumption"] = 0.0

    # WIP proxy overall fraction of in_progress events
    if not pl.empty and {"status"}.issubset(pl.columns):
        status_norm = pl["status"].astype(str).str.lower()
        defaults["wip_proxy"] = float(round((status_norm == "in_progress").sum() / max(1, len(status_norm)), 2))
    else:
        defaults["wip_proxy"] = 0.0

    # Inspection coverage overall snapshot: inspected unique units (unit_id/run_id) / completed events
    qc2 = qc
    pl2 = pl
    if (
        not qc2.empty
        and not pl2.empty
        and "status" in pl2.columns
    ):
        inspected_col = None
        for c in ("unit_id", "run_id"):
            if c in qc2.columns:
                inspected_col = c
                break
        if inspected_col:
            inspected_units = qc2[inspected_col].nunique()
        else:
            inspected_units = len(qc2)
        completed = pl2["status"].astype(str).str.lower().eq("complete").sum()
        if completed > 0:
            defaults["inspection_intensity"] = float(round(inspected_units / max(1, completed), 2))
        else:
            defaults["inspection_intensity"] = 0.0
    else:
        defaults["inspection_intensity"] = 0.0

    return defaults


def validate_and_normalize_scenario(
    scenario: Dict,
    tables: Dict[str, pd.DataFrame],
) -> Dict:
    """Validate and normalize scenario dict.

    Ensures included_variables are supported, assumptions are floats, and returns a cleaned copy.
    Does NOT fill in missing assumptions (UI supplies them), but could in future.
    """
    cleaned = dict(scenario)  # shallow copy
    included = [v for v in cleaned.get("included_variables", []) if v in SUPPORTED_FEATURES]
    cleaned["included_variables"] = included
    # Cast assumptions to float when possible
    assumptions = cleaned.get("assumptions", {}) or {}
    norm_assumptions: Dict[str, float] = {}
    for k, v in assumptions.items():
        if k in included:
            try:
                norm_assumptions[k] = float(v)
            except Exception:
                # keep original if non-castable; model layer may raise later
                norm_assumptions[k] = v
    cleaned["assumptions"] = norm_assumptions
    if not included:
        # preserve original behavior: raise upstream later; we keep here so run_multivariate_forecast can raise a clear error
        pass
    return cleaned


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
    return_meta: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Optional[Dict]]:
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

    # Normalize scenario first
    scenario = validate_and_normalize_scenario(scenario, tables)

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

    # Historical variation diagnostics (non-fatal): detect near-constant features that could yield abrupt forecast steps
    hist_var_stats: Dict[str, Dict[str, float]] = {}
    hist_low_variation: List[str] = []
    if not feat_hist.empty:
        for col in feat_hist.columns:
            series = feat_hist[col].dropna().astype(float)
            if series.empty:
                continue
            n_unique = int(series.nunique())
            std = float(series.std(ddof=0)) if len(series) > 1 else 0.0
            vmin = float(series.min())
            vmax = float(series.max())
            value_range = vmax - vmin
            is_fraction_scale = series.between(0.0, 1.0).all()
            is_percent_feature = col == "defect_rate_pct"
            # Thresholds (kept modest to avoid false positives):
            #  - Fractions: std < 0.01 OR n_unique <= 2
            #  - Percent feature: std < 0.5 OR n_unique <= 2
            trigger = False
            if is_fraction_scale:
                if std < 0.01 or n_unique <= 2:
                    trigger = True
            elif is_percent_feature:
                if std < 0.5 or n_unique <= 2:
                    trigger = True
            # Record stats
            hist_var_stats[col] = {
                "std": std,
                "n_unique": float(n_unique),
                "min": vmin,
                "max": vmax,
                "range": value_range,
            }
            if trigger:
                hist_low_variation.append(col)

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

    # Attach historical variation diagnostics to meta (initialize meta if absent)
    if influence_meta is None:
        influence_meta = {}
    influence_meta["historical_variation"] = hist_var_stats
    influence_meta["historical_low_variation"] = hist_low_variation
    influence_meta["historical_low_variation_thresholds"] = {
        "fraction_std": 0.01,
        "percent_std": 0.5,
        "min_unique_flag": 2,
    }
    # Measurement ambiguity (pairs of features that can overlap in explanatory power)
    ambiguity_pairs = []
    if {"inspection_intensity", "defect_rate_pct"}.issubset(set(included)):
        ambiguity_pairs.append(["inspection_intensity", "defect_rate_pct"])
    if ambiguity_pairs:
        influence_meta["measurement_ambiguity"] = ambiguity_pairs
    influence_meta["thresholds"] = {
        "coef_influence_eps": COEF_INFLUENCE_EPS,
        "min_train_for_influence": MIN_TRAIN_FOR_INFLUENCE,
        "min_non_null": MIN_NON_NULL,
        "min_distinct": MIN_DISTINCT,
    }

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
    if return_meta:
        return forecast, influence_meta
    return forecast


__all__ = [
    "run_multivariate_forecast",
    "ForecastFeatureAdequacyError",
    "compute_feature_defaults",
    "validate_and_normalize_scenario",
    "FEATURE_LABELS",
]
