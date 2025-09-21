import os
import pandas as pd
import pytest

from ml.time_series import time_series_forecast

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "production_log.csv"
)


@pytest.mark.fast
def test_time_series_forecast_basic(tmp_path):
    """Smoke test the forecasting pipeline:
    - Loads production_log.csv (skip if absent)
    - Runs forecasting with a modest horizon (adaptive reduction allowed)
    - Asserts required schema columns
    - Ensures at least one future row (y is NaN) exists
    - Validates model label is one of expected set
    """
    if not os.path.exists(DATA_PATH):
        pytest.skip("production_log.csv missing; skipping forecast smoke test")

    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Input production_log.csv unexpectedly empty"

    fc = time_series_forecast(
        df,
        horizon=30,
        baseline_strategy="mean",
        adapt_horizon=True,
        output_path=str(tmp_path / "__test_fc.csv"),
    )

    # Non-empty
    assert not fc.empty, "Forecast output is empty"

    # Required columns
    required = {"ds", "yhat", "yhat_lower", "yhat_upper", "model"}
    missing = required - set(fc.columns)
    assert not missing, f"Missing forecast columns: {missing}"

    # ds must be datetime convertible
    assert (
        pd.api.types.is_datetime64_any_dtype(fc["ds"])
        or not pd.to_datetime(fc["ds"], errors="coerce").isna().all()
    ), "'ds' column not datetime or convertible"

    # Numeric prediction columns
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        assert pd.api.types.is_numeric_dtype(fc[col]), f"{col} not numeric dtype"
        assert fc[col].notna().any(), f"{col} all NaN"

    # Model value sanity
    allowed_models = {"prophet", "baseline-mean", "baseline-linear"}
    models_found = set(fc["model"].dropna().unique())
    assert (
        models_found <= allowed_models
    ), f"Unexpected model labels: {models_found - allowed_models}"

    # Ensure at least one future row was produced (where y is NaN)
    future_na = fc["y"].isna().sum()
    assert future_na >= 1, "No future rows detected (y NaN count = 0)"

    # Length should exceed historical y count
    hist_len = fc["y"].notna().sum()
    assert len(fc) > hist_len, "Forecast contains no extension beyond history"

    # Optional: quick monotonic ds check
    assert fc["ds"].is_monotonic_increasing, "ds not sorted ascending"
