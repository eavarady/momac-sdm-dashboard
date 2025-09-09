import pandas as pd
from ml.time_series import time_series_forecast
from visualizations.line_chart import build_forecast_line
import traceback

# Simple script to load production_log.csv, run forecasting, and build a Plotly figure.

PROD_PATH = "data/production_log.csv"
print(f"Reading {PROD_PATH}...")
try:
    df = pd.read_csv(PROD_PATH)
except FileNotFoundError:
    print("production_log.csv not found.")
    raise
print("Loaded rows:", len(df))
print("Columns:", df.columns.tolist())

fc = time_series_forecast(df, horizon=30, output_path="__debug_fc.csv")
print("Forecast rows:", len(fc))
print("Forecast columns:", fc.columns.tolist())
print("dtypes:\n", fc.dtypes)
print("Head ds:", fc["ds"].head().tolist())
print("Tail ds:", fc["ds"].tail().tolist())

# Ensure ds is datetime64
if not pd.api.types.is_datetime64_any_dtype(fc["ds"]):
    try:
        fc["ds"] = pd.to_datetime(fc["ds"], errors="coerce")
        print("Coerced ds to datetime. dtype now:", fc["ds"].dtype)
    except Exception as e:
        print("Failed to coerce ds:", e)

print("\nAttempting to build Plotly figure...")
try:
    fig = build_forecast_line(fc)
    print("Plotly figure built successfully. Traces:", len(fig.data))
except Exception as e:
    print("Plot build failed:", e)
    print("Stacktrace:")
    traceback.print_exc()
