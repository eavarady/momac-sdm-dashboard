import os

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

import streamlit as st
import pandas as pd
from adapters.csv_adapter import read_csv_tables, get_last_load_stats
from adapters.sheets_adapter import read_sheets
from adapters.excel_adapter import (
    read_excel_tables as read_excel,
    get_last_load_stats as get_excel_stats,
)
from kpi.kpi_calculator import compute_all_kpis
from ml.bottleneck_detector import detect_bottleneck, top_bottlenecks
from visualizations.gantt import GanttChart
from kpi.progress import per_step_progress, overall_progress_by_product
from ml.time_series import time_series_forecast
from visualizations.line_chart import build_forecast_line
from ml.linear_forecast import linear_forecast

st.set_page_config(page_title="MOMAC SDM Dashboard", layout="wide")

st.title("MOMAC SDM Dashboard")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Select source", ["CSV", "Google Sheets", "Excel"], index=0)

    if source == "Google Sheets":
        st.caption("OAuth will prompt in your browser on first use.")
        spreadsheet_id = st.text_input("Spreadsheet ID", placeholder="1AbC...xyz")
        title_map_json = st.text_area(
            'Optional: Title map JSON (e.g. {"production_log": "Prod Log"})',
            value="",
            height=80,
        )
        title_map = None
        if title_map_json.strip():
            try:
                import json

                title_map = json.loads(title_map_json)
            except Exception as je:
                st.warning(f"Could not parse title map JSON: {je}")

    elif source == "Excel":
        st.caption("Upload a workbook or provide a path.")
        uploaded = st.file_uploader("Upload .xlsx", type=["xlsx"])
        excel_path = st.text_input("...or path to .xlsx", value="")
        title_map_json = st.text_area(
            'Optional: Title map JSON (e.g. {"production_log": "Prod Log"})',
            value="",
            height=80,
        )
        skiprows = st.number_input(
            "Rows to skip at top of each sheet", min_value=0, value=0, step=1
        )
        title_map = None
        if title_map_json.strip():
            try:
                import json

                title_map = json.loads(title_map_json)
            except Exception as je:
                st.warning(f"Could not parse title map JSON: {je}")

    else:
        st.caption("Reading from local CSVs in data/")

# Load data (fast-fail): if any table is invalid, surface the error and stop
try:
    if source == "CSV":
        _tables = read_csv_tables()

    elif source == "Google Sheets":
        if not spreadsheet_id:
            st.info("Enter a Spreadsheet ID to load data.")
            st.stop()
        _tables = read_sheets(spreadsheet_id=spreadsheet_id, title_map=title_map)

    else:  # Excel
        if "uploaded" in locals() and uploaded is not None:
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            _tables = read_excel(tmp_path, title_map=title_map, skiprows=int(skiprows))
        elif "excel_path" in locals() and excel_path.strip():
            _tables = read_excel(
                excel_path.strip(), title_map=title_map, skiprows=int(skiprows)
            )
        else:
            st.info("Upload an .xlsx file or provide a path to proceed.")
            st.stop()

except Exception as e:
    st.error(f"Data load failed: {e}")
    # Per-source load stats for troubleshooting
    if source == "CSV":
        stats = get_last_load_stats()
    elif source == "Excel":
        stats = get_excel_stats()
    else:
        stats = {}  # (Sheets adapter can add a stats getter later if needed)
    if stats:
        st.subheader("Last load stats")
        st.json(stats)
    st.stop()

# KPIs
kpis = compute_all_kpis(_tables)
col1, col2, col3, col4, col5, col6 = st.columns(6)

# Show throughput per hour with one decimal

col1.metric("Throughput (units/hr)", f"{kpis.get('throughput') * 3600.0:.5f}")
col2.metric("Throughput (units/day)", f"{(kpis.get('throughput') * 3600.0 * 24):.1f}")
col3.metric(
    "Throughput (units/week)", f"{(kpis.get('throughput') * 3600.0 * 24 * 7):.1f}"
)
col4.metric("WIP (Qty)", f"{kpis.get('wip', 0)}")
# schedule_efficiency displays as an index (e.g., 0.83x, 1.12x)
col5.metric("Schedule Efficiency", f"{(kpis.get('schedule_efficiency') or 0.0):.2f}x")
col6.metric("On-Time Rate", f"{(kpis.get('on_time_rate') or 0.0)*100:.1f}%")

### BOTTLENECK DETECTION AND FORECASTING ###

# HEURISTIC BOTTLENECK DETECTION

bn = detect_bottleneck(
    _tables.get("process_steps", pd.DataFrame()),
    _tables.get("production_log", pd.DataFrame()),
)
# st.subheader("Largest bottleneck")
# st.write(bn or "No bottleneck detected.")
st.subheader("Top 3 Bottlenecks by WIP")
prod = _tables.get("production_log", pd.DataFrame())
top3 = top_bottlenecks(prod, top_n=3)
if top3.empty:
    st.write("No in-progress work detected.")
else:
    st.dataframe(top3, width="stretch")

# BOTTLENECK FORECASTING HEADER
st.subheader("Bottleneck Forecasting")
st.write("Forecasting potential bottlenecks based on historical data.")

with st.expander("Time Series Forecasting", expanded=True):
    df = _tables.get("production_log", pd.DataFrame())
    # Forecast settings
    col_fs1, col_fs2, col_fs3 = st.columns(3)
    with col_fs1:
        user_horizon = st.number_input(
            "Requested Horizon (periods)",
            min_value=1,
            max_value=720,
            value=60,
            step=1,
            help="Number of future periods to forecast before adaptive reduction.",
        )
    with col_fs2:
        adapt = st.checkbox(
            "Adaptive Horizon",
            value=True,
            help="Reduce horizon based on history span * multiplier",
        )
        multiplier = st.number_input(
            "Horizon Multiplier", min_value=0.1, max_value=10.0, value=1.0, step=0.1
        )
    with col_fs3:
        baseline_strategy = st.selectbox(
            "Baseline Strategy", ["mean", "linear"], index=0
        )

    agg_freq = st.selectbox("Aggregation Frequency", ["D", "W", "M"], index=0)
    agg_metric = st.selectbox(
        "Aggregation Metric", ["mean", "median", "sum", "count"], index=0
    )

    if df.empty:
        st.info("No production_log data available for forecasting.")
    else:
        run_btn = st.button("Run Forecast")
        if run_btn:
            try:
                forecast_path = "time_series_forecasted_data.csv"
                time_series_forecast(
                    df,
                    horizon=int(user_horizon),
                    baseline_strategy=baseline_strategy,
                    adapt_horizon=adapt,
                    horizon_multiplier=float(multiplier),
                    agg_freq=agg_freq,
                    agg_metric=agg_metric,
                )
                # Attempt to load and plot
                import os

                if os.path.exists(forecast_path):
                    fc = pd.read_csv(forecast_path, parse_dates=["ds"])
                    # Derive effective horizon actually achieved (# future rows)
                    future_rows = fc[fc["y"].isna()].shape[0]
                    st.success(
                        f"Forecast complete. Effective horizon: {future_rows} periods."
                    )
                    fig_fc = build_forecast_line(fc)
                    st.plotly_chart(fig_fc, use_container_width=True)
            except Exception as e:
                st.error(f"Forecasting failed: {e}")

# LINEAR REGRESSION-BASED FORECASTING (PLACEHOLDER)
with st.expander("Regression-based forecasting (placeholder)", expanded=False):
    df_lr = _tables.get("production_log", pd.DataFrame())

    # Same basic controls for consistency
    col_lr1, col_lr2 = st.columns(2)
    with col_lr1:
        lr_horizon = st.number_input(
            "Requested Horizon (periods)",
            min_value=1,
            max_value=720,
            value=60,
            step=1,
        )
        lr_agg_freq = st.selectbox("Aggregation Frequency", ["D", "W", "M"], index=0, key="lr_freq")
    with col_lr2:
        lr_adapt = st.checkbox("Adaptive Horizon", value=True, key="lr_adapt")
        lr_multiplier = st.number_input(
            "Horizon Multiplier", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="lr_mult"
        )
        lr_agg_metric = st.selectbox(
            "Aggregation Metric",
            ["mean", "median", "sum", "count"],
            index=0,
            key="lr_metric",
        )

    if df_lr.empty:
        st.info("No production_log data available for regression-based forecasting.")
    else:
        run_lr = st.button("Run Linear Forecast")
        if run_lr:
            try:
                lr_path = "time_series_forecasted_data.csv"  # reuse the same path pattern for now
                linear_forecast(
                    df_lr,
                    horizon=int(lr_horizon),
                    adapt_horizon=lr_adapt,
                    horizon_multiplier=float(lr_multiplier),
                    agg_freq=lr_agg_freq,
                    agg_metric=lr_agg_metric,
                )
                import os
                if os.path.exists(lr_path):
                    fc_lr = pd.read_csv(lr_path, parse_dates=["ds"])
                    future_rows_lr = fc_lr[fc_lr["y"].isna()].shape[0]
                    st.success(
                        f"Linear forecast complete (placeholder). Effective horizon: {future_rows_lr} periods."
                    )
                    fig_lr = build_forecast_line(fc_lr)
                    st.plotly_chart(fig_lr, use_container_width=True)
                st.caption(
                    "Note: placeholder model holds the last observed value into the future. "
                    "A proper regression fit is coming soon."
                )
            except Exception as e:
                st.error(f"Linear forecasting failed: {e}")

# MULTIVARIATE REGRESSION FORECASTING (PLACEHOLDER)
st.write("Multivariate regression forecasting (placeholder)")


# Gantt charts
st.subheader("Gantt")
chart = GanttChart()
prod = _tables.get("production_log", pd.DataFrame())
steps = _tables.get("process_steps", pd.DataFrame())
products = _tables.get("products", pd.DataFrame())
targets = _tables.get("production_targets", pd.DataFrame())

# Optional name mappings
product_names = None
if not products.empty and {"product_id", "name"}.issubset(products.columns):
    product_names = products[["product_id", "name"]].copy()

step_names = None
if not steps.empty and {"step_id", "step_name"}.issubset(steps.columns):
    step_names = steps[["step_id", "step_name"]].drop_duplicates().copy()

fig_actual = chart.actual_gantt(
    prod,
    product_names=product_names,
    step_names=step_names,
)
if fig_actual is not None:
    st.plotly_chart(fig_actual, width="stretch")
else:
    st.info("Actual Gantt unavailable (needs start_time/end_time in production_log).")

fig_planned = chart.planned_gantt(
    steps,
    production_log=prod,  # enables optional run-based anchoring if desired
    product_names=product_names,
    anchor="run_start",  # anchor planned bars at earliest start_time per (product_id, run_id)
)
if fig_planned is not None:
    st.plotly_chart(fig_planned, width="stretch")
else:
    st.info("Planned Gantt unavailable (needs process_steps with estimated_time).")


sp = per_step_progress(steps, prod, targets=targets if not targets.empty else None)
overall = overall_progress_by_product(sp)

st.subheader("Progress")

# Overall progress per product
if not overall.empty:
    for _, row in overall.sort_values("product_id").iterrows():
        pct = float(row["overall_progress"])
        st.write(f"**{row['product_id']}** — {pct:.0%}")
        st.progress(max(0.0, min(1.0, pct)))
else:
    st.info("No overall progress available (check data or filters).")

# Per-step table (shows target_qty when present)
if not sp.empty:
    disp = sp.copy()
    disp["progress_pct"] = (disp["progress"] * 100).round(1)
    cols = ["product_id", "step_id"]
    if "step_name" in disp.columns:
        cols.append("step_name")
    if "target_qty" in disp.columns:
        cols.append("target_qty")
    cols += ["complete_qty", "in_progress_qty", "progress_pct"]
    st.dataframe(disp[cols], width="stretch")
else:
    st.info("No step-level progress available.")

# Data preview.
# NOTE: Let's keep this at the bottom as a footer when adding future data viz content.
st.divider()
with st.expander("Preview Data"):
    for name, df in _tables.items():
        st.markdown(f"### {name}")
        st.dataframe(df.head(20), width="stretch")
