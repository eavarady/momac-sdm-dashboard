import os

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

import streamlit as st
import pandas as pd
import plotly.express as px
from adapters.csv_adapter import read_csv_tables, get_last_load_stats
from adapters.sheets_adapter import read_sheets
from adapters.excel_adapter import (
    read_excel_tables as read_excel,
    get_last_load_stats as get_excel_stats,
)
from kpi.kpi_calculator import compute_all_kpis
from kpi.time_per_step import compute_time_per_step
from ml.bottleneck_detector import detect_bottleneck, top_bottlenecks
from visualizations.gantt import GanttChart
from kpi.progress import (
    per_step_progress,
    overall_progress_by_product,
    per_run_progress,
)
from ml.time_series import time_series_forecast
from visualizations.line_chart import build_forecast_line
from ml.linear_forecast import linear_forecast
from ml.multivariate_forecast import (
    run_multivariate_forecast,
    ForecastFeatureAdequacyError,
)

st.set_page_config(page_title="MOMAC SDM Dashboard", layout="wide")

st.title("MOMAC SDM Dashboard")

# --- Export (PDF) registry for charts ---
def _register_chart(key: str, label: str, fig) -> None:
    if fig is None:
        return
    reg = st.session_state.setdefault("export_charts", {})
    reg[key] = {"label": label, "fig": fig}

def _render_export_pdf_ui(container) -> None:
    container.markdown("---")
    container.subheader("Export (PDF)")
    _reg = st.session_state.get("export_charts", {})
    if not _reg:
        container.caption("Generate a chart to enable export.")
        return
    selected_keys = container.multiselect(
        "Charts ready to export",
        options=list(_reg.keys()),
        format_func=lambda k: _reg[k]["label"],
        key="pdf_export_select",
    )
    if selected_keys:
        try:
            from export.pdf_exporter import figures_to_pdf_bytes

            # Show status where the download button will appear
            dl_placeholder = container.empty()
            dl_placeholder.write("Generating PDF...")
            pdf_bytes = figures_to_pdf_bytes(
                [(f"{_reg[k]['label']}", _reg[k]["fig"]) for k in selected_keys]
            )
            # Replace status with the download button
            dl_placeholder.download_button(
                label="Download selected charts (PDF)",
                data=pdf_bytes,
                file_name="momac_charts.pdf",
                mime="application/pdf",
                key="download_pdf",
            )
        except Exception as e:
            container.warning(f"PDF export failed: {e}")

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
    # Placeholder for Export (PDF) UI to be populated after charts are registered
    export_pdf_container = st.container()

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

# FORECASTING HEADER
st.subheader("Production Performance Forecasting")
st.write("Forecast aggregated production metrics and explore what‑if scenarios.")

with st.expander("Time Series Forecasting", expanded=False):
    st.caption(
        "Predict future performance trends using time series patterns and seasonality.",
        help=(
            "Time series forecasting looks at past performance over time and projects it forward, "
            "automatically capturing patterns like trends and seasonality. Instead of "
            "just drawing a straight line, it can curve and adjust to repeating cycles "
            "in the data (for example, weekly or seasonal patterns)."
        ),
    )
    df = _tables.get("production_log", pd.DataFrame())
    # Friendly aggregation metric labels -> internal codes2
    AGG_FRIENDLY = {
        "Average Cycle Time (mean)": "mean",
        "Typical Cycle Time (median)": "median",
        "Total Processing Hours (sum)": "sum",
        "Completed Runs (count)": "count",
    }
    Y_AXIS_TITLES = {
        "mean": "Average Cycle Time (hrs)",
        "median": "Typical Cycle Time (hrs)",
        "sum": "Total Processing Hours (hrs)",
        "count": "Completed Runs (count)",
    }
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
    agg_metric_label = st.selectbox(
        "Aggregation Metric",
        list(AGG_FRIENDLY.keys()),
        index=0,
        help="How to aggregate event durations inside each period.",
        key="ts_metric",
    )
    agg_metric = AGG_FRIENDLY[agg_metric_label]

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
                    fig_fc.update_yaxes(title=Y_AXIS_TITLES.get(agg_metric, "Value"))
                    st.plotly_chart(fig_fc, use_container_width=True)
                    _register_chart("forecast_ts", "Forecast (Time Series)", fig_fc)
            except Exception as e:
                st.error(f"Forecasting failed: {e}")

# LINEAR REGRESSION-BASED FORECASTING
with st.expander("Regression-based forecasting", expanded=False):
    st.caption(
        "Predict future performance trends using linear regression.",
        help=(
            "Linear regression looks at past data and finds a straight line that best fits the trend. "
            "Imagine plotting dots on a chart and drawing the line that best passes through them. "
            "That line is then extended into the future to produce a forecast. "
            "It's a simple approach that's most useful when the data shows a clear upward or downward trend."
        ),
    )
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
        lr_agg_freq = st.selectbox(
            "Aggregation Frequency", ["D", "W", "M"], index=0, key="lr_freq"
        )
    with col_lr2:
        lr_adapt = st.checkbox("Adaptive Horizon", value=True, key="lr_adapt")
        lr_multiplier = st.number_input(
            "Horizon Multiplier",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="lr_mult",
        )
        lr_agg_metric_label = st.selectbox(
            "Aggregation Metric",
            list(AGG_FRIENDLY.keys()),
            index=0,
            key="lr_metric",
            help="How to aggregate event durations inside each period.",
        )
        lr_agg_metric = AGG_FRIENDLY[lr_agg_metric_label]

    if df_lr.empty:
        st.info("No production_log data available for regression-based forecasting.")
    else:
        run_lr = st.button("Run Linear Forecast")
        if run_lr:
            try:
                lr_path = "linear_forecasted_data.csv"
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
                        f"Linear forecast complete. Effective horizon: {future_rows_lr} periods."
                    )
                    fig_lr = build_forecast_line(fc_lr)
                    fig_lr.update_yaxes(title=Y_AXIS_TITLES.get(lr_agg_metric, "Value"))
                    st.plotly_chart(fig_lr, use_container_width=True)
                    _register_chart("forecast_lr", "Forecast (Linear Regression)", fig_lr)

            except Exception as e:
                st.error(f"Linear forecasting failed: {e}")

with st.expander("Multivariate Regression (Scenario Forecast)", expanded=False):
    st.caption(
        "Configure a what-if scenario using assumed values for operational drivers.",
        help=(
            "Multivariate regression forecasting lets you run ‘what-if’ scenarios by fixing "
            "assumptions about key factors (like operator count, shift type, or defect rate) and seeing how they affect production metrics. Unlike "
            "simple regression-based forecasts, which only use time, this approach allows you to explore how "
            "different operational conditions might impact outcomes. Think of it as a scenario "
            "planner: you choose the values, and the model shows the projected trend under those "
            "conditions."
        ),
    )

    # Core forecast controls (mirrors earlier sections; unique keys to avoid clashes)
    col_mv1, col_mv2 = st.columns(2)
    with col_mv1:
        mv_horizon = st.number_input(
            "Requested Horizon (periods)",
            min_value=1,
            max_value=720,
            value=60,
            step=1,
            key="mv_horizon",
        )
        mv_agg_freq = st.selectbox(
            "Aggregation Frequency", ["D", "W", "M"], index=0, key="mv_freq"
        )
        mv_agg_metric_label = st.selectbox(
            "Aggregation Metric",
            list(AGG_FRIENDLY.keys()),
            index=0,
            key="mv_metric",
            help="How to aggregate event durations inside each period.",
        )
        mv_agg_metric = AGG_FRIENDLY[mv_agg_metric_label]
    with col_mv2:
        mv_adapt = st.checkbox(
            "Adaptive Horizon",
            value=True,
            key="mv_adapt",
            help="Reduce horizon based on history span * multiplier",
        )
        mv_multiplier = st.number_input(
            "Horizon Multiplier",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="mv_mult",
        )
    # No explicit baseline strategy selector here; multivariate regression will auto-fallback
    # to persistence (extend last or single value) only when data is insufficient.

    st.markdown("---")
    st.subheader("Scenario Variables")
    st.caption(
        "Currently only defect rate % is available (others removed until temporal history exists)."
    )

    # Determine simple defaults from data (if available)
    machines_df = _tables.get("machines", pd.DataFrame())
    quality_df = _tables.get("quality_checks", pd.DataFrame())
    prod_log_df = _tables.get("production_log", pd.DataFrame())
    operators_df = _tables.get("operators", pd.DataFrame())

    # Operator count default
    # operator_count removed until temporal history exists

    # Defect rate default (fail / total *100)
    if not quality_df.empty and {"result"}.issubset(quality_df.columns):
        total_q = len(quality_df)
        fails = (quality_df["result"].str.lower() == "fail").sum()
        default_defect_rate = (fails / max(1, total_q)) * 100.0
    else:
        default_defect_rate = 5.0

    # Shift type heuristic (most common shift or day if absent)
    lines_df = _tables.get("production_lines", pd.DataFrame())
    if not lines_df.empty and {"shift"}.issubset(lines_df.columns):
        shift_col = lines_df["shift"].dropna().astype(str)
        # Unique observed shift options
        shift_options = sorted(shift_col.unique().tolist())
        # Safe mode(): may be empty after dropna
        common_shift = (
            shift_col.mode().iloc[0]
            if not shift_col.mode().empty
            else (shift_options[0] if shift_options else "day")
        )
    else:
        shift_options = ["day", "night"]
        common_shift = "day"

    # Friendly label -> internal feature key mapping (internal keys stay stable for modeling)
    FEATURE_LABELS = {
        "Defect rate %": "defect_rate_pct",
    }

    col_vars1, col_vars2 = st.columns(2)
    with col_vars1:
        inc_defect_rate = st.checkbox(
            "Defect rate %",
            value=False,
            key="mv_inc_defect",
            help="Feature key: defect_rate_pct",
        )
    with col_vars2:
        st.write("")

    # Inputs for selected variables
    col_inputs1, col_inputs2 = st.columns(2)
    with col_inputs1:
        pass
    with col_inputs2:
        if inc_defect_rate:
            defect_rate_pct = st.slider(
                "Assumed Defect Rate %",
                min_value=0.0,
                max_value=100.0,
                value=float(round(default_defect_rate, 2)),
                step=0.5,
                key="mv_defect_rate",
            )

    st.markdown("---")
    run_mv = st.button("Generate Multivariate Scenario Forecast")
    if run_mv:
        scenario = {
            "horizon": int(mv_horizon),
            "adapt_horizon": bool(mv_adapt),
            "horizon_multiplier": float(mv_multiplier),
            "aggregation": {"freq": mv_agg_freq, "metric": mv_agg_metric},
            "included_variables": [],
            "assumptions": {},
        }
        # (operator_count, shift_type removed)
        if inc_defect_rate:
            fk = FEATURE_LABELS["Defect rate %"]
            scenario["included_variables"].append(fk)
            scenario["assumptions"][fk] = defect_rate_pct
        st.session_state["multivariate_scenario"] = scenario
        try:
            mv_path = "multivariate_forecasted_data.csv"
            fc_mv = run_multivariate_forecast(_tables, scenario, output_path=mv_path)
            st.success("Multivariate forecast generated.")
            from visualizations.line_chart import build_forecast_line as _bfl

            fig_mv = _bfl(fc_mv)
            Y_AXIS_TITLES = {
                "mean": "Average Cycle Time (hrs)",
                "median": "Typical Cycle Time (hrs)",
                "sum": "Total Processing Hours (hrs)",
                "count": "Completed Runs (count)",
            }
            fig_mv.update_yaxes(title=Y_AXIS_TITLES.get(mv_agg_metric, "Value"))
            st.plotly_chart(fig_mv, use_container_width=True)
            _register_chart("forecast_mv", "Forecast (Scenario)", fig_mv)
            # Influence diagnostics
            try:
                import json

                meta_path = os.path.splitext(mv_path)[0] + "_meta.json"
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    low = meta.get("low_influence") or []
                    if low:
                        st.warning(
                            "Low influence: the following scenario variable(s) had near-zero standardized coefficients and may have negligible effect: "
                            + ", ".join(low)
                        )
            except Exception:
                pass
        except ForecastFeatureAdequacyError as fe:
            st.error("One or more selected features are not adequate for modeling.")
            st.warning("\n".join(f"{k}: {v}" for k, v in fe.details.items()))
            st.info("Unselect the inadequate feature(s) and re-run the forecast.")
        except Exception as e:
            if not scenario["included_variables"]:
                st.warning(
                    "Select at least one scenario variable to run a multivariate scenario."
                )
            else:
                st.error(f"Multivariate forecasting failed: {e}")


# HEURISTIC BOTTLENECK DETECTION
bn = detect_bottleneck(
    _tables.get("process_steps", pd.DataFrame()),
    _tables.get("production_log", pd.DataFrame()),
)
st.subheader("Top 3 Bottlenecks by WIP")
prod = _tables.get("production_log", pd.DataFrame())
top3 = top_bottlenecks(prod, top_n=3)
if top3.empty:
    st.write("No in-progress work detected.")
else:
    st.dataframe(top3, width="stretch")


# Gantt charts
st.subheader("Gantt")
chart = GanttChart()
prod = _tables.get("production_log", pd.DataFrame())
steps = _tables.get("process_steps", pd.DataFrame())
products = _tables.get("products", pd.DataFrame())

# Optional name mappings
product_names = None
if not products.empty and {"product_id", "name"}.issubset(products.columns):
    product_names = products[["product_id", "name"]].copy()

step_names = None
if not steps.empty and {"step_id", "step_name"}.issubset(steps.columns):
    step_names = steps[["step_id", "step_name"]].drop_duplicates().copy()

# View toggle controls
col_g1, col_g2 = st.columns(2)
with col_g1:
    # Default: by run if run_id present, else by step
    has_run = (not prod.empty) and ("run_id" in prod.columns)
    view_actual_label = st.radio(
        "Actual Gantt view",
        ["By run", "By step"],
        index=(0 if has_run else 1),
        key="gantt_actual_view",
        horizontal=True,
    )
    view_actual = "by_run" if view_actual_label == "By run" else "by_step"
with col_g2:
    has_runs_for_planned = (not prod.empty) and ("run_id" in prod.columns)
    view_planned_label = st.radio(
        "Planned Gantt view",
        ["By run", "By step"],
        index=(0 if has_runs_for_planned else 1),
        key="gantt_planned_view",
        horizontal=True,
    )
    view_planned = "by_run" if view_planned_label == "By run" else "by_step"

fig_actual = chart.actual_gantt(
    prod,
    product_names=product_names,
    step_names=step_names,
    view=view_actual,
)
if fig_actual is not None:
    st.plotly_chart(fig_actual, width="stretch")
    _register_chart(
        f"gantt_actual_{view_actual}",
        f"Gantt — Actual ({'by run' if view_actual == 'by_run' else 'by step'})",
        fig_actual,
    )
else:
    st.info("Actual Gantt unavailable (needs start_time/end_time in production_log).")

fig_planned = chart.planned_gantt(
    steps,
    production_log=prod,  # enables optional run-based anchoring if desired
    product_names=product_names,
    anchor="run_start",  # anchor planned bars at earliest start_time per (product_id, run_id)
    view=view_planned,
)
if fig_planned is not None:
    st.plotly_chart(fig_planned, width="stretch")
    _register_chart(
        f"gantt_planned_{view_planned}",
        f"Gantt — Planned ({'by run' if view_planned == 'by_run' else 'by step'})",
        fig_planned,
    )
else:
    st.info("Planned Gantt unavailable (needs process_steps with estimated_time).")

# (moved export UI render to end to include all later-registered charts)


st.subheader("Time per Step")
plog = _tables.get("production_log", pd.DataFrame())

# Optional date range filter on start_time
col_dr1, col_dr2, col_dr3 = st.columns([1, 1, 2])
with col_dr1:
    date_start = st.date_input("From date", value=None, key="tps_from")
with col_dr2:
    date_end = st.date_input("To date", value=None, key="tps_to")

# Normalize to string for KPI (None if not set)
ds_val = pd.Timestamp(date_start).isoformat() if date_start else None
de_val = pd.Timestamp(date_end).isoformat() if date_end else None

agg = compute_time_per_step(
    plog,
    process_steps=_tables.get("process_steps", pd.DataFrame()),
    products=_tables.get("products", pd.DataFrame()),
    date_start=ds_val,
    date_end=de_val,
)
if agg.empty:
    st.info("Insufficient data to compute time-per-step (need completed events).")
else:
    # Product filter
    products_list = (
        agg.sort_values(["product_label"])["product_label"].unique().tolist()
    )
    col_tp1, col_tp2 = st.columns([2, 3])
    with col_tp1:
        selected_product = st.selectbox(
            "Filter by product",
            options=["All"] + products_list,
            index=0,
            key="tps_product",
        )
        display_df = agg.copy()
        if selected_product != "All":
            display_df = display_df[display_df["product_label"] == selected_product]
        display_df = display_df.sort_values(
            ["product_label", "avg_duration_hours"], ascending=[True, False]
        )
        st.dataframe(
            display_df[
                [
                    "product_label",
                    "step_label",
                    "avg_duration_hours",
                    "median_duration_hours",
                    "std_duration_hours",
                    "events",
                ]
            ].rename(
                columns={
                    "product_label": "Product",
                    "step_label": "Step",
                    "avg_duration_hours": "Avg Duration (hrs)",
                    "median_duration_hours": "Median (hrs)",
                    "std_duration_hours": "Std Dev (hrs)",
                    "events": "Completed Events",
                }
            ),
            use_container_width=True,
        )

        # CSV export
        @st.cache_data
        def _to_csv(df: pd.DataFrame) -> bytes:
            return df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV",
            data=_to_csv(
                display_df[
                    [
                        "product_label",
                        "step_label",
                        "avg_duration_hours",
                        "median_duration_hours",
                        "std_duration_hours",
                        "events",
                    ]
                ].rename(
                    columns={
                        "product_label": "Product",
                        "step_label": "Step",
                        "avg_duration_hours": "Avg Duration (hrs)",
                        "median_duration_hours": "Median (hrs)",
                        "std_duration_hours": "Std Dev (hrs)",
                        "events": "Completed Events",
                    }
                )
            ),
            file_name="time_per_step.csv",
            mime="text/csv",
        )
    with col_tp2:
        if selected_product != "All":
            chart_df = display_df.sort_values("avg_duration_hours", ascending=False)
            if not chart_df.empty:
                from visualizations.time_per_step import build_time_per_step_bar
                fig_bar = build_time_per_step_bar(
                    chart_df[["step_label", "avg_duration_hours"]],
                    product_label=selected_product,
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                _register_chart(
                    f"tps_bar_{selected_product}",
                    f"Time per Step — {selected_product}",
                    fig_bar,
                )
        else:
            st.caption("Select a product to see a step-level bar chart.")

sp = per_step_progress(steps, prod)
spr = per_run_progress(
    steps,
    prod,
    runs=_tables.get("runs", pd.DataFrame()),
)
overall = overall_progress_by_product(sp)

st.subheader("Progress")
# Per-run progress bars, current runs only (Not 100%), sorted by run_id
st.subheader("Current Runs Progress")
if not spr.empty:
    current_runs = spr[spr["progress"] < 1.0].sort_values("run_id")
    if current_runs.empty:
        st.info("No current runs in progress (all runs complete).")
    else:
        # Show progress bars
        for _, row in current_runs.iterrows():
            pct = float(row["progress"])
            st.write(f"**{row['run_id']}** — {pct:.0%}")
            st.progress(max(0.0, min(1.0, pct)))
        # Build an exportable bar chart silently (do not render to UI)
        try:
            from visualizations.progress_charts import (
                build_current_runs_progress_bar,
            )
            fig_curr = build_current_runs_progress_bar(current_runs)
            _register_chart(
                "progress_current_runs", "Current Runs Progress (bar)", fig_curr
            )
        except Exception:
            pass

# Separator
st.markdown("---")
# Overall progress per product bars
st.subheader("Overall Progress by Product")
if not overall.empty:
    for _, row in overall.sort_values("product_id").iterrows():
        pct = float(row["overall_progress"])
        st.write(f"**{row['product_id']}** — {pct:.0%}")
        st.progress(max(0.0, min(1.0, pct)))
    # Build an exportable bar chart silently (do not render to UI)
    try:
        from visualizations.progress_charts import (
            build_overall_progress_by_product_bar,
        )
        fig_overall = build_overall_progress_by_product_bar(overall)
        _register_chart(
            "progress_overall_by_product",
            "Overall Progress by Product (bar)",
            fig_overall,
        )
    except Exception:
        pass
else:
    st.info("No overall progress available (check data or filters).")


# Separator
st.markdown("---")
# Per-run progress table
st.subheader("Per-Run Progress")
if not spr.empty:
    disp_run = spr.copy()
    # Compute percent variants if present
    if "progress_steps" in disp_run.columns:
        disp_run["progress_steps_pct"] = (
            disp_run["progress_steps"].astype(float) * 100.0
        ).round(1)
    if "progress_qty" in disp_run.columns and disp_run["progress_qty"].notna().any():
        disp_run["progress_qty_pct"] = (
            disp_run["progress_qty"].astype(float) * 100.0
        ).round(1)
    disp_run["progress_pct"] = (disp_run["progress"].astype(float) * 100.0).round(1)

    cols = ["product_id", "run_id"]
    if "planned_qty" in disp_run.columns:
        cols.append("planned_qty")
    if "execution_mode" in disp_run.columns:
        cols.append("execution_mode")
    if "progress_steps_pct" in disp_run.columns:
        cols.append("progress_steps_pct")
    if "progress_qty_pct" in disp_run.columns:
        cols.append("progress_qty_pct")
    cols.append("progress_pct")
    st.dataframe(disp_run[cols], width="stretch")
else:
    st.info("No per-run progress available (check run_id and data).")

# Data preview.
# NOTE: Let's keep this at the bottom as a footer when adding future data viz content.
st.divider()
with st.expander("Preview Data"):
    for name, df in _tables.items():
        st.markdown(f"### {name}")
        st.dataframe(df.head(20), width="stretch")

# Finally render Export (PDF) UI in the sidebar now that all charts could be registered
try:
    _render_export_pdf_ui(export_pdf_container)
except Exception:
    pass
