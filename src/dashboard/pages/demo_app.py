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
from ml.bottleneck_detector import detect_bottleneck, top_bottlenecks
from visualizations.gantt import GanttChart
from visualizations.time_per_step_viz import (
    build_time_per_step_bar,
    prepare_step_duration_events,
    build_step_duration_histogram,
    DEFAULT_HIST_BINS,
)
from kpi.time_per_step import compute_time_per_step
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
    compute_feature_defaults,
    FEATURE_LABELS,
)
from utils.date_ranges import compute_preset_range
from export.csv_exporter import to_csv_bytes, safe_filename
from export.excel_exporter import to_excel_bytes
from workflow.validator import steps_from_dataframe, validate_dependencies
from visualizations.dependency_diagram import build_step_dependency_graph

# Opt-in to pandas future behavior to avoid silent downcasting
pd.set_option("future.no_silent_downcasting", True)


# --- Export (PDF) registry for charts ---
def _register_chart(key: str, label: str, fig) -> None:
    if fig is None:
        return
    reg = st.session_state.setdefault("export_charts", {})
    reg[key] = {"label": label, "fig": fig}


# Compatibility wrapper in case build_forecast_line doesn't accept newer kwargs
def _build_forecast_line_safe(df, **kwargs):
    try:
        return build_forecast_line(df, **kwargs)
    except TypeError:
        # Drop unknown kwargs (e.g., connect_actual) and retry
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in {"connect_actual"}}
        return build_forecast_line(df, **safe_kwargs)


def _render_export_pdf_ui(container) -> None:
    container.markdown("---")
    container.subheader(
        "Export (PDF)",
        help="Download selected charts and KPIs in a single PDF document.",
    )
    _reg = st.session_state.get("export_charts", {})
    if not _reg:
        container.caption("Generate a chart to enable export.")
        return
    selected_keys = container.multiselect(
        "Charts ready to export",
        help="Generated charts will also appear here for selection.",
        options=list(_reg.keys()),
        format_func=lambda k: _reg[k]["label"],
        key="pdf_export_select",
    )

    # Use a session-scoped cache to avoid regenerating the PDF when the user
    # simply clicks the download button (Streamlit triggers a rerun on click).
    # We only regenerate when the selection of keys changes.
    cache = st.session_state.setdefault("_pdf_export_cache", {})
    sel_tuple = tuple(selected_keys) if selected_keys else None

    if not selected_keys:
        container.caption("Select one or more charts to enable export.")
        # Clear any cached bytes for empty selection
        if cache.get("keys"):
            cache.clear()
        return

    # Create a single placeholder which we will replace with the download button
    dl_placeholder = container.empty()

    # If the selection changed, recompute and store the bytes
    if cache.get("keys") != sel_tuple or "bytes" not in cache:
        try:
            from export.pdf_exporter import figures_to_pdf_bytes

            dl_placeholder.write("Generating PDF...")
            pdf_bytes = figures_to_pdf_bytes(
                [(f"{_reg[k]['label']}", _reg[k]["fig"]) for k in selected_keys]
            )
            cache["keys"] = sel_tuple
            cache["bytes"] = pdf_bytes
        except Exception as e:
            # Replace the placeholder content with a warning
            dl_placeholder.warning(f"PDF export failed: {e}")
            return

    # Replace the placeholder with the download button using the cached bytes
    dl_placeholder.download_button(
        label="Download selected charts (PDF)",
        data=cache.get("bytes"),
        file_name="momac_charts.pdf",
        mime="application/pdf",
        key="download_pdf",
    )


st.set_page_config(page_title="MOMAC SDM Dashboard", layout="wide")

st.title("MOMAC SDM Dashboard")


def render_forecasting_view(_tables):
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
        if df.empty:
            st.info("No production_log data available for forecasting.")
        else:
            submitted_ts = False
            with st.form("forecast_ts_form"):
                col_fs1, col_fs2 = st.columns(2)
                with col_fs1:
                    user_horizon = st.number_input(
                        "Requested Horizon (periods)",
                        min_value=1,
                        max_value=720,
                        value=60,
                        step=1,
                        help="Number of future periods to forecast before adaptive reduction.",
                    )
                    multiplier = st.number_input(
                        "Horizon Multiplier",
                        min_value=0.1,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                    )
                    adapt = st.checkbox(
                        "Adaptive Horizon",
                        value=True,
                        help="Reduce horizon based on history span * multiplier",
                    )
                    ts_connect = st.checkbox(
                        "Line chart (connect actual points)",
                        value=True,
                        help="If unchecked, show actuals as scatter points only.",
                        key="ts_connect",
                    )
                with col_fs2:
                    agg_freq = st.selectbox(
                        "Aggregation Frequency", ["D", "W", "M"], index=0
                    )
                    agg_metric_label = st.selectbox(
                        "Aggregation Metric",
                        list(AGG_FRIENDLY.keys()),
                        index=0,
                        help="How to aggregate event durations inside each period.",
                        key="ts_metric",
                    )
                    agg_metric = AGG_FRIENDLY[agg_metric_label]
                    baseline_strategy = st.selectbox(
                        "Baseline Strategy", ["mean", "linear"], index=0
                    )

                submitted_ts = st.form_submit_button("Run Forecast")

            if submitted_ts:
                try:
                    forecast_path = "time_series_forecasted_data.csv"
                    fc = time_series_forecast(
                        df,
                        horizon=int(user_horizon),
                        baseline_strategy=baseline_strategy,
                        adapt_horizon=adapt,
                        horizon_multiplier=float(multiplier),
                        agg_freq=agg_freq,
                        agg_metric=agg_metric,
                    )
                    # Plot directly from returned df and provide CSV download
                    if fc is not None and not fc.empty:
                        # Derive effective horizon actually achieved (# future rows)
                        future_rows = (
                            fc[fc["y"].isna()].shape[0] if "y" in fc.columns else 0
                        )
                        st.success(
                            f"Forecast complete. Effective horizon: {future_rows} periods."
                        )
                        # Save to session for potential reuse
                        st.session_state["fc_ts_df"] = fc.copy()
                        fig_fc = _build_forecast_line_safe(
                            fc, connect_actual=bool(ts_connect)
                        )
                        fig_fc.update_yaxes(
                            title=Y_AXIS_TITLES.get(agg_metric, "Value")
                        )
                        st.plotly_chart(fig_fc, use_container_width=True)
                        _register_chart(
                            "forecast_ts",
                            "Forecast (Time Series)",
                            fig_fc,
                        )
                        st.download_button(
                            label="Download CSV",
                            data=to_csv_bytes(fc, index=False),
                            file_name="time_series_forecasted_data.csv",
                            mime="text/csv",
                            key="dl_ts_fc",
                        )
                        st.download_button(
                            label="Download .xlsx",
                            data=to_excel_bytes(
                                fc, index=False, sheet_name="Forecast (TS)"
                            ),
                            file_name="time_series_forecasted_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_ts_fc_xlsx",
                        )
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
                "It's a simple approach that's most useful when the data shows a clear upward or downward linear trend."
            ),
        )
        df_lr = _tables.get("production_log", pd.DataFrame())

        if df_lr.empty:
            st.info(
                "No production_log data available for regression-based forecasting."
            )
        else:
            submitted_lr = False
            with st.form("forecast_lr_form"):
                col_lr1, col_lr2 = st.columns(2)
                with col_lr1:
                    lr_horizon = st.number_input(
                        "Requested Horizon (periods)",
                        min_value=1,
                        max_value=720,
                        value=60,
                        step=1,
                    )

                    lr_multiplier = st.number_input(
                        "Horizon Multiplier",
                        min_value=0.1,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                        key="lr_mult",
                    )
                    lr_adapt = st.checkbox(
                        "Adaptive Horizon",
                        value=True,
                        key="lr_adapt",
                        help="Reduce horizon based on history span * multiplier",
                    )

                    lr_connect = st.checkbox(
                        "Line chart (connect actual points)",
                        value=True,
                        key="lr_connect",
                        help="If unchecked, show actuals as scatter points only.",
                    )

                with col_lr2:
                    lr_agg_freq = st.selectbox(
                        "Aggregation Frequency", ["D", "W", "M"], index=0, key="lr_freq"
                    )
                    lr_agg_metric_label = st.selectbox(
                        "Aggregation Metric",
                        list(AGG_FRIENDLY.keys()),
                        index=0,
                        key="lr_metric",
                        help="How to aggregate event durations inside each period.",
                    )
                    lr_agg_metric = AGG_FRIENDLY[lr_agg_metric_label]

                submitted_lr = st.form_submit_button("Run Linear Forecast")

            if submitted_lr:
                try:
                    fc_lr = linear_forecast(
                        df_lr,
                        horizon=int(lr_horizon),
                        adapt_horizon=lr_adapt,
                        horizon_multiplier=float(lr_multiplier),
                        agg_freq=lr_agg_freq,
                        agg_metric=lr_agg_metric,
                    )
                    if fc_lr is not None and not fc_lr.empty:
                        future_rows_lr = (
                            fc_lr[fc_lr["y"].isna()].shape[0]
                            if "y" in fc_lr.columns
                            else 0
                        )
                        st.success(
                            f"Linear forecast complete. Effective horizon: {future_rows_lr} periods."
                        )
                        st.session_state["fc_lr_df"] = fc_lr.copy()
                        fig_lr = _build_forecast_line_safe(
                            fc_lr, connect_actual=bool(lr_connect)
                        )
                        fig_lr.update_yaxes(
                            title=Y_AXIS_TITLES.get(lr_agg_metric, "Value")
                        )
                        st.plotly_chart(fig_lr, use_container_width=True)
                        _register_chart(
                            "forecast_lr",
                            "Forecast (Linear Regression)",
                            fig_lr,
                        )
                        st.download_button(
                            label="Download CSV",
                            data=to_csv_bytes(fc_lr, index=False),
                            file_name="linear_forecasted_data.csv",
                            mime="text/csv",
                            key="dl_lr_fc",
                        )
                        st.download_button(
                            label="Download .xlsx",
                            data=to_excel_bytes(
                                fc_lr, index=False, sheet_name="Forecast (LR)"
                            ),
                            file_name="linear_forecasted_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_lr_fc_xlsx",
                        )

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

        # Track confirmed scenario setup across reruns
        confirmed_selection = st.session_state.get("mv_confirmed_selection", {})
        confirmed_settings = st.session_state.get("mv_confirmed_settings")

        # Centralized defaults & label mapping now provided by ml.multivariate_forecast
        _mv_defaults = compute_feature_defaults(_tables)
        default_defect_rate = _mv_defaults.get("defect_rate_pct", 5.0)
        night_share_default = _mv_defaults.get("shift_night_share", 0.0)
        energy_default = _mv_defaults.get("avg_energy_consumption", 0.0)
        wip_default = _mv_defaults.get("wip_proxy", 0.0)
        cov_default = _mv_defaults.get("inspection_intensity", 0.0)
        headcount_default = int(
            round(_mv_defaults.get("operator_headcount", 0.0) or 0.0)
        )
        eff_fte_default = float(_mv_defaults.get("effective_operator_fte", 0.0) or 0.0)

        with st.form("forecast_mv_setup_form"):
            col_mv1, col_mv2 = st.columns(2)
            with col_mv1:
                st.number_input(
                    "Requested Horizon (periods)",
                    min_value=1,
                    max_value=720,
                    value=60,
                    step=1,
                    key="mv_horizon",
                )
                st.number_input(
                    "Horizon Multiplier",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key="mv_mult",
                )
                st.checkbox(
                    "Adaptive Horizon",
                    value=True,
                    key="mv_adapt",
                    help="Reduce horizon based on history span * multiplier",
                )
                st.checkbox(
                    "Line chart (connect actual points)",
                    value=True,
                    key="mv_connect",
                    help="If unchecked, show actuals as scatter points only.",
                )
            with col_mv2:
                st.selectbox(
                    "Aggregation Frequency", ["D", "W", "M"], index=0, key="mv_freq"
                )
                st.selectbox(
                    "Aggregation Metric",
                    list(AGG_FRIENDLY.keys()),
                    index=0,
                    key="mv_metric",
                    help="How to aggregate event durations inside each period.",
                )

            st.markdown("---")
            st.subheader("Scenario Variables")
            st.caption(
                "Select variables to include in the model. Historical values are derived from data; your inputs apply only to future periods."
            )

            col_vars1, col_vars2 = st.columns(2)
            with col_vars1:
                st.checkbox(
                    "Night shift share",
                    value=False,
                    key="mv_inc_shift_night",
                    help="Feature key: shift_night_share (night events / total events)",
                )
                st.checkbox(
                    "Inspection coverage",
                    value=False,
                    key="mv_inc_inspection_cov",
                    help="Feature key: inspection_intensity (inspected units / produced units)",
                )
                st.checkbox(
                    "Operator headcount",
                    value=False,
                    key="mv_inc_headcount",
                    help="Feature key: operator_headcount (distinct operators active per period). "
                    "Simple presence/headcount proxy — does not reflect hours worked. ",
                )
                st.checkbox(
                    "Effective operator FTE",
                    value=False,
                    key="mv_inc_eff_fte",
                    help="Feature key: effective_operator_fte (sum of operator work hours per period / 8h). "
                    "A capacity (FTE) measure that reflects hours worked.",
                )
            with col_vars2:
                st.checkbox(
                    "Avg energy consumption",
                    value=False,
                    key="mv_inc_energy",
                    help="Feature key: avg_energy_consumption (mean energy metric)",
                )
                st.checkbox(
                    "WIP proxy (fraction)",
                    value=False,
                    key="mv_inc_wip_proxy",
                    help="Feature key: wip_proxy (in-progress / total events per period)",
                )
                st.checkbox(
                    "Defect rate %",
                    value=False,
                    key="mv_inc_defect",
                    help="Feature key: defect_rate_pct (fails/total * 100)",
                )

            confirm_setup = st.form_submit_button("Confirm Selection And Continue")

        if confirm_setup:
            selection_snapshot = {
                "inc_defect_rate": bool(st.session_state.get("mv_inc_defect", False)),
                "inc_shift_night": bool(
                    st.session_state.get("mv_inc_shift_night", False)
                ),
                "inc_inspection": bool(
                    st.session_state.get("mv_inc_inspection_cov", False)
                ),
                "inc_headcount": bool(st.session_state.get("mv_inc_headcount", False)),
                "inc_energy": bool(st.session_state.get("mv_inc_energy", False)),
                "inc_wip_proxy": bool(st.session_state.get("mv_inc_wip_proxy", False)),
                "inc_eff_fte": bool(st.session_state.get("mv_inc_eff_fte", False)),
            }
            if not any(selection_snapshot.values()):
                st.warning("Select at least one scenario variable before confirming.")
                st.session_state.pop("mv_confirmed_selection", None)
                st.session_state.pop("mv_confirmed_settings", None)
                confirmed_selection = {}
                confirmed_settings = None
            else:
                confirmed_selection = selection_snapshot.copy()
                mv_metric_label = st.session_state.get(
                    "mv_metric", list(AGG_FRIENDLY.keys())[0]
                )
                confirmed_settings = {
                    "horizon": int(st.session_state.get("mv_horizon", 60)),
                    "horizon_multiplier": float(st.session_state.get("mv_mult", 1.0)),
                    "adapt_horizon": bool(st.session_state.get("mv_adapt", True)),
                    "connect": bool(st.session_state.get("mv_connect", True)),
                    "agg_freq": st.session_state.get("mv_freq", "D"),
                    "agg_metric_label": mv_metric_label,
                    "agg_metric": AGG_FRIENDLY.get(mv_metric_label, "mean"),
                }
                st.session_state["mv_confirmed_selection"] = confirmed_selection
                st.session_state["mv_confirmed_settings"] = confirmed_settings
                # Remove stale slider state for unselected variables
                slider_keys = {
                    "inc_defect_rate": "mv_defect_rate",
                    "inc_shift_night": "mv_shift_night_share",
                    "inc_inspection": "mv_inspection_cov",
                    "inc_headcount": "mv_operator_headcount",
                    "inc_energy": "mv_avg_energy_consumption",
                    "inc_wip_proxy": "mv_wip_proxy",
                    "inc_eff_fte": "mv_effective_operator_fte",
                }
                for key, slider_key in slider_keys.items():
                    if not confirmed_selection.get(key):
                        st.session_state.pop(slider_key, None)
                st.success(
                    "Scenario setup confirmed. Scroll down to input assumptions and run the forecast."
                )

        confirmed_selection = st.session_state.get("mv_confirmed_selection", {})
        confirmed_settings = st.session_state.get("mv_confirmed_settings")
        has_confirmed_variables = bool(confirmed_selection) and any(
            confirmed_selection.values()
        )

        submitted_mv = False
        if confirmed_settings and has_confirmed_variables:
            with st.form("forecast_mv_run_form"):
                st.caption(
                    "Adjust the assumptions for your confirmed scenario variables, then generate the forecast."
                )
                col_inputs1, col_inputs2 = st.columns(2)
                with col_inputs1:
                    if confirmed_selection.get("inc_shift_night"):
                        st.slider(
                            "Assumed Night Shift Share",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(round(night_share_default, 2)),
                            step=0.01,
                            key="mv_shift_night_share",
                        )
                    if confirmed_selection.get("inc_inspection"):
                        st.slider(
                            "Assumed Inspection Coverage (fraction)",
                            min_value=0.0,
                            max_value=1.0,
                            value=cov_default,
                            step=0.01,
                            key="mv_inspection_cov",
                        )
                    if confirmed_selection.get("inc_headcount"):
                        max_headcount = int(max(10, (headcount_default or 0) * 2))
                        st.slider(
                            "Assumed Operator Headcount",
                            min_value=0,
                            max_value=max_headcount,
                            value=int(min(headcount_default, max_headcount)),
                            step=1,
                            key="mv_operator_headcount",
                        )
                    if confirmed_selection.get("inc_eff_fte"):
                        dynamic_cap = (headcount_default or 0) * 2
                        max_eff_fte = float(
                            max(10.0, dynamic_cap if dynamic_cap > 0 else 10.0)
                        )
                        st.slider(
                            "Assumed Effective Operator FTE",
                            min_value=0.0,
                            max_value=max_eff_fte,
                            value=float(min(eff_fte_default or 0.0, max_eff_fte)),
                            step=0.1,
                            key="mv_effective_operator_fte",
                        )
                with col_inputs2:
                    if confirmed_selection.get("inc_energy"):
                        st.slider(
                            "Assumed Avg Energy Consumption",
                            min_value=0.0,
                            max_value=float(max(10.0, energy_default * 3 or 100.0)),
                            value=energy_default,
                            step=0.1,
                            key="mv_avg_energy_consumption",
                        )
                    if confirmed_selection.get("inc_wip_proxy"):
                        st.slider(
                            "Assumed WIP Proxy (fraction)",
                            min_value=0.0,
                            max_value=1.0,
                            value=wip_default,
                            step=0.01,
                            key="mv_wip_proxy",
                        )
                    if confirmed_selection.get("inc_defect_rate"):
                        st.slider(
                            "Assumed Defect Rate %",
                            min_value=0.0,
                            max_value=100.0,
                            value=float(round(default_defect_rate, 2)),
                            step=0.5,
                            key="mv_defect_rate",
                        )
                st.markdown("---")
                submitted_mv = st.form_submit_button(
                    "Run Multivariate Scenario Forecast"
                )
        else:
            st.caption(
                "Confirm the scenario setup to enter assumptions and generate the forecast."
            )

        if submitted_mv and not confirmed_settings:
            st.error("Scenario settings missing. Please confirm the setup again.")
            submitted_mv = False

        if submitted_mv:
            scenario = {
                "horizon": int(confirmed_settings.get("horizon", 60)),
                "adapt_horizon": bool(confirmed_settings.get("adapt_horizon", True)),
                "horizon_multiplier": float(
                    confirmed_settings.get("horizon_multiplier", 1.0)
                ),
                "aggregation": {
                    "freq": confirmed_settings.get("agg_freq", "D"),
                    "metric": confirmed_settings.get("agg_metric", "mean"),
                },
                "included_variables": [],
                "assumptions": {},
            }
            mv_connect = bool(confirmed_settings.get("connect", True))
            mv_agg_metric = scenario["aggregation"]["metric"]

            sel_defect = confirmed_selection.get("inc_defect_rate")
            sel_shift = confirmed_selection.get("inc_shift_night")
            sel_energy = confirmed_selection.get("inc_energy")
            sel_wip = confirmed_selection.get("inc_wip_proxy")
            sel_inspection = confirmed_selection.get("inc_inspection")
            sel_headcount = confirmed_selection.get("inc_headcount")
            sel_eff_fte = confirmed_selection.get("inc_eff_fte")

            if sel_defect:
                fk = FEATURE_LABELS["Defect rate %"]
                scenario["included_variables"].append(fk)
                scenario["assumptions"][fk] = st.session_state.get(
                    "mv_defect_rate", float(round(default_defect_rate, 2))
                )
            if sel_shift:
                fk = FEATURE_LABELS["Night shift share"]
                scenario["included_variables"].append(fk)
                scenario["assumptions"][fk] = st.session_state.get(
                    "mv_shift_night_share", float(round(night_share_default, 2))
                )
            if sel_energy:
                fk = FEATURE_LABELS["Avg energy consumption"]
                scenario["included_variables"].append(fk)
                scenario["assumptions"][fk] = st.session_state.get(
                    "mv_avg_energy_consumption", energy_default
                )
            if sel_wip:
                fk = FEATURE_LABELS["WIP proxy (fraction)"]
                scenario["included_variables"].append(fk)
                scenario["assumptions"][fk] = st.session_state.get(
                    "mv_wip_proxy", wip_default
                )
            if sel_inspection:
                fk = FEATURE_LABELS["Inspection coverage"]
                scenario["included_variables"].append(fk)
                scenario["assumptions"][fk] = st.session_state.get(
                    "mv_inspection_cov", cov_default
                )
            if sel_headcount:
                fk = FEATURE_LABELS["Operator headcount"]
                scenario["included_variables"].append(fk)
                scenario["assumptions"][fk] = int(
                    st.session_state.get("mv_operator_headcount", headcount_default)
                )
            if sel_eff_fte:
                fk = FEATURE_LABELS["Effective operator FTE"]
                scenario["included_variables"].append(fk)
                scenario["assumptions"][fk] = float(
                    st.session_state.get("mv_effective_operator_fte", eff_fte_default)
                )

            st.session_state["multivariate_scenario"] = scenario
            try:
                mv_path = "multivariate_forecasted_data.csv"
                result = run_multivariate_forecast(_tables, scenario, return_meta=True)
                if isinstance(result, tuple):
                    fc_mv, meta = result
                else:
                    fc_mv, meta = result, None
                st.success("Multivariate forecast generated.")
                fig_mv = _build_forecast_line_safe(
                    fc_mv, connect_actual=bool(mv_connect)
                )
                Y_AXIS_TITLES = {
                    "mean": "Average Cycle Time (hrs)",
                    "median": "Typical Cycle Time (hrs)",
                    "sum": "Total Processing Hours (hrs)",
                    "count": "Completed Runs (count)",
                }
                fig_mv.update_yaxes(title=Y_AXIS_TITLES.get(mv_agg_metric, "Value"))
                st.plotly_chart(fig_mv, use_container_width=True)
                _register_chart(
                    "forecast_mv",
                    "Forecast (Scenario)",
                    fig_mv,
                )
                st.session_state["fc_mv_df"] = fc_mv.copy()
                st.download_button(
                    label="Download CSV",
                    data=to_csv_bytes(fc_mv, index=False),
                    file_name=safe_filename(mv_path),
                    mime="text/csv",
                    key="dl_mv_fc",
                )
                st.download_button(
                    label="Download .xlsx",
                    data=to_excel_bytes(fc_mv, index=False, sheet_name="Forecast (MV)"),
                    file_name=safe_filename("multivariate_forecasted_data", ext="xlsx"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_mv_fc_xlsx",
                )
                try:
                    low = (meta or {}).get("low_influence") or []
                    if low:
                        st.warning(
                            "Low influence: the following scenario variable(s) had near-zero standardized coefficients and may have negligible effect: "
                            + ", ".join(low)
                        )
                    hist_low = (meta or {}).get("historical_low_variation") or []
                    hist_stats = (meta or {}).get("historical_variation") or {}
                    if hist_low:
                        msgs = []
                        for feat in hist_low:
                            s = hist_stats.get(feat, {})
                            std = s.get("std")
                            n_unique = s.get("n_unique")
                            msgs.append(
                                f"{feat}: std={std:.4f} n_unique={int(n_unique) if n_unique is not None else 'NA'}"
                            )
                        st.warning(
                            "Low historical variation: the following scenario variable(s) had near-constant historical values and the model had little information to learn their effect. Future assumptions may produce abrupt steps.\n"
                            + "\n".join(msgs)
                        )
                    amb_pairs = (meta or {}).get("measurement_ambiguity") or []
                    for pair in amb_pairs:
                        st.warning(
                            "Measurement caution: these features are measurement-linked and may have ambiguous attribution: "
                            + ", ".join(pair)
                        )
                    col_pairs = (meta or {}).get("collinearity_caution") or []
                    for pair in col_pairs:
                        st.warning(
                            "Collinearity caution: these variables can convey overlapping capacity and may be highly correlated. Consider using one or interpret coefficients with care: "
                            + ", ".join(pair)
                        )
                except Exception:
                    pass
            except ForecastFeatureAdequacyError as fe:
                st.error("One or more selected features are not adequate for modeling.")
                st.warning("\n".join(f"{k}: {v}" for k, v in fe.details.items()))
                st.info("Unselect the inadequate feature(s) and re-run the forecast.")
            except Exception as e:
                if not scenario["included_variables"]:
                    st.error(
                        "Select at least one scenario variable to run a multivariate scenario."
                    )
                else:
                    st.error(f"Multivariate forecasting failed: {e}")


def render_tracking_view(_tables):
    # KPIs
    kpis = compute_all_kpis(_tables)
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

    # Show throughput per hour with one decimal

    col1.metric("Throughput (units/hr)", f"{kpis.get('throughput') * 3600.0:.1f}")
    col2.metric(
        "Throughput (units/day)", f"{(kpis.get('throughput') * 3600.0 * 24):.1f}"
    )
    col3.metric(
        "Throughput (units/week)", f"{(kpis.get('throughput') * 3600.0 * 24 * 7):.1f}"
    )
    col4.metric("WIP (Qty)", f"{kpis.get('wip', 0)}")
    # schedule_efficiency displays as an index (e.g., 0.83x, 1.12x)
    col5.metric(
        "Schedule Efficiency", f"{(kpis.get('schedule_efficiency') or 0.0):.2f}x"
    )
    col6.metric("On-Time Rate", f"{(kpis.get('on_time_rate') or 0.0)*100:.1f}%")
    # Avg cycle time: use backward-compatible key plus explicit mean key
    col7.metric(
        "Avg. Cycle Time (hrs)",
        f"{(kpis.get('avg_cycle_time') or kpis.get('avg_cycle_time_mean') or 0.0):.2f}",
    )
    # Median cycle time (use the median key added in KPI calculator)
    col8.metric(
        "Median Cycle Time (hrs)", f"{(kpis.get('avg_cycle_time_median') or 0.0):.2f}"
    )
    # Removed overall labor metric from the top row (moved to dedicated manpower row below)

    # Silently build a PDF-friendly KPI summary figure and register for export
    try:
        from visualizations.kpi_summary import build_kpi_summary_figure

        kpi_fig = build_kpi_summary_figure(kpis)
        _register_chart("kpi_summary", "KPI Summary", kpi_fig)
    except Exception:
        pass

    # --- Manpower / labor KPIs on their own row ---
    manpower_overall = kpis.get("manpower_utilization_overall", 0.0)
    labor_eff_overall = kpis.get("labor_efficiency_overall", 0.0)

    # Average operator-level utilization (if per-operator rollup available)
    _man_by_op = kpis.get("manpower_by_operator", {}) or {}
    if isinstance(_man_by_op, dict) and _man_by_op:
        try:
            avg_util_ops = sum(
                float(v.get("utilization", 0.0)) for v in _man_by_op.values()
            ) / max(1, len(_man_by_op))
        except Exception:
            avg_util_ops = 0.0
    else:
        avg_util_ops = 0.0

    # Average operator-level efficiency (if per-operator labor efficiency available)
    _lab_by_op = kpis.get("labor_eff_by_operator", {}) or {}
    if isinstance(_lab_by_op, dict) and _lab_by_op:
        try:
            avg_eff_ops = sum(
                float(v.get("efficiency", 0.0)) for v in _lab_by_op.values()
            ) / max(1, len(_lab_by_op))
        except Exception:
            avg_eff_ops = 0.0
    else:
        avg_eff_ops = 0.0

    # Render manpower KPIs in a dedicated row of columns (below the primary KPI row)
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Labor Utilization (overall)", f"{manpower_overall*100:.1f}%")
    mcol2.metric("Labor Efficiency (overall)", f"{labor_eff_overall*100:.1f}%")
    mcol3.metric("Avg Utilization (operators)", f"{avg_util_ops*100:.1f}%")
    mcol4.metric("Avg Efficiency (operators)", f"{avg_eff_ops*100:.1f}%")

    ### BOTTLENECK DETECTION AND FORECASTING ###

    # Separator
    st.markdown("---")

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
        # CSV/XLSX download for Top 3 Bottlenecks
        try:
            bt_rename = {"step_id": "Step", "total_wip": "WIP"}
            st.download_button(
                label="Download CSV",
                data=to_csv_bytes(top3, rename=bt_rename, index=False),
                file_name="top_bottlenecks.csv",
                mime="text/csv",
                key="top3_bottlenecks_csv",
            )
            st.download_button(
                label="Download .xlsx",
                data=to_excel_bytes(
                    top3, rename=bt_rename, index=False, sheet_name="Top 3 Bottlenecks"
                ),
                file_name="top_bottlenecks.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="top3_bottlenecks_xlsx",
            )
        except Exception:
            pass

    # Separator
    st.markdown("---")

    # Gantt charts
    st.subheader("Gantt")
    chart = GanttChart()
    prod_full = _tables.get("production_log", pd.DataFrame())
    process_steps_df = _tables.get("process_steps", pd.DataFrame())
    products = _tables.get("products", pd.DataFrame())

    # Time filtering (applies to both actual and planned anchoring logic)
    g_col1, g_col2, g_col3 = st.columns([1, 1, 2])
    with g_col1:
        gantt_from = st.date_input(
            "From date (Gantt)",
            value=None,
            key="gantt_from",
            help="Filter actual events by start_time >= this date",
        )
    with g_col2:
        gantt_to = st.date_input(
            "To date (Gantt)",
            value=None,
            key="gantt_to",
            help="Filter actual events by start_time <= this date",
        )
    with g_col3:
        preset = st.selectbox(
            "Quick range",
            [
                "(none)",
                "Last 7 days",
                "Last 14 days",
                "Last 30 days",
                "This month",
                "Last month",
                "Year to date",
            ],
            index=0,
            help="Apply a quick preset range; manual date inputs override if changed afterward.",
            key="gantt_preset",
        )
        if preset != "(none)":
            pf, pt = compute_preset_range(preset)
            if gantt_from is None and pf is not None:
                gantt_from = pf.date()
            if gantt_to is None and pt is not None:
                gantt_to = pt.date()

    prod = prod_full.copy()
    if not prod.empty and (gantt_from or gantt_to) and "start_time" in prod.columns:
        prod["start_time"] = pd.to_datetime(
            prod["start_time"], utc=True, errors="coerce"
        )
        if gantt_from:
            g_from_ts = (
                pd.Timestamp(gantt_from).tz_localize("UTC")
                if pd.Timestamp(gantt_from).tz is None
                else pd.Timestamp(gantt_from).tz_convert("UTC")
            )
            prod = prod[prod["start_time"] >= g_from_ts]
        if gantt_to:
            g_to_ts = (
                pd.Timestamp(gantt_to).tz_localize("UTC")
                if pd.Timestamp(gantt_to).tz is None
                else pd.Timestamp(gantt_to).tz_convert("UTC")
            )
            # inclusive end-of-day: add 1 day minus 1 microsecond
            g_to_ts_end = g_to_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            prod = prod[prod["start_time"] <= g_to_ts_end]

    if prod_full is not None and not prod_full.empty and prod.empty:
        st.info("No production events in selected Gantt date range.")

    # Optional name mappings
    product_names = None
    if not products.empty and {"product_id", "name"}.issubset(products.columns):
        product_names = products[["product_id", "name"]].copy()

    step_names = None
    if not process_steps_df.empty and {"step_id", "step_name"}.issubset(
        process_steps_df.columns
    ):
        step_names = process_steps_df[["step_id", "step_name"]].drop_duplicates().copy()

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
        st.plotly_chart(fig_actual, use_container_width=True)
        _register_chart(
            f"gantt_actual_{view_actual}",
            f"Gantt — Actual ({'by run' if view_actual == 'by_run' else 'by step'})",
            fig_actual,
        )
    else:
        st.info(
            "Actual Gantt unavailable (needs start_time/end_time in production_log)."
        )

    fig_planned = chart.planned_gantt(
        process_steps_df,
        production_log=prod,  # enables optional run-based anchoring if desired
        product_names=product_names,
        anchor="run_start",  # anchor planned bars at earliest start_time per (product_id, run_id)
        view=view_planned,
    )
    if fig_planned is not None:
        st.plotly_chart(fig_planned, use_container_width=True)
        _register_chart(
            f"gantt_planned_{view_planned}",
            f"Gantt — Planned ({'by run' if view_planned == 'by_run' else 'by step'})",
            fig_planned,
        )
    else:
        st.info("Planned Gantt unavailable (needs process_steps with estimated_time).")

    # Separator
    st.markdown("---")

    # TIME PER STEP
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
                width="stretch",
            )

            # CSV/XLSX export (inline)
            st.download_button(
                label="Download CSV",
                data=to_csv_bytes(
                    display_df,
                    columns=[
                        "product_label",
                        "step_label",
                        "avg_duration_hours",
                        "median_duration_hours",
                        "std_duration_hours",
                        "events",
                    ],
                    rename={
                        "product_label": "Product",
                        "step_label": "Step",
                        "avg_duration_hours": "Avg Duration (hrs)",
                        "median_duration_hours": "Median (hrs)",
                        "std_duration_hours": "Std Dev (hrs)",
                        "events": "Completed Events",
                    },
                    index=False,
                ),
                file_name="time_per_step.csv",
                mime="text/csv",
            )
            st.download_button(
                label="Download .xlsx",
                data=to_excel_bytes(
                    display_df,
                    columns=[
                        "product_label",
                        "step_label",
                        "avg_duration_hours",
                        "median_duration_hours",
                        "std_duration_hours",
                        "events",
                    ],
                    rename={
                        "product_label": "Product",
                        "step_label": "Step",
                        "avg_duration_hours": "Avg Duration (hrs)",
                        "median_duration_hours": "Median (hrs)",
                        "std_duration_hours": "Std Dev (hrs)",
                        "events": "Completed Events",
                    },
                    index=False,
                    sheet_name="Time per Step",
                ),
                file_name="time_per_step.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with col_tp2:
            if selected_product != "All":
                chart_df = display_df.sort_values(
                    "avg_duration_hours", ascending=False
                )[["step_label", "avg_duration_hours"]].copy()
                if not chart_df.empty:
                    try:
                        fig_bar = build_time_per_step_bar(
                            chart_df, product_label=selected_product
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                        _register_chart(
                            f"time_per_step_{selected_product}",
                            f"{selected_product} Time per Step",
                            fig_bar,
                        )
                    except Exception as e:
                        st.warning(f"Failed to build bar chart: {e}")
            else:
                st.caption("Select a product to see a step-level bar chart.")

    # Separator
    st.markdown("---")

    # Step Durration Histogram
    st.subheader("Step Duration Distribution")
    if agg.empty:
        st.info("Insufficient data to compute time-per-step (need completed events).")
    else:
        # Rebuild raw durations from production_log for distribution
        raw = plog.copy()
        needed_cols = {"start_time", "end_time", "status", "product_id", "step_id"}
        if needed_cols.issubset(raw.columns):
            raw["start_time"] = pd.to_datetime(
                raw["start_time"], utc=True, errors="coerce"
            )
            raw["end_time"] = pd.to_datetime(raw["end_time"], utc=True, errors="coerce")
            raw = raw.dropna(subset=["start_time", "end_time"])
            raw = raw[raw["status"].astype(str).str.lower() == "complete"]
            raw = raw[raw["end_time"] >= raw["start_time"]].copy()
            raw["duration_hours"] = (
                raw["end_time"] - raw["start_time"]
            ).dt.total_seconds() / 3600.0
            # Optional date filter (same as earlier)
            if ds_val:
                ds = pd.to_datetime(ds_val, utc=True, errors="coerce")
                raw = raw[raw["start_time"] >= ds]
            if de_val:
                de = pd.to_datetime(de_val, utc=True, errors="coerce")
                raw = raw[raw["start_time"] <= de]
            # Map labels if available from agg (faster than re-merging)
            label_map = agg.set_index(["product_id", "step_id"])[
                ["product_label", "step_label"]
            ].to_dict("index")
            raw["product_label"] = raw.apply(
                lambda r: label_map.get((r["product_id"], r["step_id"]), {}).get(
                    "product_label", str(r["product_id"])
                ),
                axis=1,
            )
            raw["step_label"] = raw.apply(
                lambda r: label_map.get((r["product_id"], r["step_id"]), {}).get(
                    "step_label", str(r["step_id"])
                ),
                axis=1,
            )

            # Histogram of individual step duration events (distribution) via helper functions
            if raw.empty:
                st.info("No events available for the selected filters.")
            else:
                col_hd1, col_hd2 = st.columns([2, 3])
                with col_hd1:
                    hist_product = st.selectbox(
                        "Product (distribution)",
                        options=["All"] + sorted(raw["product_label"].unique()),
                        key="hist_product",
                    )
                    subset = (
                        raw
                        if hist_product == "All"
                        else raw[raw["product_label"] == hist_product]
                    )
                    step_filter = st.multiselect(
                        "Steps (optional)",
                        options=sorted(subset["step_label"].unique()),
                        default=[],
                        key="hist_steps",
                    )
                    if step_filter:
                        subset = subset[subset["step_label"].isin(step_filter)]
                    bins = st.slider(
                        "Bins", min_value=5, max_value=100, value=30, step=1
                    )
                    log_y = st.checkbox("Log Y-axis", value=False, key="hist_logy")
                with col_hd2:
                    if subset.empty:
                        st.info("No events matching filters.")
                    else:
                        fig_hist = build_step_duration_histogram(
                            subset,
                            bins=bins,
                            log_y=log_y,
                            title="Step Duration Distribution"
                            + ("" if hist_product == "All" else f" — {hist_product}"),
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                        # Register for export
                        try:
                            _register_chart(
                                "hist_step_duration",
                                "Step Duration Distribution",
                                fig_hist,
                            )
                        except Exception:
                            pass
        else:
            st.info("Cannot build histogram (missing required columns).")

    # Separator
    st.markdown("---")

    # Step Dependency Diagram
    st.subheader("Step Dependency Diagram")
    ps_df = _tables.get("process_steps", pd.DataFrame())
    if ps_df is None or ps_df.empty:
        st.info("No process_steps available to render dependencies.")
    else:
        # Streamlit embed width only; diagram layout/styling handled in visualization module
        DAG_EMBED_WIDTH = 1000
        try:
            g, warnings = build_step_dependency_graph(ps_df)
            for msg in warnings or []:
                st.warning(msg)
            st.graphviz_chart(g, width=int(DAG_EMBED_WIDTH))
        except Exception as e:
            st.error(f"Failed to render dependency diagram: {e}")

    # Separator
    st.markdown("---")

    #
    # PROGRESS BARS
    #
    st.subheader("Progress")
    # Per-run progress, current runs only (Not 100%), sorted by run_id
    st.subheader("Current Runs Progress")
    sp = per_step_progress(process_steps_df, prod)
    spr = per_run_progress(
        process_steps_df,
        prod,
        runs=_tables.get("runs", pd.DataFrame()),
    )
    overall = overall_progress_by_product(sp)
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
        if (
            "progress_qty" in disp_run.columns
            and disp_run["progress_qty"].notna().any()
        ):
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
        # CSV/XLSX export for Per-Run Progress
        rename_map = {
            "product_id": "Product",
            "run_id": "Run",
            "planned_qty": "Planned Qty",
            "execution_mode": "Mode",
            "progress_steps_pct": "Steps Progress (%)",
            "progress_qty_pct": "Qty Progress (%)",
            "progress_pct": "Progress (%)",
        }
        st.download_button(
            label="Download CSV",
            data=to_csv_bytes(disp_run[cols].rename(columns=rename_map), index=False),
            file_name="per_run_progress.csv",
            mime="text/csv",
            key="per_run_progress_csv",
        )
        st.download_button(
            label="Download .xlsx",
            data=to_excel_bytes(
                disp_run[cols].rename(columns=rename_map),
                index=False,
                sheet_name="Per-Run Progress",
            ),
            file_name="per_run_progress.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="per_run_progress_xlsx",
        )
    else:
        st.info("No per-run progress available (check run_id and data).")

    # Data preview.
    # NOTE: Let's keep this at the bottom as a footer when adding future data viz content.
    st.divider()
    with st.expander("Preview Data"):
        for name, df in _tables.items():
            st.markdown(f"### {name}")
            st.dataframe(df.head(20), width="stretch")


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
        _tables = read_csv_tables(prefix="mock_")

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

tab_labels = ("Tracking / Status", "Forecasting")
default_tab = st.session_state.get("dashboard_active_tab", tab_labels[0])
default_index = tab_labels.index(default_tab) if default_tab in tab_labels else 0
if hasattr(st, "segmented_control"):
    selected_tab = st.segmented_control(
        "Dashboard view",
        tab_labels,
        default=tab_labels[default_index],
        key="dashboard_active_tab",
    )
else:
    st.warning(
        "Segmented control isn’t available in this Streamlit version; falling back to radio buttons."
    )
    selected_tab = st.radio(
        "Dashboard view",
        tab_labels,
        index=default_index,
        horizontal=True,
        key="dashboard_active_tab_radio",
        label_visibility="collapsed",
    )
    st.session_state["dashboard_active_tab"] = selected_tab

if selected_tab == tab_labels[0]:
    render_tracking_view(_tables)
else:
    render_forecasting_view(_tables)

# Finally render Export (PDF) UI in the sidebar now that all charts can be registered
try:
    _render_export_pdf_ui(export_pdf_container)
except Exception:
    pass
