import os

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

import streamlit as st
import pandas as pd
from adapters.csv_adapter import read_csv_tables, get_last_load_stats
from adapters.sheets_adapter import read_sheets
from kpi.kpi_calculator import compute_all_kpis
from sdm_bottlenecks.bottleneck_detector import detect_bottleneck, top_bottlenecks

st.set_page_config(page_title="MOMAC SDM Dashboard", layout="wide")

st.title("MOMAC SDM Dashboard")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Select source", ["CSV", "Google Sheets"], index=0)
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
    else:
        st.caption("Reading from local CSVs in data/")

# Load data (fast-fail): if any table is invalid, surface the error and stop
try:
    if source == "CSV":
        _tables = read_csv_tables()
    else:
        if not spreadsheet_id:
            st.stop()
        _tables = read_sheets(spreadsheet_id=spreadsheet_id, title_map=title_map)
except Exception as e:
    st.error(f"Data load failed: {e}")
    if source == "CSV":
        stats = get_last_load_stats()
        st.subheader("Last load stats")
        st.json(stats)
    st.stop()

# KPIs
kpis = compute_all_kpis(_tables)
col1, col2, col3, col4, col5 = st.columns(5)

# Show throughput per hour with one decimal

col1.metric("Throughput (units/hr)", f"{kpis.get('throughput') * 3600.0:.5f}")
col2.metric("Throughput (units/day)", f"{(kpis.get('throughput') * 3600.0 * 24):.1f}")
col3.metric(
    "Throughput (units/week)", f"{(kpis.get('throughput') * 3600.0 * 24 * 7):.1f}"
)
col4.metric("WIP (Qty)", f"{kpis.get('wip', 0)}")
# schedule_efficiency displays as an index (e.g., 0.83x, 1.12x)
col5.metric("Schedule Efficiency", f"{(kpis.get('schedule_efficiency') or 0.0):.2f}x")
# col5.metric("On-Time Rate", f"{(kpis.get('on_time_rate') or 0.0)*100:.1f}%")

# Bottleneck
bn = detect_bottleneck(
    _tables.get("process_steps", pd.DataFrame()),
    _tables.get("production_log", pd.DataFrame()),
)
st.subheader("Largest bottleneck")
st.write(bn or "No bottleneck detected.")
st.subheader("Top 3 Bottlenecks by WIP")
prod = _tables.get("production_log", pd.DataFrame())
top3 = top_bottlenecks(prod, top_n=3)
if top3.empty:
    st.write("No in-progress work detected.")
else:
    st.dataframe(top3, use_container_width=True)

# Data preview
with st.expander("Preview Data"):
    for name, df in _tables.items():
        st.markdown(f"### {name}")
        st.dataframe(df.head(20))
