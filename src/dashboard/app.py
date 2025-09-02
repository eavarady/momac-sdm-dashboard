import streamlit as st
import pandas as pd
from adapters.csv_adapter import read_csv_tables
from kpi.kpi_calculator import compute_all_kpis
from sdm_bottlenecks.bottleneck_detector import detect_bottleneck, top_bottlenecks

st.set_page_config(page_title="MOMAC SDM Dashboard", layout="wide")

st.title("MOMAC SDM Dashboard")

with st.sidebar:
    st.header("Data Source")
    st.caption("Reading from local CSVs in data/")

# Load data
_tables = read_csv_tables()

# KPIs
kpis = compute_all_kpis(_tables)
col1, col2, col3 = st.columns(3)
col1.metric("Throughput (Qty)", f"{kpis['throughput']:.0f}")
col2.metric("WIP (Qty)", f"{kpis['wip']}")
col3.metric("On-Time Rate", f"{kpis['on_time_rate']*100:.1f}%")

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
