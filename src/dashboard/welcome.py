import streamlit as st

st.set_page_config(page_title="MOMAC SDM Dashboard", layout="wide")

st.title("Welcome to MOMAC SDM Dashboard")

st.write(
    """
This dashboard provides comprehensive analytics for MOMAC SDM operations, including:
- Key Performance Indicator (KPI) calculations for cycle time, throughput, labor efficiency, and more
- Bottleneck detection in production processes
- Time series forecasting and multivariate forecasting for production metrics
- Interactive visualizations including Gantt charts, progress charts, and dependency diagrams
- Data export capabilities to CSV, Excel, and PDF formats
- Workflow validation and dependency checking
"""
)

col1, col2 = st.columns(2)

with col1:
    if st.button("LOG IN", use_container_width=True):
        st.write(
            "Login functionality not implemented yet. This would connect to the company's actual data systems."
        )

with col2:
    if st.button("DEMO", use_container_width=True):
        st.switch_page("pages/demo_app.py")
