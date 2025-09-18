"""Progress visualization utilities.

Reusable Plotly bar charts for exporting progress-related visuals.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def build_current_runs_progress_bar(current_runs: pd.DataFrame) -> "px.Figure":
    """Horizontal bar chart for current runs progress.

    Expects columns: run_id, progress (0..1)
    """
    if current_runs is None or current_runs.empty:
        raise ValueError("Empty DataFrame for current runs progress chart.")
    req = {"run_id", "progress"}
    missing = req - set(current_runs.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for current runs chart: {missing}"
        )
    cr_df = current_runs.copy()
    cr_df["progress_pct"] = (cr_df["progress"].astype(float) * 100.0).round(1)
    fig = px.bar(
        cr_df.sort_values(["progress_pct", "run_id"]),
        x="progress_pct",
        y="run_id",
        orientation="h",
        labels={"progress_pct": "Progress (%)", "run_id": "Run"},
        title="Current Runs Progress",
        text="progress_pct",
    )
    fig.update_layout(showlegend=False)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_xaxes(range=[0, 100])
    return fig


def build_overall_progress_by_product_bar(overall: pd.DataFrame) -> "px.Figure":
    """Horizontal bar chart for overall progress by product.

    Expects columns: product_id, overall_progress (0..1)
    """
    if overall is None or overall.empty:
        raise ValueError("Empty DataFrame for overall progress chart.")
    req = {"product_id", "overall_progress"}
    missing = req - set(overall.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for overall progress chart: {missing}"
        )
    ov_df = overall.copy()
    ov_df["overall_progress_pct"] = (
        ov_df["overall_progress"].astype(float) * 100.0
    ).round(1)
    fig = px.bar(
        ov_df.sort_values(["overall_progress_pct", "product_id"]),
        x="overall_progress_pct",
        y="product_id",
        orientation="h",
        labels={
            "overall_progress_pct": "Progress (%)",
            "product_id": "Product",
        },
        title="Overall Progress by Product",
        text="overall_progress_pct",
    )
    fig.update_layout(showlegend=False)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_xaxes(range=[0, 100])
    return fig


__all__ = [
    "build_current_runs_progress_bar",
    "build_overall_progress_by_product_bar",
]
