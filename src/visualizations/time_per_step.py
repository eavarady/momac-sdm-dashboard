"""Time-per-step visualization utilities.

Provides a reusable Plotly bar chart builder for the Time-per-Step KPI section.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
import plotly.express as px


def build_time_per_step_bar(
    df: pd.DataFrame,
    *,
    product_label: Optional[str] = None,
) -> "px.Figure":
    """Build a bar chart of average duration per step for a single product.

    Expects columns: step_label, avg_duration_hours
    Optional: product_label used only for chart title if provided.
    """
    if df is None or df.empty:
        raise ValueError("Empty DataFrame for time-per-step bar chart.")

    req = {"step_label", "avg_duration_hours"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for time-per-step chart: {missing}")

    chart_df = df.sort_values("avg_duration_hours", ascending=False)
    title = (
        f"Average Duration by Step — {product_label}" if product_label else "Average Duration by Step"
    )
    fig = px.bar(
        chart_df,
        x="step_label",
        y="avg_duration_hours",
        color="step_label",
        title=title,
        labels={
            "step_label": "Step",
            "avg_duration_hours": "Avg Duration (hrs)",
        },
    )
    fig.update_layout(showlegend=False)
    return fig


__all__ = ["build_time_per_step_bar"]
