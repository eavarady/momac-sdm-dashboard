"""Forecast line chart utilities.

Provides a single entry point `build_forecast_line` to render a standardized
forecast DataFrame (historical + future) returned by time_series_forecast or
other models. Keeps visualization logic separate from the Streamlit layer.

Expected input schema (flexible, missing columns tolerated with graceful skips):
    ds : datetime index column
    y  : (optional) historical actual values; plotted as solid line / markers
    yhat : model prediction for both historic (in-sample) + future points
    yhat_lower, yhat_upper : optional prediction interval bounds
    model : string label of model used (e.g., 'prophet', 'baseline-mean')

Design choices:
  * All styling centralized here so future dashboards (e.g., diagnostics) reuse
    consistent color palette and hover formatting.
  * Bounds rendered as a translucent band when show_bounds=True and both lower
    and upper columns exist.
  * Future horizon visually distinguished (vertical line at last actual y).
"""

from __future__ import annotations

from typing import Optional, Sequence, Mapping
import pandas as pd
import plotly.graph_objects as go

DEFAULT_COLORS = {
    "actual": "#1f77b4",
    "forecast": "#ff7f0e",
    "interval": "rgba(255,127,14,0.15)",
}


def build_forecast_line(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    show_history: bool = True,
    show_bounds: bool = True,
    markers: bool = True,
    color_map: Optional[Mapping[str, str]] = None,
    height: int = 420,
    width: Optional[int] = None,
) -> go.Figure:
    """Create a Plotly figure for forecast data.

    Parameters
    ----------
    df : DataFrame
        Forecast frame. Must contain at least columns ['ds', 'yhat'].
    title : str, optional
        Chart title. If None, auto-generated from model column when present.
    show_history : bool
        Plot actual y values when available.
    show_bounds : bool
        Show prediction interval if yhat_lower & yhat_upper exist.
    markers : bool
        Draw markers on actual points.
    color_map : mapping
        Override default colors (keys: 'actual','forecast','interval').
    height, width : int
        Figure dimensions (width None -> auto / container width).
    """
    if df is None or df.empty:
        raise ValueError("Empty forecast DataFrame provided.")

    required = {"ds", "yhat"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Forecast DataFrame missing required columns: {missing}")

    colors = {**DEFAULT_COLORS, **(color_map or {})}

    data = df.copy()
    # Ensure datetime ordering & strip timezone to avoid Plotly / pandas ops issues
    data["ds"] = pd.to_datetime(data["ds"], errors="coerce", utc=True)
    if getattr(data["ds"].dt, "tz", None) is not None:
        data["ds"] = data["ds"].dt.tz_convert("UTC").dt.tz_localize(None)
    data = data.dropna(subset=["ds"]).sort_values("ds")
    # Convert to Python datetime objects to avoid deprecated integer arithmetic inside libs
    # Use list of Timestamp objects (Plotly handles these) avoiding deprecated to_pydatetime semantics
    ds_py = data["ds"].tolist()

    fig = go.Figure()

    # Interval band (must plot lower first, then upper with fill='tonexty')
    has_bounds = show_bounds and {"yhat_lower", "yhat_upper"}.issubset(data.columns)
    if has_bounds:
        fig.add_trace(
            go.Scatter(
                name="Lower Bound",
                x=ds_py,
                y=data["yhat_lower"],
                line=dict(width=0),
                mode="lines",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                name="Prediction Interval",
                x=ds_py,
                y=data["yhat_upper"],
                line=dict(width=0),
                mode="lines",
                fill="tonexty",
                fillcolor=colors["interval"],
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            name="Forecast",
            x=ds_py,
            y=data["yhat"],
            mode="lines",
            line=dict(color=colors["forecast"], width=2),
        )
    )

    # Historical actuals
    if show_history and "y" in data.columns:
        fig.add_trace(
            go.Scatter(
                name="Actual",
                x=ds_py,
                y=data["y"],
                mode="lines+markers" if markers else "lines",
                line=dict(color=colors["actual"], width=1.5),
                marker=dict(size=5) if markers else None,
            )
        )

    # Vertical separator at last actual point (custom shape to avoid add_vline Timestamp arithmetic)
    if "y" in data.columns:
        hist_mask = data["y"].notna()
        if hist_mask.any() and hist_mask.sum() < len(data):
            last_hist_time = data.loc[hist_mask, "ds"].max()
            # Add a vertical line shape
            fig.add_shape(
                type="line",
                x0=last_hist_time,
                x1=last_hist_time,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="#555", width=1, dash="dot"),
            )
            # Add annotation near the top
            fig.add_annotation(
                x=last_hist_time,
                y=1,
                xref="x",
                yref="paper",
                text="Forecast start",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=10, color="#555"),
            )

    model_label = None
    if "model" in data.columns:
        model_label = data["model"].dropna().unique()
        if len(model_label) == 1:
            model_label = model_label[0]
        else:
            model_label = None

    if title is None:
        title = f"Forecast ({model_label})" if model_label else "Forecast"

    fig.update_layout(
        title=title,
        height=height,
        width=width,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(title="Date", showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(title="Value", showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    return fig


__all__ = ["build_forecast_line"]
