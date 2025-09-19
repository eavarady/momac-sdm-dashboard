"""Time-per-step visualization utilities.

Provides reusable Plotly chart builders for the Time-per-Step KPI section:

1. Bar chart of average duration per step for a single product.
2. Histogram / distribution of individual step event durations with optional filters.

Keeping these builders here keeps `dashboard/app.py` focused on light UI wiring
and avoids duplicating data prep logic.
"""

from __future__ import annotations

from typing import Optional, Iterable, Sequence
import pandas as pd
import plotly.express as px

DEFAULT_HIST_BINS = 30


def prepare_step_duration_events(
    production_log: pd.DataFrame,
    *,
    date_start: Optional[pd.Timestamp | str] = None,
    date_end: Optional[pd.Timestamp | str] = None,
    required: Sequence[str] = (
        "start_time",
        "end_time",
        "status",
        "product_id",
        "step_id",
    ),
) -> pd.DataFrame:
    """Return per-event completed step durations (hours) with basic validation.

    Filters to completed events (status == 'complete'), ensures end >= start, and
    applies optional start_time filtering (inclusive bounds). Returns DataFrame with:
      product_id, step_id, start_time, end_time, duration_hours
    Rows with invalid/missing times are dropped.
    """
    if production_log is None or production_log.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "step_id",
                "start_time",
                "end_time",
                "duration_hours",
            ]
        )

    df = production_log.copy()
    if not set(required).issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "product_id",
                "step_id",
                "start_time",
                "end_time",
                "duration_hours",
            ]
        )

    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "step_id",
                "start_time",
                "end_time",
                "duration_hours",
            ]
        )

    # Completed + valid ordering
    df = df[df["status"].astype(str).str.lower() == "complete"].copy()
    df = df[df["end_time"] >= df["start_time"]]
    if df.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "step_id",
                "start_time",
                "end_time",
                "duration_hours",
            ]
        )

    # Date filtering on start_time
    if date_start is not None:
        ds = pd.to_datetime(date_start, utc=True, errors="coerce")
        if ds is not None and not pd.isna(ds):
            df = df[df["start_time"] >= ds]
    if date_end is not None:
        de = pd.to_datetime(date_end, utc=True, errors="coerce")
        if de is not None and not pd.isna(de):
            df = df[df["start_time"] <= de]
    if df.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "step_id",
                "start_time",
                "end_time",
                "duration_hours",
            ]
        )

    df["duration_hours"] = (
        df["end_time"] - df["start_time"]
    ).dt.total_seconds() / 3600.0
    return df[["product_id", "step_id", "start_time", "end_time", "duration_hours"]]


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
        f"Average Duration by Step — {product_label}"
        if product_label
        else "Average Duration by Step"
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


def build_step_duration_histogram(
    events_df: pd.DataFrame,
    *,
    product_label_filter: Optional[str] = None,
    step_labels_filter: Optional[Iterable[str]] = None,
    label_lookup: Optional[pd.DataFrame] = None,
    nbins: int = DEFAULT_HIST_BINS,
    # Backward-compatible alias: callers may pass bins= (earlier inline usage)
    bins: Optional[int] = None,
    log_y: bool = False,
    title: Optional[str] = None,
) -> Optional["px.Figure"]:
    """Build a histogram of per-event step durations.

    Parameters
    ----------
    events_df : DataFrame
        Output of prepare_step_duration_events plus optional product/step labels.
    product_label_filter : str | None
        If provided, restrict to rows with this product_label.
    step_labels_filter : iterable[str] | None
        If provided and non-empty, restrict to those step_label values.
    label_lookup : DataFrame | None
        DataFrame with columns (product_id, step_id, product_label, step_label) to merge
        if events_df lacks label columns.
    nbins : int
        Histogram bin count.
    log_y : bool
        Whether to use logarithmic y-axis.
    """
    if events_df is None or events_df.empty:
        return None

    df = events_df.copy()
    # Ensure duration_hours present
    if "duration_hours" not in df.columns:
        return None

    # Attach labels if needed
    need_labels = {"product_label", "step_label"} - set(df.columns)
    if need_labels and label_lookup is not None and not label_lookup.empty:
        lk_cols = {"product_id", "step_id", "product_label", "step_label"}
        if lk_cols.issubset(label_lookup.columns):
            df = df.merge(
                label_lookup[list(lk_cols)].drop_duplicates(),
                on=["product_id", "step_id"],
                how="left",
            )
    # Fallback labels
    if "product_label" not in df.columns:
        df["product_label"] = df["product_id"].astype(str)
    if "step_label" not in df.columns:
        df["step_label"] = df["step_id"].astype(str)

    if product_label_filter and product_label_filter != "All":
        df = df[df["product_label"] == product_label_filter]
    if step_labels_filter:
        step_labels = list(step_labels_filter)
        if step_labels:
            df = df[df["step_label"].isin(step_labels)]

    if df.empty:
        return None

    # Prefer explicit bins alias if provided
    nbins_final = bins if bins is not None else nbins

    # Use provided title or construct default
    base_title = (
        title
        if title is not None
        else "Step Duration Distribution"
        + (
            ""
            if not product_label_filter or product_label_filter == "All"
            else f" — {product_label_filter}"
        )
    )

    fig = px.histogram(
        df,
        x="duration_hours",
        color="step_label",
        nbins=nbins_final,
        barmode="overlay",
        opacity=0.6,
        title=base_title,
        labels={"duration_hours": "Duration (hrs)", "step_label": "Step"},
    )
    fig.update_layout(yaxis_type="log" if log_y else "linear")
    return fig


__all__ = [
    "build_time_per_step_bar",
    "prepare_step_duration_events",
    "build_step_duration_histogram",
    "DEFAULT_HIST_BINS",
]
