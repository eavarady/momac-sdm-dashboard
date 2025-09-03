from __future__ import annotations
from typing import Optional
import pandas as pd
import plotly.express as px


class GanttChart:
    """Build Plotly Gantt charts from normalized tables.

    Methods return Plotly Figure objects (or None if insufficient data).
    Keep this module UI-agnostic; Streamlit rendering happens in the dashboard.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _now_utc() -> pd.Timestamp:
        """Return a UTC-aware Timestamp robustly across pandas versions.

        If utcnow() returns a naive timestamp, localize to UTC.
        If it returns tz-aware, convert to UTC (no-op if already UTC).
        """
        ts = pd.Timestamp.utcnow()
        if ts.tz is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def actual_gantt(self, production_log: pd.DataFrame) -> Optional["px.Figure"]:
        """Actual Gantt using start_time/end_time from production_log.

        Requirements: columns [product_id, step_id, status, start_time], optional end_time.
        In-progress rows with no end_time get a temporary end at 'now' for visualization only.
        """
        if production_log is None or production_log.empty:
            return None
        required = {"product_id", "step_id", "status", "start_time"}
        if not required.issubset(production_log.columns):
            return None

        df = production_log.copy()
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
        if "end_time" in df.columns:
            df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
        else:
            df["end_time"] = pd.NaT

        # For in-progress rows, fill end_time with now (for visualization only)
        now_utc = self._now_utc()
        mask_inp = (df["status"].astype(str).str.lower() == "in_progress") & (
            df["end_time"].isna()
        )
        df.loc[mask_inp, "end_time"] = now_utc

        # Keep rows with valid intervals
        df = df.dropna(subset=["start_time", "end_time"]).reset_index(drop=True)
        if df.empty:
            return None

        # Aggregate to one bar per (product_id, step_id, status)
        agg = (
            df.groupby(["product_id", "step_id", "status"], dropna=False)
            .agg(start_time=("start_time", "min"), end_time=("end_time", "max"))
            .reset_index()
        )
        agg["task"] = agg["product_id"].astype(str) + " • " + agg["step_id"].astype(str)

        fig = px.timeline(
            agg,
            x_start="start_time",
            x_end="end_time",
            y="task",
            color="status",
            hover_data=["product_id", "step_id", "status"],
            title="Actual Gantt (by step)",
        )
        fig.update_yaxes(autorange="reversed")
        return fig

    def planned_gantt(self, process_steps: pd.DataFrame) -> Optional["px.Figure"]:
        """Planned Gantt using process_steps estimated_time (hours).

        Placeholder sequential schedule per product: sorts by (product_id, step_id),
        treats missing estimated_time as 1h, minimum bar length 0.25h.
        """
        # PLACEHOLDER
        return None
