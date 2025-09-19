"""KPI summary figure for PDF export.

Provides a compact, PDF-friendly summary of top KPIs as a Plotly Table.
"""
from __future__ import annotations

from typing import Mapping, Any
import plotly.graph_objects as go


def _fmt(val: Any, kind: str) -> str:
    try:
        if kind == "throughput_hr":
            return f"{float(val):.5f}"
        if kind == "throughput_day" or kind == "throughput_week":
            return f"{float(val):.1f}"
        if kind == "wip":
            return f"{int(val)}"
        if kind == "schedule_efficiency":
            return f"{float(val or 0.0):.2f}x"
        if kind == "on_time_rate":
            return f"{float(val or 0.0)*100:.1f}%"
    except Exception:
        return "-"
    return "-"


def build_kpi_summary_figure(kpis: Mapping[str, Any], *, title: str = "Key Performance Indicators") -> go.Figure:
    """Create a table-style KPI summary for export.

    Parameters
    ----------
    kpis : Mapping
        Dictionary returned by compute_all_kpis.
    title : str
        Title to display above the table in the PDF.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly Table figure.
    """
    throughput = float(kpis.get("throughput") or 0.0)
    wip = kpis.get("wip", 0)
    schedule_efficiency = kpis.get("schedule_efficiency") or 0.0
    on_time_rate = kpis.get("on_time_rate") or 0.0

    thr_hr = throughput * 3600.0
    thr_day = throughput * 3600.0 * 24
    thr_week = throughput * 3600.0 * 24 * 7

    rows = [
        ("Throughput (units/hr)", _fmt(thr_hr, "throughput_hr")),
        ("Throughput (units/day)", _fmt(thr_day, "throughput_day")),
        ("Throughput (units/week)", _fmt(thr_week, "throughput_week")),
        ("WIP (Qty)", _fmt(wip, "wip")),
        ("Schedule Efficiency", _fmt(schedule_efficiency, "schedule_efficiency")),
        ("On-Time Rate", _fmt(on_time_rate, "on_time_rate")),
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color="#f0f0f0",
                    align="left",
                    font=dict(size=12, color="#222"),
                ),
                cells=dict(
                    values=[[r[0] for r in rows], [r[1] for r in rows]],
                    align=["left", "right"],
                    font=dict(size=12),
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_white",
    )
    return fig


__all__ = ["build_kpi_summary_figure"]
