from __future__ import annotations
from io import BytesIO
from typing import Dict, Iterable, Tuple

import plotly.graph_objects as go
from PyPDF2 import PdfMerger


def _prepare_for_export(fig: go.Figure) -> go.Figure:
    """Clone and style a figure for high-quality PDF export.

    Ensures a clean white background, readable fonts, consistent margins,
    and slightly thicker lines for legibility in a static document.
    """
    f = go.Figure(fig)  # clone to avoid mutating the on-screen chart
    # Apply a neutral template if none set (Streamlit interactive theme doesn't carry to Kaleido)
    try:
        tmpl = f.layout.template.template if getattr(f.layout, "template", None) else None
    except Exception:
        tmpl = None
    if not tmpl:
        f.update_layout(template="plotly_white")

    # Consistent font + margins + backgrounds
    f.update_layout(
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#222"),
        margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.03, x=0),
    )
    return f


def figures_to_pdf_bytes(
    figures: Dict[str, go.Figure] | Iterable[Tuple[str, go.Figure]],
    *,
    width: int | None = 1200,
    height: int | None = 720,
) -> bytes:
    """
    Render Plotly figures to a single multi-page PDF and return as bytes.

    Parameters
    ----------
    figures : mapping or iterable
        Either a dict {label: figure} or an iterable of (label, figure) pairs.
    width, height : optional export dimensions in pixels (Plotly/Kaleido will scale to page).

    Returns
    -------
    bytes
        The merged PDF content as bytes.
    """
    if isinstance(figures, dict):
        items = list(figures.items())
    else:
        items = list(figures)

    if not items:
        raise ValueError("No figures provided for PDF export.")

    merger = PdfMerger()
    temp_buffers: list[BytesIO] = []

    try:
        for label, fig in items:
            if not isinstance(fig, go.Figure):
                raise TypeError(f"Item '{label}' is not a Plotly Figure.")
            styled = _prepare_for_export(fig)
            # Respect figure-level explicit sizes if provided; else fall back to defaults
            export_width = width or getattr(styled.layout, "width", None) or 1200
            export_height = height or getattr(styled.layout, "height", None) or 720
            pdf_bytes = styled.to_image(format="pdf", width=export_width, height=export_height)
            buf = BytesIO(pdf_bytes)
            temp_buffers.append(buf)
            merger.append(buf)
        out = BytesIO()
        merger.write(out)
        return out.getvalue()
    finally:
        try:
            merger.close()
        except Exception:
            pass
        for b in temp_buffers:
            try:
                b.close()
            except Exception:
                pass


__all__ = ["figures_to_pdf_bytes"]
