from __future__ import annotations
from io import BytesIO
from typing import Dict, Iterable, Tuple

import plotly.graph_objects as go
from PyPDF2 import PdfMerger


def figures_to_pdf_bytes(figures: Dict[str, go.Figure] | Iterable[Tuple[str, go.Figure]], *, width: int | None = None, height: int | None = None) -> bytes:
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
            pdf_bytes = fig.to_image(format="pdf", width=width, height=height)
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
