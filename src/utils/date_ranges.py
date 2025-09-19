from __future__ import annotations
import pandas as pd
from typing import Optional, Tuple

PresetRange = Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]


def compute_preset_range(name: str) -> PresetRange:
    """Return (date_from, date_to) date objects (UTC date precision) for a given preset name.

    The returned values are naive dates (no time component) represented as pandas Timestamp
    with UTC normalization where applicable, but only the date part is meaningful.

    Supported presets:
      - Last 7 days
      - Last 14 days
      - Last 30 days
      - This month
      - Last month
      - Year to date
    Unknown / '(none)' returns (None, None).
    """
    # Get current UTC date safely regardless of pandas version behavior.
    # pd.Timestamp.utcnow() may return tz-aware in some versions; using now(tz='UTC') avoids tz_localize errors.
    today = pd.Timestamp.now(tz="UTC").date()
    first_of_this_month = today.replace(day=1)
    if name == "Last 7 days":
        return (pd.Timestamp(today - pd.Timedelta(days=6)), pd.Timestamp(today))
    if name == "Last 14 days":
        return (pd.Timestamp(today - pd.Timedelta(days=13)), pd.Timestamp(today))
    if name == "Last 30 days":
        return (pd.Timestamp(today - pd.Timedelta(days=29)), pd.Timestamp(today))
    if name == "This month":
        return (pd.Timestamp(first_of_this_month), pd.Timestamp(today))
    if name == "Last month":
        first_prev = (first_of_this_month - pd.Timedelta(days=1)).replace(day=1)
        last_prev = first_of_this_month - pd.Timedelta(days=1)
        return (pd.Timestamp(first_prev), pd.Timestamp(last_prev))
    if name == "Year to date":
        first_of_year = today.replace(month=1, day=1)
        return (pd.Timestamp(first_of_year), pd.Timestamp(today))
    return (None, None)


__all__ = ["compute_preset_range"]
