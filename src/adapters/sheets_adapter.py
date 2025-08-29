from typing import Dict
import pandas as pd

# Placeholder for a Google Sheets adapter.
# In production, implement OAuth and use gspread or Google Sheets API
# to fetch sheets into DataFrames mirroring the CSV adapter interface.


def read_sheets(spreadsheet_id: str) -> Dict[str, pd.DataFrame]:
    """Placeholder. Google Sheets adapter not implemented yet."""
    raise NotImplementedError("Google Sheets adapter not implemented yet.")
