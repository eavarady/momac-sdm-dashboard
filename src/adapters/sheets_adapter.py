from __future__ import annotations
from typing import Dict, Optional, Mapping
from pathlib import Path
import pandas as pd
import gspread
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# Reuse CSV adapter normalization/validation (single source of truth)
from adapters.csv_adapter import (
    _normalize_production_log,
    _normalize_process_steps,
    _normalize_machine_metrics,
    _normalize_quality_checks,
    _normalize_dimension_table,
)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
SECRETS_DIR = Path(".secrets")
# Place the OAuth client JSON you downloaded from Google Cloud Console here as '.secrets/client_secret.json'.
# How to get it (short): Console > APIs & Services > Credentials > Create Credentials > OAuth client ID > Desktop app > Download JSON.
CLIENT_SECRET = SECRETS_DIR / "client_secret.json"
# After first browser consent, the adapter writes a cached token to '.secrets/token.json' so you won't be prompted again.
TOKEN_FILE = SECRETS_DIR / "token.json"

TABLES_ORDER = [
    "machines",
    "production_lines",
    "products",
    "operators",
    "process_steps",
    "production_log",
    "machine_metrics",
    "quality_checks",
]


def _get_gspread_client() -> gspread.Client:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    creds: Optional[Credentials] = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CLIENT_SECRET.exists():
                raise RuntimeError(
                    f"Google OAuth client secret not found at {CLIENT_SECRET}. "
                    "Create an OAuth Client ID (Desktop app) and place the JSON here."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
    return gspread.authorize(creds)


def _read_tab(gc: gspread.Client, spreadsheet_id: str, title: str) -> pd.DataFrame:
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(title)
    values = ws.get_all_values()
    if not values or not values[0]:
        return pd.DataFrame()
    headers = [h.strip() for h in values[0]]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)
    return df.dropna(how="all").reset_index(drop=True)


def read_sheets(
    spreadsheet_id: str, title_map: Optional[Mapping[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Strict read: fast-fail on missing tabs/columns or normalization errors.
    Normalization/validation is identical to CSV adapter.
    """
    gc = _get_gspread_client()
    tables: Dict[str, pd.DataFrame] = {}
    for name in TABLES_ORDER:
        title = (title_map or {}).get(name, name)
        df = _read_tab(gc, spreadsheet_id, title)

        # Delegate to the same normalizers/validators as CSV adapter
        if name == "production_log":
            tables[name] = _normalize_production_log(df)
        elif name == "process_steps":
            tables[name] = _normalize_process_steps(df)
        elif name == "machine_metrics":
            tables[name] = _normalize_machine_metrics(df)
        elif name == "quality_checks":
            tables[name] = _normalize_quality_checks(df)
        elif name == "machines":
            tables[name] = _normalize_dimension_table(
                df, ["machine_id", "line_id", "type", "status"], pk="machine_id"
            )
        elif name == "production_lines":
            tables[name] = _normalize_dimension_table(
                df, ["line_id", "name", "shift"], pk="line_id"
            )
        elif name == "products":
            tables[name] = _normalize_dimension_table(
                df, ["product_id", "name", "category", "spec_version"], pk="product_id"
            )
        elif name == "operators":
            tables[name] = _normalize_dimension_table(
                df, ["operator_id", "name", "role"], pk="operator_id"
            )
        else:
            raise ValueError(f"Unsupported table: {name}")
    return tables
