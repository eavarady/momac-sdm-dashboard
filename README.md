### Excel (.xlsx)

Prepare a workbook with one sheet per table name
(`machines`, `production_lines`, `products`, `operators`, `process_steps`,
`production_log`, `machine_metrics`, `quality_checks`).

In the dashboard sidebar, choose **Excel** as the source and either:

- Upload a `.xlsx` file, or
- Enter a local path to a workbook.

Optional:
- Provide a JSON title map if your sheet names differ
  (e.g. `{"production_log": "Prod Log"}`).
- Specify how many header rows to skip (`skiprows`) if your data starts below
  the first row.

Validation and normalization are identical to CSV/Sheets.  
Requires `openpyxl` (already in requirements).

## Google Sheets (OAuth, Desktop App)

To read data from Google Sheets using OAuth (no service account required):

1) Create OAuth Client ID (Desktop app)
	- Go to Google Cloud Console > APIs & Services > Enabled APIs: enable "Google Sheets API".
	- OAuth consent screen: set User Type to External, add your Google account as a Test user.
	- Credentials > Create Credentials > OAuth client ID > Application type: Desktop app.
	- Download the JSON and save it as `.secrets/client_secret.json` at the repo root.

2) First-time authorization (browser prompt)
	- On first run, a browser opens for consent; a token is saved to `.secrets/token.json` for future runs.

3) Sheet access
	- Ensure the Google account you use for consent has at least Viewer access to the target spreadsheet.
	- Use the spreadsheet ID (the long ID in the URL) when calling the adapter.

Where files go
- Place the OAuth client JSON at: `.secrets/client_secret.json`
- The adapter will cache the user token at: `.secrets/token.json`

Note
- This uses read-only scope: `https://www.googleapis.com/auth/spreadsheets.readonly`.
- For headless/CI usage, prefer a service account instead.

## Production Targets (Optional)

In addition to CSV/Excel/Sheets input for core tables, you may provide an optional production_targets table.
This defines target output quantities per step, enabling progress bars to be measured against planned goals rather than only actual activity.

Schema:
	•	product_id – links to products table
	•	step_id – links to process_steps table
	•	target_qty – positive integer quantity goal

Behavior:
	•	If production_targets is present, progress for each step is calculated as completed_qty / target_qty (clipped to 100%).
	•	If absent, the system falls back to the default ratio completed_qty / (completed_qty + in_progress_qty).
	•	Validation ensures all (product_id, step_id) pairs referenced in targets exist in process_steps.

This table is optional; the dashboard will continue to function without it.

# MOMAC SDM Dashboard

Purpose
The SDM app is a lightweight, modular, and flexible Python dashboard for defining, tracking, and analyzing assembly and manufacturing workflows. It allows managers to define processes as sequences of steps, track KPIs, detect bottlenecks, and make informed decisions about manpower and timelines.

Problem Statement
Assembly managers and C-suite personnel currently lack a simple, ERP-agnostic tool to define, monitor, and analyze manufacturing workflows. This makes it difficult to track key performance indicators (KPIs), identify process bottlenecks in real-time, and accurately forecast project completion timelines.

Target Audience
The ideal users are MOMAC assembly managers and C-suite personnel (2–3 users) who need a high-level overview of workflow progress without the complexity of a full-scale ERP system.

Strategic Objective
MOMAC will be among the first in Saudi Arabia to implement Software-Defined Manufacturing, transforming production into a flexible, intelligent, and software-driven capability. By harnessing digital twins, AI-defined workflows, and ontology-based models, MOMAC will pioneer a new industrial standard aligned with Vision 2030, ensuring adaptability, efficiency, and strategic competitiveness in the global market.

## Project Structure

See the repo for this layout:

```
momac-sdm-dashboard/
├── data/                    # CSV datasets
├── notebooks/               # Prototyping & analysis notebooks
├── scripts/                 # Utilities (mock data, deploy)
├── src/                     # Production code by module
├── tests/                   # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Quickstart

1) Create a virtual environment and install dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) (Optional) Regenerate mock CSV data into `data/`.

```powershell
python .\scripts\generate_mock_data.py
```

3) Run tests.

```powershell
pytest -q
```

4) Launch the Streamlit dashboard.

```powershell
streamlit run src\dashboard\app.py
```

## Modules Overview

- adapters: CSV/Excel/Google Sheets adapters for data sources.
- workflow: Workflow data model, step definitions, and validators.
- kpi: KPI calculations (throughput, WIP, cycle time, on-time rate).
- bottleneck: Simple bottleneck detection from logs and step estimates.
- dashboard: Streamlit UI for exploration and KPIs.
- export: CSV exporter utilities.
- utils: Common helpers.

## Notes

- The current storage is CSV-based. You can later swap adapters for SQL/ERP.
- The structure is future-proof for AI inference, SQL backends, and ERP integration.
