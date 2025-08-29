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

- adapters: CSV/Google Sheets adapters for data sources.
- workflow: Workflow data model, step definitions, and validators.
- kpi: KPI calculations (throughput, WIP, cycle time, on-time rate).
- bottleneck: Simple bottleneck detection from logs and step estimates.
- dashboard: Streamlit UI for exploration and KPIs.
- export: CSV exporter utilities.
- utils: Common helpers.

## Notes

- The current storage is CSV-based. You can later swap adapters for SQL/ERP.
- The structure is future-proof for AI inference, SQL backends, and ERP integration.
