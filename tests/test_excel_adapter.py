import pandas as pd
import pytest
from pathlib import Path

pytest.importorskip("openpyxl")  # skip cleanly if engine not installed


def test_read_excel_tables_happy_path(tmp_path):
    from adapters.excel_adapter import read_excel_tables

    xlsx = tmp_path / "data.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        pd.DataFrame(
            {
                "machine_id": ["M1"],
                "line_id": ["L1"],
                "type": ["robot"],
                "status": ["online"],
            }
        ).to_excel(w, sheet_name="machines", index=False)

        pd.DataFrame({"line_id": ["L1"], "name": ["Line 1"], "shift": ["A"]}).to_excel(
            w, sheet_name="production_lines", index=False
        )

        pd.DataFrame(
            {
                "product_id": ["P1"],
                "name": ["Prod"],
                "category": [""],
                "spec_version": [""],
            }
        ).to_excel(w, sheet_name="products", index=False)

        pd.DataFrame({"operator_id": ["O1"], "name": ["Op"], "role": [""]}).to_excel(
            w, sheet_name="operators", index=False
        )

        pd.DataFrame(
            {
                "product_id": ["P1"],
                "step_id": ["S1"],
                "step_name": ["A"],
                # Provide a valid assigned_machine to satisfy requires_machine=True
                "assigned_machine": ["M1"],
                "assigned_operators": [""],
                "estimated_time": [1],
                "dependency_step_id": [None],
            }
        ).to_excel(w, sheet_name="process_steps", index=False)

        pd.DataFrame(
            {
                "timestamp": ["2025-09-01T10:00:00Z"],
                "start_time": ["2025-09-01T10:00:00Z"],
                "end_time": [None],
                "line_id": ["L1"],
                "product_id": ["P1"],
                "step_id": ["S1"],
                "quantity": [1],
                "status": ["in_progress"],
            }
        ).to_excel(w, sheet_name="production_log", index=False)

        pd.DataFrame(
            {
                "timestamp": ["2025-09-01T10:00:00Z"],
                "machine_id": ["M1"],
                "metric_type": ["temp"],
                "metric_value": ["42.0"],
            }
        ).to_excel(w, sheet_name="machine_metrics", index=False)

        pd.DataFrame(
            {
                "timestamp": ["2025-09-01T10:00:00Z"],
                "product_id": ["P1"],
                "check_type": ["dim"],
                "result": ["pass"],
                "inspector_id": ["O1"],
            }
        ).to_excel(w, sheet_name="quality_checks", index=False)

    tables = read_excel_tables(xlsx)
    assert set(tables.keys()) == {
        "machines",
        "production_lines",
        "products",
        "operators",
        "process_steps",
        "production_log",
        "machine_metrics",
        "quality_checks",
    }
