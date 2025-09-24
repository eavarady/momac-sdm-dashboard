import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from src.schema.models import ProductionLogRow, ProcessStepRow
from src.schema import validate as schema_validate


def iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_pl_rows():
    s1 = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
    rows = [
        {
            "timestamp": iso(s1 + timedelta(hours=1)),
            "start_time": iso(s1),
            "end_time": iso(s1 + timedelta(hours=2)),
            "line_id": "LINE-A",
            "product_id": "PRD-1",
            "step_id": "S1",
            "run_id": "R1",
            "quantity": 1,
            "status": "complete",
            "actual_machine_id": "MX-101",
        },
        # overlapping on same machine
        {
            "timestamp": iso(s1 + timedelta(hours=1, minutes=30)),
            "start_time": iso(s1 + timedelta(hours=1, minutes=30)),
            "end_time": iso(s1 + timedelta(hours=3)),
            "line_id": "LINE-A",
            "product_id": "PRD-1",
            "step_id": "S2",
            "run_id": "R1",
            "quantity": 1,
            "status": "complete",
            "actual_machine_id": "MX-101",
        },
    ]
    return pd.DataFrame(rows)


def test_machine_overlap_detected_by_validator():
    pl = make_pl_rows()
    ps = pd.DataFrame(
        [
            {
                "product_id": "PRD-1",
                "step_id": "S1",
                "step_name": "A",
                "requires_machine": True,
                "assigned_machine": "MX-101",
                "assigned_operators": "OP-1",
                "estimated_time": 1,
            },
            {
                "product_id": "PRD-1",
                "step_id": "S2",
                "step_name": "B",
                "requires_machine": True,
                "assigned_machine": "MX-101",
                "assigned_operators": "OP-2",
                "estimated_time": 1,
            },
        ]
    )

    tables = {"production_log": pl, "process_steps": ps}
    errs = schema_validate.check_uniques_and_fks(tables)
    # Expect at least one OVERLAP machine error
    assert any(str(e).startswith("OVERLAP machine") for e in errs)


def test_requires_machine_false_but_assigned_warns():
    ps = pd.DataFrame(
        [
            {
                "product_id": "PRD-1",
                "step_id": "S1",
                "step_name": "A",
                "requires_machine": False,
                "assigned_machine": "MX-101",
                "assigned_operators": "OP-1",
                "estimated_time": 1,
            }
        ]
    )
    tables = {
        "process_steps": ps,
        "operators": pd.DataFrame(
            [{"operator_id": "OP-1", "name": "X", "role": "Assembler"}]
        ),
    }
    errs = schema_validate.check_uniques_and_fks(tables)
    assert any("WARN process_steps" in e for e in errs)
