import pandas as pd
from datetime import datetime, timezone, timedelta

from schema import validate


def _utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def test_operator_overlap_detected():
    operators = pd.DataFrame(
        [{"operator_id": "OP-1", "name": "A", "role": "Assembler"}]
    )

    la = pd.DataFrame(
        [
            {
                "activity_id": "A1",
                "operator_id": "OP-1",
                "product_id": "P1",
                "step_id": "S1",
                "start_time": _utc("2025-09-23T08:00:00Z"),
                "end_time": _utc("2025-09-23T09:00:00Z"),
                "activity_type": "direct",
            },
            {
                "activity_id": "A2",
                "operator_id": "OP-1",
                "product_id": "P1",
                "step_id": "S1",
                "start_time": _utc("2025-09-23T08:30:00Z"),
                "end_time": _utc("2025-09-23T09:30:00Z"),
                "activity_type": "direct",
            },
        ]
    )

    errs = validate.check_uniques_and_fks(
        {"operators": operators, "labor_activities": la}
    )
    assert any(
        "OVERLAP operator" in e for e in errs
    ), f"expected operator overlap, got: {errs}"


def test_machine_overlap_detected_via_assigned():
    # process_steps assigns MX-1 for P1/S1
    ps = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "step_id": "S1",
                "step_name": "Do",
                "requires_machine": True,
                "assigned_machine": "MX-1",
                "assigned_operators": [],
                "estimated_time": 1,
            }
        ]
    )

    pl = pd.DataFrame(
        [
            {
                "timestamp": _utc("2025-09-23T08:00:00Z"),
                "start_time": _utc("2025-09-23T08:00:00Z"),
                "end_time": _utc("2025-09-23T09:00:00Z"),
                "line_id": "L1",
                "product_id": "P1",
                "step_id": "S1",
                "run_id": "R1",
                "quantity": 1,
                "status": "complete",
                "actual_machine_id": None,
            },
            {
                "timestamp": _utc("2025-09-23T08:30:00Z"),
                "start_time": _utc("2025-09-23T08:30:00Z"),
                "end_time": _utc("2025-09-23T09:30:00Z"),
                "line_id": "L1",
                "product_id": "P1",
                "step_id": "S1",
                "run_id": "R2",
                "quantity": 1,
                "status": "complete",
                "actual_machine_id": None,
            },
        ]
    )

    errs = validate.check_uniques_and_fks({"process_steps": ps, "production_log": pl})
    assert any(
        "OVERLAP machine" in e for e in errs
    ), f"expected machine overlap, got: {errs}"


def test_process_step_planning_sanity():
    # requires_machine=False but assigned_machine set -> WARN
    ps_warn = pd.DataFrame(
        [
            {
                "product_id": "P2",
                "step_id": "S1",
                "requires_machine": False,
                "assigned_machine": "MX-1",
                "assigned_operators": [],
                "estimated_time": 1,
            }
        ]
    )

    errs = validate.check_uniques_and_fks({"process_steps": ps_warn})
    assert any(
        "WARN process_steps" in e for e in errs
    ), f"expected planning WARN, got: {errs}"

    # requires_machine=True but assigned_machine empty -> ERR
    ps_err = pd.DataFrame(
        [
            {
                "product_id": "P3",
                "step_id": "S1",
                "requires_machine": True,
                "assigned_machine": None,
                "assigned_operators": [],
                "estimated_time": 1,
            }
        ]
    )

    errs2 = validate.check_uniques_and_fks({"process_steps": ps_err})
    assert any(
        "ERR process_steps" in e for e in errs2
    ), f"expected planning ERR, got: {errs2}"
