import pandas as pd
import pytest

from workflow.validator import steps_from_dataframe, validate_dependencies


def test_steps_from_dataframe_parses_operators_and_estimated_time():
    df = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "step_id": "S1",
                "step_name": "Assemble",
                "assigned_machine": "M-1",
                "assigned_operators": "OP-1, OP-1, , OP-2",
                "estimated_time": "10.0",  # stringy float -> int(10)
                "dependency_step_id": "",  # empty -> None
            }
        ]
    )
    steps = steps_from_dataframe(df)
    assert len(steps) == 1
    s = steps[0]
    assert s.product_id == "P1"
    assert s.step_id == "S1"
    assert s.step_name == "Assemble"
    assert s.assigned_machine == "M-1"
    # dedup + drop empties + preserve order
    assert s.assigned_operators == ["OP-1", "OP-2"]
    # coerced to non-negative int
    assert s.estimated_time == 10
    # empty dependency -> None
    assert s.dependency_step_id is None


def test_steps_from_dataframe_handles_bad_estimated_time_and_duplicates():
    df = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "step_id": "S1",
                "estimated_time": "abc",  # invalid -> 0
            },
            {
                "product_id": "P1",
                "step_id": "S1",  # duplicate key -> dropped (first wins)
                "estimated_time": 99,
            },
            {
                "product_id": "P1",
                "step_id": "S2",
                "estimated_time": -5,  # negative -> 0
                "dependency_step_id": " ",  # whitespace -> None
            },
        ]
    )
    steps = steps_from_dataframe(df)
    # duplicate (P1,S1) should be ignored; we keep first
    assert len(steps) == 2
    s1 = next(s for s in steps if s.step_id == "S1")
    s2 = next(s for s in steps if s.step_id == "S2")
    assert s1.estimated_time == 0
    assert s2.estimated_time == 0
    assert s2.dependency_step_id is None


def test_validate_dependencies_happy_path_linear_chain():
    df = pd.DataFrame(
        [
            {"product_id": "P1", "step_id": "S1", "estimated_time": 1},
            {"product_id": "P1", "step_id": "S2", "dependency_step_id": "S1", "estimated_time": 2},
            {"product_id": "P1", "step_id": "S3", "dependency_step_id": "S2", "estimated_time": 3},
        ]
    )
    steps = steps_from_dataframe(df)
    errs = validate_dependencies(steps)
    assert errs == []


def test_validate_dependencies_reports_missing_dependency():
    df = pd.DataFrame(
        [
            {"product_id": "P1", "step_id": "S2", "dependency_step_id": "S9", "estimated_time": 2},
            {"product_id": "P1", "step_id": "S1", "estimated_time": 1},
        ]
    )
    steps = steps_from_dataframe(df)
    errs = validate_dependencies(steps)
    # one missing dep for S2->S9
    assert any("MISSING_DEP" in e and "product=P1" in e and "step=S2" in e and "depends_on=S9" in e for e in errs)


def test_validate_dependencies_detects_cycle_simple():
    df = pd.DataFrame(
        [
            {"product_id": "P1", "step_id": "S1", "dependency_step_id": "S2", "estimated_time": 1},
            {"product_id": "P1", "step_id": "S2", "dependency_step_id": "S1", "estimated_time": 1},
        ]
    )
    steps = steps_from_dataframe(df)
    errs = validate_dependencies(steps)
    # cycle should be reported
    assert any("CYCLE" in e and "product=P1" in e for e in errs)


def test_validate_dependencies_isolated_by_product():
    # P1 is valid; P2 references a missing S1 within P2
    df = pd.DataFrame(
        [
            {"product_id": "P1", "step_id": "S1"},
            {"product_id": "P1", "step_id": "S2", "dependency_step_id": "S1"},
            {"product_id": "P2", "step_id": "S2", "dependency_step_id": "S1"},  # missing within P2
        ]
    )
    steps = steps_from_dataframe(df)
    errs = validate_dependencies(steps)
    # Only P2 should have a missing dep error
    assert any("MISSING_DEP" in e and "product=P2" in e and "step=S2" in e and "depends_on=S1" in e for e in errs)
    assert not any("product=P1" in e for e in errs)