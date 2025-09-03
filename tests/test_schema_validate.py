import pandas as pd
import pytest

from schema.validate import validate_dataframe


def test_production_log_happy_path():
    df = pd.DataFrame(
        {
            "timestamp": ["2025-09-01T10:00:00Z", "2025-09-01T10:05:00Z"],
            "start_time": ["2025-09-01T10:00:00Z", "2025-09-01T10:04:00Z"],
            "end_time": ["2025-09-01T10:01:00Z", None],  # second row in_progress
            "line_id": ["L1", "L1"],
            "product_id": ["P1", "P1"],
            "step_id": ["S1", "S2"],
            "quantity": [3, 2],
            "status": ["complete", "in_progress"],
        }
    )
    out = validate_dataframe(df, "production_log")
    # dtypes & normalization
    assert str(out["timestamp"].dtype).startswith("datetime64[ns, UTC]")
    assert str(out["start_time"].dtype).startswith("datetime64[ns, UTC]")
    assert str(out["end_time"].dtype).startswith("datetime64[ns, UTC]")
    assert out["quantity"].dtype == "int64"
    assert set(out["status"].unique()) == {"complete", "in_progress"}


def test_missing_required_column_raises():
    df = pd.DataFrame(
        {
            # "timestamp" intentionally missing
            "start_time": ["2025-09-01T10:00:00Z"],
            "end_time": [None],
            "line_id": ["L1"],
            "product_id": ["P1"],
            "step_id": ["S1"],
            "quantity": [1],
            "status": ["complete"],
        }
    )
    with pytest.raises(ValueError) as e:
        validate_dataframe(df, "production_log")
    msg = str(e.value)
    assert "missing required columns" in msg or "Validation failed for production_log" in msg


def test_machines_status_synonyms():
    df = pd.DataFrame(
        {
            "machine_id": ["M1", "M2"],
            "line_id": ["L1", "L1"],
            "type": ["robot", "drill"],
            "status": ["active", "maintenance"],  # "active" should normalize to "online"
        }
    )
    out = validate_dataframe(df, "machines")
    assert set(out["status"].unique()) == {"online", "maintenance"}


def test_process_steps_dependency_nan_to_none():
    df = pd.DataFrame(
        {
            "product_id": ["P1", "P1"],
            "step_id": ["S1", "S2"],
            "step_name": ["A", "B"],
            "assigned_machine": ["M1", "M1"],
            "assigned_operators": ["op1, op2", ""],
            "estimated_time": [1, 2],
            "dependency_step_id": ["nan", "S1"],  # first should become None
        }
    )
    out = validate_dataframe(df, "process_steps")
    # First row dep should be None after normalization
    assert pd.isna(out.loc[0, "dependency_step_id"])
    assert out.loc[1, "dependency_step_id"] == "S1"