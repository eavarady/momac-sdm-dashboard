import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from src.schema.models import LaborActivityRow
from src.kpi.kpi_calculator import (
    compute_manpower_utilization,
    compute_labor_efficiency,
)


def tznow():
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_labor_activity_time_logic_valid_and_invalid():
    # valid: end >= start
    start = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
    row = {
        "activity_id": "ACT-1",
        "operator_id": "OP-1",
        "product_id": "PRD-1",
        "step_id": "S1",
        "start_time": start,
        "end_time": end,
        "activity_type": "direct",
    }
    obj = LaborActivityRow(**row)
    assert obj.start_time <= obj.end_time

    # invalid: end < start
    bad_end = datetime(2024, 1, 1, 7, 0, tzinfo=timezone.utc)
    row2 = dict(row)
    row2["activity_id"] = "ACT-2"
    row2["start_time"] = start
    row2["end_time"] = bad_end
    with pytest.raises(Exception):
        LaborActivityRow(**row2)


def test_compute_manpower_utilization_simple():
    # One operator with two non-overlapping intervals: available = union = 2h, work (direct) = 1.5h
    start = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
    rows = [
        {
            "activity_id": "A1",
            "operator_id": "OP-1",
            "product_id": "PRD-1",
            "step_id": "S1",
            "start_time": (start).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": (start + timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "activity_type": "direct",
            "line_id": "LINE-A",
        },
        {
            "activity_id": "A2",
            "operator_id": "OP-1",
            "product_id": "PRD-1",
            "step_id": "S2",
            "start_time": (start + timedelta(hours=1, minutes=30)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "end_time": (start + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "activity_type": "indirect",
            "line_id": "LINE-A",
        },
    ]
    df = pd.DataFrame(rows)
    period_start = pd.to_datetime(df.start_time.min())
    period_end = pd.to_datetime(df.end_time.max())
    res = compute_manpower_utilization(df, period_start, period_end)
    assert "by_operator" in res
    op = res["by_operator"].get("OP-1")
    assert op is not None
    # available seconds = 2 hours = 7200, work seconds = 90 minutes = 5400
    assert pytest.approx(op["available_seconds"], rel=1e-3) == 7200
    assert pytest.approx(op["work_seconds"], rel=1e-3) == 5400
    assert pytest.approx(op["utilization"], rel=1e-3) == 5400 / 7200


def test_compute_labor_efficiency_simple():
    # One activity duration 2h, standard_time_per_unit = 1h -> efficiency = earned/actual = 1/2
    lab_rows = [
        {
            "activity_id": "A1",
            "operator_id": "OP-1",
            "product_id": "PRD-1",
            "step_id": "S1",
            "line_id": "LINE-A",
            "start_time": "2024-01-01T08:00:00Z",
            "end_time": "2024-01-01T10:00:00Z",
            "activity_type": "direct",
        }
    ]
    df_lab = pd.DataFrame(lab_rows)
    proc = pd.DataFrame([{"product_id": "PRD-1", "step_id": "S1", "estimated_time": 1}])
    res = compute_labor_efficiency(df_lab, proc)
    assert "overall" in res
    assert pytest.approx(res["overall"], rel=1e-3) == 0.5
