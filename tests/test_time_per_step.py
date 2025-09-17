import pandas as pd
from kpi.time_per_step import compute_time_per_step


def _make_log():
    # Two products, two steps; different durations
    data = [
        {
            "product_id": "P1",
            "step_id": "S1",
            "status": "complete",
            "start_time": "2025-09-01T00:00:00Z",
            "end_time": "2025-09-01T01:00:00Z",  # 1 hr
        },
        {
            "product_id": "P1",
            "step_id": "S1",
            "status": "complete",
            "start_time": "2025-09-02T00:00:00Z",
            "end_time": "2025-09-02T02:00:00Z",  # 2 hr
        },
        {
            "product_id": "P1",
            "step_id": "S2",
            "status": "complete",
            "start_time": "2025-09-01T00:30:00Z",
            "end_time": "2025-09-01T02:30:00Z",  # 2 hr
        },
        {
            "product_id": "P2",
            "step_id": "S1",
            "status": "complete",
            "start_time": "2025-09-03T00:00:00Z",
            "end_time": "2025-09-03T04:00:00Z",  # 4 hr
        },
        {
            "product_id": "P2",
            "step_id": "S1",
            "status": "in_progress",
            "start_time": "2025-09-03T05:00:00Z",
            "end_time": None,
        },
    ]
    return pd.DataFrame(data)


def test_basic_stats():
    df = _make_log()
    out = compute_time_per_step(df)
    # P1-S1: durations [1, 2] -> mean=1.5, median=1.5, std ~ 0.7071
    p1s1 = out[(out["product_id"] == "P1") & (out["step_id"] == "S1")].iloc[0]
    assert abs(p1s1["avg_duration_hours"] - 1.5) < 1e-6
    assert abs(p1s1["median_duration_hours"] - 1.5) < 1e-6
    assert 0.70 <= p1s1["std_duration_hours"] <= 0.71
    assert p1s1["events"] == 2

    # P2-S1: only 4 hr completed (the in_progress row should be excluded)
    p2s1 = out[(out["product_id"] == "P2") & (out["step_id"] == "S1")].iloc[0]
    assert abs(p2s1["avg_duration_hours"] - 4.0) < 1e-6
    assert abs(p2s1["median_duration_hours"] - 4.0) < 1e-6
    assert abs(p2s1["std_duration_hours"] - 0.0) < 1e-6
    assert p2s1["events"] == 1


def test_date_filtering():
    df = _make_log()
    # Only include events starting on 2025-09-02 or later
    out = compute_time_per_step(df, date_start="2025-09-02T00:00:00Z")
    # P1-S1 retains only the 2 hr entry -> mean=2, events=1
    p1s1 = out[(out["product_id"] == "P1") & (out["step_id"] == "S1")].iloc[0]
    assert abs(p1s1["avg_duration_hours"] - 2.0) < 1e-6
    assert p1s1["events"] == 1

    # P1-S2 should be excluded (start_time 2025-09-01)
    assert out[(out["product_id"] == "P1") & (out["step_id"] == "S2")].empty


def test_empty_and_missing_columns():
    empty = pd.DataFrame()
    out = compute_time_per_step(empty)
    assert out.empty

    missing_cols = pd.DataFrame({"start_time": ["2025-01-01"]})
    out2 = compute_time_per_step(missing_cols)
    assert out2.empty
