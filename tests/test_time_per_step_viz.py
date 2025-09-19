import pandas as pd
import pytest

from visualizations.time_per_step_viz import (
    prepare_step_duration_events,
    build_step_duration_histogram,
)


def test_prepare_events_empty():
    df = pd.DataFrame()
    out = prepare_step_duration_events(df)
    assert out.empty
    assert set(out.columns) == {
        "product_id",
        "step_id",
        "start_time",
        "end_time",
        "duration_hours",
    }


def test_prepare_events_invalid_columns():
    df = pd.DataFrame({"start_time": ["2024-01-01"], "end_time": ["2024-01-01"]})
    out = prepare_step_duration_events(df)
    assert out.empty


def test_prepare_events_filters_and_duration():
    df = pd.DataFrame(
        {
            "product_id": [1, 1, 2],
            "step_id": ["A", "A", "B"],
            "status": ["complete", "in_progress", "complete"],
            "start_time": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T01:00:00Z",
                "2024-01-02T00:00:00Z",
            ],
            "end_time": [
                "2024-01-01T02:00:00Z",
                "2024-01-01T02:30:00Z",
                "2024-01-02T03:00:00Z",
            ],
        }
    )
    out = prepare_step_duration_events(
        df, date_start="2024-01-01", date_end="2024-01-01T23:59:59Z"
    )
    # Only first row qualifies (complete + within date range)
    assert len(out) == 1
    assert pytest.approx(out.iloc[0]["duration_hours"], rel=1e-6) == 2.0


def test_prepare_events_end_before_start_filtered():
    df = pd.DataFrame(
        {
            "product_id": [1],
            "step_id": ["A"],
            "status": ["complete"],
            "start_time": ["2024-01-01T02:00:00Z"],
            "end_time": ["2024-01-01T01:00:00Z"],  # earlier than start -> invalid
        }
    )
    out = prepare_step_duration_events(df)
    assert out.empty


def test_histogram_no_data_returns_none():
    events = pd.DataFrame(
        columns=["product_id", "step_id", "start_time", "end_time", "duration_hours"]
    )
    fig = build_step_duration_histogram(events)
    assert fig is None


def test_histogram_with_filters():
    events = pd.DataFrame(
        {
            "product_id": [1, 1, 2, 2],
            "step_id": ["S1", "S2", "S1", "S2"],
            "start_time": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T02:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-02T03:00:00Z",
                ],
                utc=True,
            ),
            "end_time": pd.to_datetime(
                [
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T03:30:00Z",
                    "2024-01-02T01:30:00Z",
                    "2024-01-02T04:00:00Z",
                ],
                utc=True,
            ),
            "duration_hours": [1.0, 1.5, 1.5, 1.0],
            "product_label": ["P1", "P1", "P2", "P2"],
            "step_label": ["Prep", "Assemble", "Prep", "Assemble"],
        }
    )
    fig_all = build_step_duration_histogram(events, nbins=10)
    assert fig_all is not None
    fig_filtered = build_step_duration_histogram(
        events,
        product_label_filter="P1",
        step_labels_filter=["Prep"],
        nbins=5,
    )
    assert fig_filtered is not None
    # Ensure filtered data size corresponds (only one matching row duration 1.0)
    # We can inspect underlying figure data
    filtered_x = []
    for trace in fig_filtered.data:
        filtered_x.extend(list(trace.x))
    assert all(v == 1.0 for v in filtered_x)
