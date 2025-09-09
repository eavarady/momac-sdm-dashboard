import pandas as pd
import pytest

from ml.bottleneck_detector import detect_bottleneck


def _df_production_log(rows):
    return pd.DataFrame(
        rows,
        columns=["timestamp", "line_id", "product_id", "step_id", "quantity", "status"],
    )


def _df_process_steps(rows):
    # not used by v1 heuristic, but keep shape realistic
    return pd.DataFrame(
        rows,
        columns=[
            "product_id",
            "step_id",
            "step_name",
            "assigned_machine",
            "assigned_operators",
            "estimated_time",
            "dependency_step_id",
        ],
    )


def test_bottleneck_happy_path():
    # two steps in progress: step_B has higher total WIP
    production_log = _df_production_log(
        [
            ("2025-09-01T10:00Z", "L1", "P1", "step_A", 3, "in_progress"),
            ("2025-09-01T10:01Z", "L1", "P1", "step_A", 2, "in_progress"),
            ("2025-09-01T10:02Z", "L1", "P1", "step_B", 8, "in_progress"),
            ("2025-09-01T10:03Z", "L1", "P1", "step_C", 5, "complete"),  # not counted
        ]
    )
    process_steps = _df_process_steps([])

    got = detect_bottleneck(process_steps, production_log)
    assert got == "step_B"


def test_bottleneck_tie_breaks_lexicographic_step_id():
    # step_A and step_B both have WIP=10 -> tie-break should pick lexicographically smaller "step_A"
    production_log = _df_production_log(
        [
            ("2025-09-01T10:00Z", "L1", "P1", "step_A", 5, "in_progress"),
            ("2025-09-01T10:01Z", "L1", "P1", "step_A", 5, "in_progress"),
            ("2025-09-01T10:02Z", "L1", "P1", "step_B", 4, "in_progress"),
            ("2025-09-01T10:03Z", "L1", "P1", "step_B", 6, "in_progress"),
        ]
    )
    process_steps = _df_process_steps([])

    got = detect_bottleneck(process_steps, production_log)
    assert got == "step_A"


def test_bottleneck_no_in_progress_returns_none():
    # nothing in progress -> no bottleneck
    production_log = _df_production_log(
        [
            ("2025-09-01T10:00Z", "L1", "P1", "step_A", 3, "complete"),
            ("2025-09-01T10:01Z", "L1", "P1", "step_B", 2, "complete"),
        ]
    )
    process_steps = _df_process_steps([])

    got = detect_bottleneck(process_steps, production_log)
    assert got is None


def test_bottleneck_ignores_bad_quantities():
    # negative, zero, and NaN quantities should not count toward WIP
    production_log = _df_production_log(
        [
            ("2025-09-01T10:00Z", "L1", "P1", "step_A", -5, "in_progress"),  # ignored
            ("2025-09-01T10:01Z", "L1", "P1", "step_A", 0, "in_progress"),  # ignored
            ("2025-09-01T10:02Z", "L1", "P1", "step_A", 7, "in_progress"),  # counts
            ("2025-09-01T10:03Z", "L1", "P1", "step_B", None, "in_progress"),  # ignored
            ("2025-09-01T10:04Z", "L1", "P1", "step_B", 6, "in_progress"),  # counts
        ]
    )
    process_steps = _df_process_steps([])

    got = detect_bottleneck(process_steps, production_log)
    # step_A WIP=7; step_B WIP=6
    assert got == "step_A"


def test_bottleneck_status_is_case_insensitive():
    # status casing variants should be normalized (only in_progress counts)
    production_log = _df_production_log(
        [
            ("2025-09-01T10:00Z", "L1", "P1", "step_A", 4, "IN_PROGRESS"),
            ("2025-09-01T10:01Z", "L1", "P1", "step_B", 5, "In_Progress"),
            ("2025-09-01T10:02Z", "L1", "P1", "step_C", 9, "Complete"),
        ]
    )
    process_steps = _df_process_steps([])

    got = detect_bottleneck(process_steps, production_log)
    # step_B has higher WIP among the in_progress rows (5 vs 4)
    assert got == "step_B"


def test_bottleneck_missing_required_columns_returns_none():
    # drop one of the required columns to ensure the function safely returns None
    production_log = pd.DataFrame(
        [
            # missing 'status' column entirely
            ("2025-09-01T10:00Z", "L1", "P1", "step_A", 3),
        ],
        columns=["timestamp", "line_id", "product_id", "step_id", "quantity"],
    )
    process_steps = _df_process_steps([])

    got = detect_bottleneck(process_steps, production_log)
    assert got is None
