import pandas as pd
from kpi.progress import per_step_progress, overall_progress_by_product
import pytest

def test_per_step_progress_basic():
    steps = pd.DataFrame(
        [
            {"product_id": "P1", "step_id": "S1", "step_name": "A"},
            {"product_id": "P1", "step_id": "S2", "step_name": "B"},
            {"product_id": "P1", "step_id": "S3", "step_name": "C"},
        ]
    )
    log = pd.DataFrame(
        [
            # S1: 2 complete, 2 in_progress -> 0.5
            {"product_id": "P1", "step_id": "S1", "status": "completed", "quantity": 2},
            {"product_id": "P1", "step_id": "S1", "status": "in_progress", "quantity": 2},
            # S2: 5 in_progress -> 0.0
            {"product_id": "P1", "step_id": "S2", "status": "in_progress", "quantity": 5},
            # S3: no rows -> 0.0
        ]
    )

    sp = per_step_progress(steps, log).sort_values(["product_id", "step_id"]).reset_index(drop=True)
    # Extract progress as dict for easy assertions
    prog = {row.step_id: row.progress for _, row in sp.iterrows()}

    assert prog["S1"] == 0.5
    assert prog["S2"] == 0.0
    assert prog["S3"] == 0.0

def test_overall_progress_mean():
    steps = pd.DataFrame(
        [
            {"product_id": "P1", "step_id": "S1"},
            {"product_id": "P1", "step_id": "S2"},
            {"product_id": "P1", "step_id": "S3"},
        ]
    )
    log = pd.DataFrame(
        [
            {"product_id": "P1", "step_id": "S1", "status": "complete", "quantity": 3},
            {"product_id": "P1", "step_id": "S2", "status": "in_progress", "quantity": 3},
            # S3 no activity
        ]
    )
    sp = per_step_progress(steps, log)
    overall = overall_progress_by_product(sp)
    val = float(overall.loc[overall["product_id"] == "P1", "overall_progress"].iloc[0])
    # S1=1.0, S2=0.0, S3=0.0 -> mean = 1/3
    assert val == pytest.approx(1.0 / 3.0, rel=1e-9, abs=1e-12)

def test_per_step_progress_empty_log():
    steps = pd.DataFrame(
        [
            {"product_id": "P2", "step_id": "A"},
            {"product_id": "P2", "step_id": "B"},
        ]
    )
    log = pd.DataFrame([])
    sp = per_step_progress(steps, log).sort_values(["product_id", "step_id"])
    assert (sp["progress"] == 0.0).all()