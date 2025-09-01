import pandas as pd
import pytest
from sdm_bottlenecks import detect_bottleneck


@pytest.mark.xfail(reason="Bottleneck logic is placeholder")
def test_bottleneck_none_on_empty():
    assert detect_bottleneck(pd.DataFrame(), pd.DataFrame()) is None


@pytest.mark.xfail(reason="Bottleneck logic is placeholder")
def test_bottleneck_simple():
    steps = pd.DataFrame(
        [
            {"step_id": "S1", "estimated_time": 5},
            {"step_id": "S2", "estimated_time": 20},
        ]
    )
    prod_log = pd.DataFrame(
        [
            {"step_id": "S1", "status": "in_progress", "quantity": 1},
            {"step_id": "S2", "status": "in_progress", "quantity": 1},
        ]
    )
    bn = detect_bottleneck(steps, prod_log)
    assert bn == "S2"
