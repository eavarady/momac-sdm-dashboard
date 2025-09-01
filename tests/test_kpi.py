import pandas as pd
import pytest
from kpi.kpi_calculator import compute_all_kpis


@pytest.mark.xfail(reason="KPI logic is placeholder")
def test_kpis_basic():
    prod_log = pd.DataFrame(
        [
            {"quantity": 3, "status": "completed"},
            {"quantity": 2, "status": "in_progress"},
        ]
    )
    kpis = compute_all_kpis({"production_log": prod_log})
    assert kpis["throughput"] == 5.0
    assert kpis["wip"] == 2
