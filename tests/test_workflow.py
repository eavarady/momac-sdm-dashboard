import pandas as pd
import pytest
from workflow.validator import steps_from_dataframe, validate_dependencies


@pytest.mark.xfail(reason="Workflow validator is placeholder")
def test_steps_and_validation():
    df = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "step_id": "S1",
                "step_name": "A",
                "estimated_time": 1,
            },
            {
                "product_id": "P1",
                "step_id": "S2",
                "step_name": "B",
                "dependency_step_id": "S1",
                "estimated_time": 2,
            },
        ]
    )
    steps = steps_from_dataframe(df)
    errs = validate_dependencies(steps)
    assert errs == []
