import pytest
from schema.models import ProcessStepRow


def test_requires_machine_enforced():
    with pytest.raises(ValueError):
        ProcessStepRow(
            product_id="P1", step_id="S1", requires_machine=True, assigned_machine=None
        )
    ok = ProcessStepRow(
        product_id="P1", step_id="S1", requires_machine=True, assigned_machine="MX-1"
    )
    assert ok.assigned_machine == "MX-1"


def test_not_required_when_false():
    row = ProcessStepRow(
        product_id="P1", step_id="S2", requires_machine=False, assigned_machine=None
    )
    assert row.assigned_machine is None
