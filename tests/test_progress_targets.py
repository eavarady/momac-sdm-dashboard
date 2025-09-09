import pandas as pd
from kpi.progress import per_step_progress, overall_progress_by_product

def test_per_step_progress_with_targets_overrides_formula():
    steps = pd.DataFrame(
        [{"product_id":"P1","step_id":"S1"}, {"product_id":"P1","step_id":"S2"}]
    )
    log = pd.DataFrame(
        [
            {"product_id":"P1","step_id":"S1","status":"complete","quantity":3},
            {"product_id":"P1","step_id":"S1","status":"in_progress","quantity":7},  # would be 3/(3+7)=0.3
            {"product_id":"P1","step_id":"S2","status":"complete","quantity":5},
        ]
    )
    targets = pd.DataFrame(
        [
            {"product_id":"P1","step_id":"S1","target_qty":4},  # should be 3/4 = 0.75 (override)
            {"product_id":"P1","step_id":"S2","target_qty":10}, # 5/10 = 0.5
        ]
    )
    sp = per_step_progress(steps, log, targets=targets)
    s1 = float(sp.loc[(sp.product_id=="P1")&(sp.step_id=="S1"), "progress"].iloc[0])
    s2 = float(sp.loc[(sp.product_id=="P1")&(sp.step_id=="S2"), "progress"].iloc[0])
    assert s1 == 0.75
    assert s2 == 0.5

def test_overall_progress_uses_per_step_progress_with_targets():
    steps = pd.DataFrame(
        [{"product_id":"P1","step_id":"A"}, {"product_id":"P1","step_id":"B"}]
    )
    log = pd.DataFrame(
        [
            {"product_id":"P1","step_id":"A","status":"complete","quantity":2},
            {"product_id":"P1","step_id":"B","status":"complete","quantity":1},
        ]
    )
    targets = pd.DataFrame(
        [
            {"product_id":"P1","step_id":"A","target_qty":4},  # 2/4 = 0.5
            {"product_id":"P1","step_id":"B","target_qty":2},  # 1/2 = 0.5
        ]
    )
    sp = per_step_progress(steps, log, targets=targets)
    overall = overall_progress_by_product(sp)
    val = float(overall.loc[overall["product_id"]=="P1","overall_progress"].iloc[0])
    assert val == 0.5  # mean(0.5, 0.5)