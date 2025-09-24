"""Quick validation runner: prints operator overlaps, machine overlaps, and simple process_steps sanity checks."""

from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _parse(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return None


def find_overlaps(df, id_col, start_col="start_time", end_col="end_time"):
    out = []
    if df is None or df.empty:
        return out
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col])
    df[end_col] = pd.to_datetime(df[end_col])
    for key, grp in df.groupby(id_col):
        ivs = sorted(
            grp[[start_col, end_col]].itertuples(index=False, name=None),
            key=lambda x: x[0],
        )
        prev_s, prev_e = None, None
        for s, e in ivs:
            if prev_s is None:
                prev_s, prev_e = s, e
                continue
            if s < prev_e:
                out.append((key, prev_s, prev_e, s, e))
            if e > prev_e:
                prev_s, prev_e = s, e
    return out


def main():
    data_files = {
        "production_log": DATA / "production_log.csv",
        "process_steps": DATA / "process_steps.csv",
        "labor_activities": DATA / "labor_activities.csv",
    }
    tables = {}
    for name, path in data_files.items():
        if path.exists():
            tables[name] = pd.read_csv(path)
        else:
            tables[name] = pd.DataFrame()

    print("-- Operator overlaps --")
    la = tables.get("labor_activities")
    if not la.empty:
        la["start_time"] = (
            pd.to_datetime(la["start_time"]) if "start_time" in la.columns else pd.NaT
        )
        la["end_time"] = (
            pd.to_datetime(la["end_time"]) if "end_time" in la.columns else pd.NaT
        )
        ops = find_overlaps(la, "operator_id")
        for o in ops[:50]:
            print(f"OVERLAP operator {o[0]}: {o[1]}-{o[2]} overlaps {o[3]}-{o[4]}")
    else:
        print("no labor_activities present")

    print("\n-- Machine overlaps --")
    pl = tables.get("production_log")
    if not pl.empty:
        pl["start_time"] = (
            pd.to_datetime(pl["start_time"]) if "start_time" in pl.columns else pd.NaT
        )
        pl["end_time"] = (
            pd.to_datetime(pl["end_time"]) if "end_time" in pl.columns else pd.NaT
        )
        pl["machine"] = pl.get("actual_machine_id").fillna("")
        mows = find_overlaps(pl, "machine")
        for m in mows[:50]:
            print(f"OVERLAP machine {m[0]}: {m[1]}-{m[2]} overlaps {m[3]}-{m[4]}")
    else:
        print("no production_log present")

    print("\n-- Process step sanity --")
    ps = tables.get("process_steps")
    if not ps.empty:
        # zero estimated time
        zeros = ps[ps.get("estimated_time", 0) == 0]
        for _, r in zeros.iterrows():
            print(
                f"ZERO_ESTIMATED: product={r.get('product_id')} step={r.get('step_id')}"
            )
        # requires_machine false but has assigned_machine
        for _, r in ps.iterrows():
            req = r.get("requires_machine")
            am = r.get("assigned_machine")
            if (not bool(req)) and pd.notna(am) and str(am).strip():
                print(
                    f"WARN_REQUIRES_FALSE_ASSIGNED: product={r.get('product_id')} step={r.get('step_id')} assigned={am}"
                )
    else:
        print("no process_steps present")


if __name__ == "__main__":
    main()
