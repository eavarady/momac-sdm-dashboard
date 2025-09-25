import pandas as pd


def compute_manpower_utilization(
    labor_log: pd.DataFrame, period_start: pd.Timestamp, period_end: pd.Timestamp
) -> dict:
    if labor_log is None or labor_log.empty:
        return {"overall": 0.0, "by_operator": {}, "by_role": {}, "by_line": {}}

    df = labor_log.copy()

    def _ensure_utc(series: pd.Series) -> pd.Series:
        s = pd.to_datetime(series)
        if s.dt.tz is None:
            s = s.dt.tz_localize("UTC")
        else:
            s = s.dt.tz_convert("UTC")
        return s

    df["start_time"] = _ensure_utc(df["start_time"])
    df["end_time"] = _ensure_utc(df["end_time"]) if "end_time" in df.columns else pd.NaT

    period_start = pd.to_datetime(period_start)
    if getattr(period_start, "tzinfo", None) is None:
        period_start = period_start.tz_localize("UTC")
    period_end = pd.to_datetime(period_end)
    if getattr(period_end, "tzinfo", None) is None:
        period_end = period_end.tz_localize("UTC")

    def _clip_interval(s, e):
        if pd.isna(s):
            return None
        if pd.isna(e):
            e = period_end
        if e <= period_start or s >= period_end:
            return None
        start = max(s, period_start)
        end = min(e, period_end)
        if end <= start:
            return None
        return (start, end)

    ops = {}
    for _, row in df.iterrows():
        op = row.get("operator_id")
        s = row.get("start_time")
        e = row.get("end_time")
        clipped = _clip_interval(s, e)
        if clipped is None:
            continue
        ops.setdefault(op, []).append(
            (clipped[0], clipped[1], row.get("activity_type"), row.get("line_id"))
        )

    def _union_seconds(intervals):
        if not intervals:
            return 0.0
        ivs = sorted([(s, e) for s, e in intervals], key=lambda x: x[0])
        total = 0.0
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                total += (cur_e - cur_s).total_seconds()
                cur_s, cur_e = s, e
        total += (cur_e - cur_s).total_seconds()
        return total

    by_operator = {}
    total_work = 0.0
    total_available = 0.0
    for op, rows in ops.items():
        all_intervals = [(s, e) for s, e, _, _ in rows]
        work_intervals = [
            (s, e)
            for s, e, atype, _ in rows
            if str(atype).lower() in ("direct", "setup", "rework")
        ]
        avail_sec = _union_seconds(all_intervals)
        work_sec = _union_seconds(work_intervals)
        util = 0.0 if avail_sec == 0 else min(work_sec / avail_sec, 1.0)
        by_operator[op] = {
            "utilization": util,
            "work_seconds": work_sec,
            "available_seconds": avail_sec,
        }
        total_work += work_sec
        total_available += avail_sec

    overall = 0.0
    if total_available > 0:
        overall = min(total_work / total_available, 1.0)

    by_line = {}
    for op, v in by_operator.items():
        line = None
        rows = ops.get(op, [])
        for s, e, atype, line_id in rows:
            if line_id:
                line = line_id
                break
        key = line or "__unknown__"
        entry = by_line.setdefault(key, {"work_seconds": 0.0, "available_seconds": 0.0})
        entry["work_seconds"] += v["work_seconds"]
        entry["available_seconds"] += v["available_seconds"]

    for k in list(by_line.keys()):
        e = by_line[k]
        e["utilization"] = (
            0.0
            if e["available_seconds"] == 0
            else min(e["work_seconds"] / e["available_seconds"], 1.0)
        )

    by_role = {}

    return {
        "overall": overall,
        "by_operator": by_operator,
        "by_role": by_role,
        "by_line": by_line,
    }
