"""
Consolidated mock data generator (new schema only).

Outputs CSVs under data/ with the enforced schema:
- process_steps: estimated_time (hours)
- production_log: columns [timestamp, start_time, end_time, line_id, product_id, step_id, run_id, quantity, status]
  where status in {'in_progress','complete'} and timestamp is ISO 8601 Z
- machine_metrics, quality_checks: timestamps ISO 8601 Z
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import random
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def rand_ts_iso(start: datetime, end: datetime) -> str:
    delta = end - start
    sec = random.randint(0, int(delta.total_seconds()))
    t = start + timedelta(seconds=sec)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_mock_data(
    num_machines: int = 5,
    num_lines: int = 2,
    num_products: int = 3,
    num_operators: int = 4,
    steps_per_product: int = 4,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    years: float | None = 1.0,
    daily: bool = True,
    min_steps_per_day: int = 3,
    max_steps_per_day: int = 8,
    runs_per_week: float = 2.0,  # expected run starts per product per week (daily mode)
    seed: int | None = 42,
    unit_mode: bool = True,  # default to unit/SFC (qty == 1)
    include_requires_machine: bool = True,
    include_actual_machine: bool = False,
    qc_per_run: bool = False,
    qc_per_product: bool = False,
):
    if seed is not None:
        random.seed(seed)

    # Establish date window: prefer explicit start/end; else use years back from today (naive UTC)
    def _to_naive_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt
        return dt.astimezone(timezone.utc).replace(tzinfo=None)

    if end_date is None:
        # normalize to midnight UTC for stable day boundaries
        today_utc = (
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .replace(tzinfo=None)
        )
        end_date = today_utc
    if start_date is None:
        days = int(round((years or 1.0) * 365))
        start_date = end_date - timedelta(days=max(0, days - 1))
    # normalize any provided tz-aware inputs to naive UTC
    start_date = _to_naive_utc(start_date)
    end_date = _to_naive_utc(end_date)

    DATA.mkdir(parents=True, exist_ok=True)

    # Machines
    machines = pd.DataFrame(
        {
            "machine_id": [f"MX-{i+101}" for i in range(num_machines)],
            "line_id": [
                f"LINE-{chr(65 + (i % num_lines))}" for i in range(num_machines)
            ],
            "type": [
                random.choice(["CNC", "RobotArm", "Press"]) for _ in range(num_machines)
            ],
            "status": [
                random.choice(["online", "maintenance", "offline"])
                for _ in range(num_machines)
            ],
        }
    )
    machines.to_csv(DATA / "machines.csv", index=False)

    # Operators
    operators = pd.DataFrame(
        {
            "operator_id": [f"OP-{i+1}" for i in range(num_operators)],
            "name": [f"Operator {i+1}" for i in range(num_operators)],
            "role": [
                random.choice(["Assembler", "Tester", "Supervisor"])
                for _ in range(num_operators)
            ],
        }
    )
    operators.to_csv(DATA / "operators.csv", index=False)

    # Production Lines
    lines = pd.DataFrame(
        {
            "line_id": [f"LINE-{chr(65 + i)}" for i in range(num_lines)],
            "name": [f"Assembly Line {chr(65 + i)}" for i in range(num_lines)],
            "shift": [random.choice(["day", "night"]) for _ in range(num_lines)],
        }
    )
    lines.to_csv(DATA / "production_lines.csv", index=False)

    # Products
    products = pd.DataFrame(
        {
            "product_id": [f"PRD-{i+1}" for i in range(num_products)],
            "name": [f"Generator {i+1}" for i in range(num_products)],
            "category": ["Generator"] * num_products,
            "spec_version": [f"v{random.randint(1, 3)}.0" for _ in range(num_products)],
        }
    )
    products.to_csv(DATA / "products.csv", index=False)

    # Process Steps (estimated_time in hours, dependency optional)
    step_names = ["Unpack", "Assemble", "Test", "Repack"]
    process_steps_rows = []
    for prod_id in products["product_id"]:
        for idx in range(steps_per_product):
            step_id = f"{prod_id}-STEP-{idx+1}"
            dep = f"{prod_id}-STEP-{idx}" if idx > 0 else ""
            step_name = step_names[idx % len(step_names)]
            # Heuristic: unpack/repack are manual; assemble/test use machines
            if include_requires_machine:
                requires_machine = False if step_name in ("Unpack", "Repack") else True
            else:
                requires_machine = None

            assigned_machine = (
                random.choice(machines["machine_id"])
                if (requires_machine or requires_machine is None)
                else ""
            )

            row = {
                "product_id": prod_id,
                "step_id": step_id,
                "step_name": step_name,
                "assigned_machine": assigned_machine,
                "assigned_operators": ",".join(
                    random.sample(
                        list(operators["operator_id"]), k=random.randint(1, 2)
                    )
                ),
                "estimated_time": random.randint(1, 8),  # hours
                "dependency_step_id": dep,
            }
            if include_requires_machine:
                row["requires_machine"] = requires_machine
            process_steps_rows.append(row)
    process_steps = pd.DataFrame(process_steps_rows)
    process_steps.to_csv(DATA / "process_steps.csv", index=False)

    # Production Log: simulate realistic multi-day runs with consistent run_ids
    def within_workday(day: datetime) -> tuple[datetime, datetime]:
        # Return a time window (08:00 to 18:00 UTC) inside the given day
        start = day.replace(hour=8, minute=0, second=0, microsecond=0)
        end = day.replace(hour=18, minute=0, second=0, microsecond=0)
        return start, end

    date_count = (end_date.date() - start_date.date()).days + 1
    all_days = [start_date + timedelta(days=i) for i in range(date_count)]

    production_log_rows: list[dict] = []
    planned_runs: list[dict] = []  # for runs.csv

    def _step_order_for_product(pid: str) -> list[dict]:
        rows = process_steps.loc[process_steps["product_id"] == pid].copy()

        # Sort by numeric suffix if present, else by step_id
        def key_fn(sid: str) -> tuple:
            import re

            m = re.search(r"(\d+)$", str(sid))
            return (int(m.group(1)) if m else 10**9, str(sid))

        rows = rows.sort_values(by="step_id", key=lambda c: c.map(key_fn))
        return rows.to_dict(orient="records")

    if daily:
        # Calculate run count per product based on weeks in range
        weeks = max(1, int(round(date_count / 7)))
        runs_per_prod = max(1, int(round(weeks * max(0.1, runs_per_week))))
        recent_window_days = min(14, date_count)
        recent_threshold = end_date - timedelta(days=recent_window_days)

        run_counter = 0
        for pid in products["product_id"]:
            steps_for_pid = _step_order_for_product(pid)
            if not steps_for_pid:
                continue
            for _ in range(runs_per_prod):
                run_counter += 1
                run_id = f"{pid}-RUN-{run_counter:06d}"
                # assign a random start day in the window
                current_day = random.choice(all_days)
                planned_qty = 1 if unit_mode else random.randint(2, 10)
                planned_runs.append(
                    {
                        "run_id": run_id,
                        "product_id": pid,
                        "planned_qty": planned_qty,
                    }
                )

                in_recent_period = current_day >= recent_threshold
                inject_inprogress = in_recent_period or (random.random() < 0.1)
                made_inprogress = False
                for s in steps_for_pid:
                    if current_day.date() > end_date.date():
                        break
                    day_start, day_end = within_workday(current_day)
                    start_iso = rand_ts_iso(day_start, day_end)
                    start_dt = datetime.strptime(start_iso, "%Y-%m-%dT%H:%M:%SZ")
                    est_hours = float(s.get("estimated_time", 1) or 1)
                    jitter = random.uniform(0.6, 1.6)
                    dur_seconds = int(max(0.25, est_hours * jitter) * 3600)

                    status = "complete"
                    if (
                        not made_inprogress
                        and inject_inprogress
                        and current_day >= recent_threshold
                        and random.random() < 0.5
                    ):
                        status = "in_progress"
                        made_inprogress = True

                    if status == "complete":
                        end_dt = start_dt + timedelta(seconds=dur_seconds)
                        if end_dt > day_end:
                            end_dt = day_end
                        end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        ts_iso = end_iso
                    else:
                        end_iso = ""
                        ts_iso = start_iso

                    qty = 1 if unit_mode else random.randint(1, planned_qty)
                    production_log_rows.append(
                        {
                            "timestamp": ts_iso,
                            "start_time": start_iso,
                            "end_time": end_iso,
                            "line_id": random.choice(lines["line_id"]),
                            "product_id": s["product_id"],
                            "step_id": s["step_id"],
                            "run_id": run_id,
                            "quantity": qty,
                            "status": status,
                        }
                    )

                    # If this step is in progress, stop generating later steps for this run
                    if status == "in_progress":
                        break

                    # advance day with a small gap (bias toward 0-1 days)
                    current_day = current_day + timedelta(
                        days=random.choice([0, 1, 1, 2])
                    )
    else:
        # legacy scattered runs (uniform within full window)
        steps = process_steps.to_dict(orient="records")
        total_runs = len(all_days) * max(
            1, (min_steps_per_day + max_steps_per_day) // 2
        )
        for _ in range(total_runs):
            s = random.choice(steps)
            start_iso = rand_ts_iso(start_date, end_date)
            start_dt = datetime.strptime(start_iso, "%Y-%m-%dT%H:%M:%SZ")
            est_hours = float(s.get("estimated_time", 1) or 1)
            jitter = random.uniform(0.5, 1.5)
            dur_seconds = int(est_hours * jitter * 3600)
            status = random.choices(["complete", "in_progress"], weights=[0.8, 0.2])[0]
            if status == "complete":
                end_dt = start_dt + timedelta(seconds=dur_seconds)
                end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                ts_iso = end_iso
            else:
                end_iso = ""
                ts_iso = start_iso
            run_id = f"{s['product_id']}-RUN-{random.randint(1, 999999):06d}"
            qty = 1 if unit_mode else random.randint(1, 10)
            production_log_rows.append(
                {
                    "timestamp": ts_iso,
                    "start_time": start_iso,
                    "end_time": end_iso,
                    "line_id": random.choice(lines["line_id"]),
                    "product_id": s["product_id"],
                    "step_id": s["step_id"],
                    "run_id": run_id,
                    "quantity": qty,
                    "status": status,
                }
            )

    # If include_actual_machine requested, post-process production_log_rows to add actual_machine_id
    if include_actual_machine and process_steps is not None:
        # build lookup of assigned_machine per (product_id, step_id) and normalize blanks/'nan' -> None
        assign_map = {}
        if "product_id" in process_steps.columns and "step_id" in process_steps.columns:
            for _, rr in process_steps.reset_index(drop=True).iterrows():
                key = (rr.get("product_id"), rr.get("step_id"))
                am = rr.get("assigned_machine")
                if am is None:
                    assign_map[key] = None
                else:
                    s = str(am).strip()
                    if s == "" or s.lower() == "nan":
                        assign_map[key] = None
                    else:
                        assign_map[key] = s

        # prepare machine list and per-machine schedules
        machines_list = (
            list(machines["machine_id"]) if "machine_id" in machines.columns else []
        )
        machine_schedule = {m: [] for m in machines_list}

        def _parse_iso_or_none(val: str | None):
            if not val or (isinstance(val, str) and val.strip() == ""):
                return None
            try:
                return datetime.strptime(val, "%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                return None

        # Sort production_log rows by start_time so scheduling is deterministic and avoids
        # assigning a later-start job before an earlier-start job (which created overlaps).
        rows_with_dt: list[tuple[datetime | None, datetime | None, dict]] = []
        for r in production_log_rows:
            start = _parse_iso_or_none(r.get("start_time"))
            end = _parse_iso_or_none(r.get("end_time"))
            rows_with_dt.append((start, end, r))
        rows_with_dt.sort(key=lambda t: (t[0] is None, t[0] or datetime.min))

        for start, end, r in rows_with_dt:
            # if no start_time we cannot schedule a machine
            if start is None:
                r["actual_machine_id"] = ""
                continue

            # approximate missing end_time so open intervals don't block scheduling forever
            if end is None:
                end = start + timedelta(hours=1)

            # prefer the assigned_machine for the step (if valid), else try any machine
            preferred = assign_map.get((r.get("product_id"), r.get("step_id")))
            candidate_machines: list[str] = []
            if preferred and preferred in machines_list:
                candidate_machines.append(preferred)
            for m in machines_list:
                if m not in candidate_machines:
                    candidate_machines.append(m)

            chosen: str | None = None
            for m in candidate_machines:
                sched = machine_schedule.setdefault(m, [])
                # check for overlap: intervals intersect -> skip
                has_overlap = any(
                    not (end <= ex_s or start >= ex_e) for ex_s, ex_e in sched
                )
                if not has_overlap:
                    chosen = m
                    sched.append((start, end))
                    break

            r["actual_machine_id"] = chosen if chosen is not None else ""

    production_log = pd.DataFrame(production_log_rows)
    production_log.to_csv(DATA / "production_log.csv", index=False)

    # Runs: planned quantities
    if daily and planned_runs:
        runs_df = pd.DataFrame(planned_runs)
    elif not production_log.empty:
        runs_df = (
            production_log[["run_id", "product_id"]]
            .drop_duplicates()
            .assign(
                planned_qty=lambda d: [
                    (1 if unit_mode else random.randint(1, 10)) for _ in range(len(d))
                ]
            )
        )
    else:
        runs_df = pd.DataFrame(columns=["run_id", "product_id", "planned_qty"])
    runs_out = runs_df[
        [c for c in ["run_id", "product_id", "planned_qty"] if c in runs_df.columns]
    ].copy()
    runs_out.to_csv(DATA / "runs.csv", index=False)

    # Machine Metrics (ISO timestamps)
    metrics_rows = []
    # Generate 1-3 metrics per day to reflect daily monitoring
    for day in all_days:
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day.replace(hour=23, minute=59, second=59, microsecond=0)
        for _ in range(random.randint(1, 3)):
            metrics_rows.append(
                {
                    "timestamp": rand_ts_iso(day_start, day_end),
                    "machine_id": random.choice(machines["machine_id"]),
                    "metric_type": random.choice(
                        ["temperature", "throughput", "energy_consumption"]
                    ),
                    "metric_value": round(random.uniform(10, 100), 2),
                }
            )
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(DATA / "machine_metrics.csv", index=False)

    # Quality Checks (ISO timestamps)
    qc_rows = []
    # Sampling QC events across the window
    for day in all_days:
        if random.random() < 0.25:  # ~1-2 per week on average
            day_start = day.replace(hour=7, minute=0, second=0, microsecond=0)
            day_end = day.replace(hour=17, minute=0, second=0, microsecond=0)
            prod = random.choice(products["product_id"])
            # try to attach a run/step when available
            run_id = None
            step_id = None
            if not production_log.empty:
                # pick a random production_log row for this product if exists
                matches = production_log[production_log.product_id == prod]
                if not matches.empty:
                    sel = matches.sample(1).iloc[0]
                    run_id = sel.get("run_id")
                    step_id = sel.get("step_id")
            qc_rows.append(
                {
                    "timestamp": rand_ts_iso(day_start, day_end),
                    "product_id": prod,
                    "run_id": run_id,
                    "step_id": step_id,
                    "check_type": random.choice(
                        ["visual", "dimensional", "functional"]
                    ),
                    "result": random.choice(["pass", "fail"]),
                    "inspector_id": random.choice(operators["operator_id"]),
                }
            )

    # Guarantee at least one QC per run or per product if requested
    if qc_per_run and not production_log.empty:
        # ensure one QC per run: iterate unique runs and add one QC if none exist
        if qc_rows:
            existing_runs = set(
                pd.DataFrame(qc_rows).get("run_id").dropna().unique().tolist()
            )
        else:
            existing_runs = set()
        for run in production_log["run_id"].dropna().unique():
            if run not in existing_runs:
                # pick a representative production_log row for this run
                sel = production_log[production_log.run_id == run].iloc[0]
                day = datetime.strptime(sel.start_time, "%Y-%m-%dT%H:%M:%SZ")
                day_start = day.replace(hour=7, minute=0, second=0, microsecond=0)
                day_end = day.replace(hour=17, minute=0, second=0, microsecond=0)
                qc_rows.append(
                    {
                        "timestamp": rand_ts_iso(day_start, day_end),
                        "product_id": sel.product_id,
                        "run_id": run,
                        "step_id": sel.step_id,
                        "check_type": "visual",
                        "result": "pass",
                        "inspector_id": random.choice(operators["operator_id"]),
                    }
                )

    if qc_per_product and not production_log.empty:
        existing_products = set(
            [p for p in (pd.DataFrame(qc_rows).get("product_id") or []) if pd.notna(p)]
        )
        for prod in production_log["product_id"].dropna().unique():
            if prod not in existing_products:
                matches = production_log[production_log.product_id == prod]
                sel = matches.sample(1).iloc[0]
                day = datetime.strptime(sel.start_time, "%Y-%m-%dT%H:%M:%SZ")
                day_start = day.replace(hour=7, minute=0, second=0, microsecond=0)
                day_end = day.replace(hour=17, minute=0, second=0, microsecond=0)
                qc_rows.append(
                    {
                        "timestamp": rand_ts_iso(day_start, day_end),
                        "product_id": prod,
                        "run_id": sel.run_id,
                        "step_id": sel.step_id,
                        "check_type": "visual",
                        "result": "pass",
                        "inspector_id": random.choice(operators["operator_id"]),
                    }
                )

    quality_checks = pd.DataFrame(qc_rows)
    quality_checks.to_csv(DATA / "quality_checks.csv", index=False)

    # Labor Activities (derive from production_log rows)
    labor_rows = []
    if not production_log.empty:
        # Map each production_log to 1–2 labor activities
        for _, r in production_log.iterrows():
            base_start = datetime.strptime(r.start_time, "%Y-%m-%dT%H:%M:%SZ")
            if r.end_time:
                base_end = datetime.strptime(r.end_time, "%Y-%m-%dT%H:%M:%SZ")
            else:
                base_end = base_start + timedelta(minutes=random.randint(15, 120))

            duration = (base_end - base_start).total_seconds()
            splits = 1 if duration < 2700 or random.random() < 0.6 else 2
            operators_pool = []
            # Try to pull assigned_operators from process_steps
            try:
                ops_match = process_steps[
                    (process_steps.product_id == r.product_id)
                    & (process_steps.step_id == r.step_id)
                ]
                if not ops_match.empty:
                    ops = str(ops_match.iloc[0]["assigned_operators"]).split(",")
                    operators_pool = [o.strip() for o in ops if o.strip()]
            except Exception:
                pass
            if not operators_pool:
                operators_pool = list(operators["operator_id"])

            for sidx in range(splits):
                seg_start = base_start + timedelta(seconds=(duration / splits) * sidx)
                seg_end = (
                    base_end
                    if sidx == splits - 1
                    else seg_start + timedelta(seconds=(duration / splits))
                )
                act_type = random.choices(
                    ["direct", "setup", "rework", "indirect"],
                    weights=[0.65, 0.15, 0.05, 0.15],
                )[0]
                operator_id = random.choice(operators_pool)
                labor_rows.append(
                    {
                        "activity_id": f"ACT-{r.run_id or 'NA'}-{r.step_id}-{sidx}-{random.randint(1000,9999)}",
                        "operator_id": operator_id,
                        "product_id": r.product_id,
                        "step_id": r.step_id,
                        "run_id": r.run_id,
                        "line_id": r.line_id,
                        "start_time": seg_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "end_time": (
                            ""
                            if r.status == "in_progress" and sidx == splits - 1
                            else seg_end.strftime("%Y-%m-%dT%H:%M:%SZ")
                        ),
                        "activity_type": act_type,
                    }
                )

    labor_activities = pd.DataFrame(labor_rows)
    labor_activities.to_csv(DATA / "labor_activities.csv", index=False)

    print("Mock CSV dataset generated into data/ (schema-aligned).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock MOMAC SDM dataset")
    parser.add_argument(
        "--start", type=str, default=None, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--years", type=float, default=1.0, help="Span in years if start not provided"
    )
    parser.add_argument(
        "--daily", action="store_true", help="Generate daily activity coverage"
    )
    parser.add_argument(
        "--no-daily",
        dest="daily",
        action="store_false",
        help="Disable daily coverage (random scatter)",
    )
    parser.set_defaults(daily=True)
    parser.add_argument("--min-steps-per-day", type=int, default=3)
    parser.add_argument("--max-steps-per-day", type=int, default=8)
    parser.add_argument("--runs-per-week", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-machines", type=int, default=5)
    parser.add_argument("--num-lines", type=int, default=2)
    parser.add_argument("--num-products", type=int, default=3)
    parser.add_argument("--num-operators", type=int, default=4)
    parser.add_argument("--steps-per-product", type=int, default=4)
    parser.add_argument(
        "--unit-mode", action="store_true", help="Use unit quantity (1) per run"
    )
    parser.add_argument(
        "--batch-mode",
        dest="unit_mode",
        action="store_false",
        help="Random batch quantities per run",
    )
    parser.set_defaults(unit_mode=True)

    parser.add_argument(
        "--no-requires-machine",
        dest="include_requires_machine",
        action="store_false",
        help="Do not include requires_machine column in process_steps",
    )
    parser.set_defaults(include_requires_machine=True)

    parser.add_argument(
        "--actual-machine",
        dest="include_actual_machine",
        action="store_true",
        help="Include actual_machine_id in production_log rows",
    )
    parser.set_defaults(include_actual_machine=False)
    parser.add_argument(
        "--qc-per-run",
        dest="qc_per_run",
        action="store_true",
        help="Ensure at least one quality check row exists per run",
    )
    parser.set_defaults(qc_per_run=False)
    parser.add_argument(
        "--qc-per-product",
        dest="qc_per_product",
        action="store_true",
        help="Ensure at least one quality check row exists per product",
    )
    parser.set_defaults(qc_per_product=False)

    args = parser.parse_args()

    def _parse_date(s: str | None) -> datetime | None:
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            raise SystemExit(f"Invalid date format '{s}'. Use YYYY-MM-DD")

    generate_mock_data(
        num_machines=args.num_machines,
        num_lines=args.num_lines,
        num_products=args.num_products,
        num_operators=args.num_operators,
        steps_per_product=args.steps_per_product,
        start_date=_parse_date(args.start),
        end_date=_parse_date(args.end),
        years=args.years,
        daily=args.daily,
        min_steps_per_day=args.min_steps_per_day,
        max_steps_per_day=args.max_steps_per_day,
        runs_per_week=args.runs_per_week,
        seed=args.seed,
        unit_mode=args.unit_mode,
        include_requires_machine=args.include_requires_machine,
        include_actual_machine=args.include_actual_machine,
        qc_per_run=args.qc_per_run,
        qc_per_product=args.qc_per_product,
    )
