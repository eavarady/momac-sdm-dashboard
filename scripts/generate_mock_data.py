"""
Consolidated mock data generator (new schema only).

Outputs CSVs under data/ with the enforced schema:
- process_steps: estimated_time (hours)
- production_log: columns [timestamp, start_time, end_time, line_id, product_id, step_id, run_id, quantity, status]
  where status in {'in_progress','complete'} and timestamp is ISO 8601 Z
- machine_metrics, quality_checks: timestamps ISO 8601 Z
"""

from pathlib import Path
from datetime import datetime, timedelta
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
    seed: int | None = 42,
    runs_min: int = 6,  # keep higher run counts
    runs_max: int = 12,  # keep higher run counts
):
    if seed is not None:
        random.seed(seed)

    start_date = start_date or datetime(2025, 1, 1)
    end_date = end_date or datetime(2025, 8, 29)

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
            process_steps_rows.append(
                {
                    "product_id": prod_id,
                    "step_id": step_id,
                    "step_name": step_names[idx % len(step_names)],
                    "assigned_machine": random.choice(machines["machine_id"]),
                    "assigned_operators": ",".join(
                        random.sample(
                            list(operators["operator_id"]), k=random.randint(1, 2)
                        )
                    ),
                    "estimated_time": random.randint(1, 8),  # hours
                    "dependency_step_id": dep,
                }
            )
    process_steps = pd.DataFrame(process_steps_rows)
    process_steps.to_csv(DATA / "process_steps.csv", index=False)

    # Production Log (add start_time/end_time; keep legacy timestamp for compatibility)
    def rand_status() -> str:
        # Unbiased 50/50 selection
        return random.choice(["complete", "in_progress"])

    production_log_rows = []
    # Increase runs per product (previously 1–3)
    runs_per_product = {
        pid: random.randint(runs_min, runs_max) for pid in products["product_id"]
    }
    for _, s in process_steps.iterrows():
        pid = s["product_id"]
        for run_idx in range(1, runs_per_product[pid] + 1):
            run_id = f"{pid}-RUN-{run_idx}"
            # choose a start_time within window
            start_iso = rand_ts_iso(start_date, end_date)
            start_dt = datetime.strptime(start_iso, "%Y-%m-%dT%H:%M:%SZ")
            status = rand_status()
            qty = random.randint(1, 10)
            # derive a rough duration around estimated_time (0.5x - 1.5x hours)
            est_hours = float(s.get("estimated_time", 1) or 1)
            jitter = random.uniform(0.5, 1.5)
            dur_seconds = int(est_hours * jitter * 3600)
            if status == "complete":
                end_dt = start_dt + timedelta(seconds=dur_seconds)
                end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                # legacy timestamp = completion time for complete rows
                ts_iso = end_iso
            else:
                end_iso = ""
                # legacy timestamp = start time for in_progress rows
                ts_iso = start_iso
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
    production_log = pd.DataFrame(production_log_rows)
    production_log.to_csv(DATA / "production_log.csv", index=False)

    # Production Targets (run-based, default target_qty=1 unless provided elsewhere)
    if not production_log.empty:
        pt_df = (
            production_log[["run_id", "product_id"]]
            .drop_duplicates()
            .assign(target_qty=random.randint(1, 10))
        )
        pt_df.to_csv(DATA / "production_targets.csv", index=False)

    # Machine Metrics (ISO timestamps)
    metrics_rows = []
    for _ in range(50):
        metrics_rows.append(
            {
                "timestamp": rand_ts_iso(start_date, end_date),
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
    for _ in range(20):
        qc_rows.append(
            {
                "timestamp": rand_ts_iso(start_date, end_date),
                "product_id": random.choice(products["product_id"]),
                "check_type": random.choice(["visual", "dimensional", "functional"]),
                "result": random.choice(["pass", "fail"]),
                "inspector_id": random.choice(operators["operator_id"]),
            }
        )
    quality_checks = pd.DataFrame(qc_rows)
    quality_checks.to_csv(DATA / "quality_checks.csv", index=False)

    print("Mock CSV dataset generated into data/ (schema-aligned).")


if __name__ == "__main__":
    generate_mock_data(
        runs_min=6,
        runs_max=12,
    )
