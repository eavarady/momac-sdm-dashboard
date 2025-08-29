import pandas as pd
import random
from datetime import datetime, timedelta


# --- Helper Functions ---
def random_timestamp(start, end):
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def random_step_status():
    return random.choice(["not_started", "in_progress", "completed"])


# --- Configuration ---
num_machines = 5
num_lines = 2
num_products = 3
num_operators = 4
num_steps_per_product = 4
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 8, 29)

# --- Dimension Tables ---

# Machines
machines = pd.DataFrame(
    {
        "machine_id": [f"MX-{i+101}" for i in range(num_machines)],
        "line_id": [f"LINE-{chr(65+i%num_lines)}" for i in range(num_machines)],
        "type": [
            random.choice(["CNC", "RobotArm", "Press"]) for _ in range(num_machines)
        ],
        "status": [
            random.choice(["active", "maintenance", "offline"])
            for _ in range(num_machines)
        ],
    }
)
machines.to_csv("machines.csv", index=False)

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
operators.to_csv("operators.csv", index=False)

# Production Lines (with line manager)
lines = pd.DataFrame(
    {
        "line_id": [f"LINE-{chr(65+i)}" for i in range(num_lines)],
        "name": [f"Assembly Line {chr(65+i)}" for i in range(num_lines)],
        "shift": [random.choice(["day", "night"]) for _ in range(num_lines)],
        "line_manager_id": [
            random.choice(operators["operator_id"]) for _ in range(num_lines)
        ],
    }
)
lines.to_csv("production_lines.csv", index=False)

# Products
products = pd.DataFrame(
    {
        "product_id": [f"PRD-{i+1}" for i in range(num_products)],
        "name": [f"Generator {i+1}" for i in range(num_products)],
        "category": ["Generator"] * num_products,
        "spec_version": [f"v{random.randint(1,3)}.0" for _ in range(num_products)],
    }
)
products.to_csv("products.csv", index=False)

# --- Fact Tables ---

# Process Steps
process_steps_list = []
step_names = ["Unpack", "Assemble", "Test", "Repack"]
for prod_id in products["product_id"]:
    for idx, step_name in enumerate(step_names):
        process_steps_list.append(
            {
                "product_id": prod_id,
                "step_id": f"{prod_id}-STEP-{idx+1}",
                "step_name": step_name,
                "assigned_machine": random.choice(machines["machine_id"]),
                "assigned_operators": ",".join(
                    random.sample(
                        list(operators["operator_id"]), k=random.randint(1, 2)
                    )
                ),
                "estimated_time_min": random.randint(10, 60),
                "dependency_step_id": f"{prod_id}-STEP-{idx}" if idx > 0 else "",
            }
        )
process_steps = pd.DataFrame(process_steps_list)
process_steps.to_csv("process_steps.csv", index=False)

# Production Log (linked to steps)
prod_log_list = []
for _, row in process_steps.iterrows():
    prod_log_list.append(
        {
            "timestamp": random_timestamp(start_date, end_date),
            "line_id": random.choice(lines["line_id"]),
            "line_manager_id": lines.loc[
                lines["line_id"] == row.get("line_id", random.choice(lines["line_id"])),
                "line_manager_id",
            ].values[0],
            "product_id": row["product_id"],
            "step_id": row["step_id"],
            "quantity": random.randint(1, 10),
            "status": random_step_status(),
        }
    )
production_log = pd.DataFrame(prod_log_list)
production_log.to_csv("production_log.csv", index=False)

# Machine Metrics
metrics_list = []
for _ in range(50):
    metrics_list.append(
        {
            "timestamp": random_timestamp(start_date, end_date),
            "machine_id": random.choice(machines["machine_id"]),
            "metric_type": random.choice(
                ["temperature", "throughput", "energy_consumption"]
            ),
            "metric_value": round(random.uniform(10, 100), 2),
        }
    )
metrics = pd.DataFrame(metrics_list)
metrics.to_csv("machine_metrics.csv", index=False)

# Quality Checks
quality_checks_list = []
for _ in range(20):
    quality_checks_list.append(
        {
            "timestamp": random_timestamp(start_date, end_date),
            "product_id": random.choice(products["product_id"]),
            "check_type": random.choice(["visual", "dimensional", "functional"]),
            "result": random.choice(["pass", "fail"]),
            "inspector_id": random.choice(operators["operator_id"]),
        }
    )
quality_checks = pd.DataFrame(quality_checks_list)
quality_checks.to_csv("quality_checks.csv", index=False)

print("Mock CSV dataset generated successfully!")
