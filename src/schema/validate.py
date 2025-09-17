from __future__ import annotations
from typing import Iterable, Type, Tuple, Dict, List
import pandas as pd
from pydantic import BaseModel, ValidationError

from schema.models import (
    MachineRow,
    ProductionLineRow,
    ProductRow,
    OperatorRow,
    ProcessStepRow,
    ProductionLogRow,
    MachineMetricRow,
    QualityCheckRow,
    ProductionTargetRow,
    RunsRow,
)

# Map logical table name -> (pydantic model, required columns for friendly messages)
TABLE_REGISTRY: Dict[str, Tuple[Type[BaseModel], Tuple[str, ...]]] = {
    "machines": (MachineRow, ("machine_id", "line_id", "type", "status")),
    "production_lines": (ProductionLineRow, ("line_id", "name", "shift")),
    "products": (ProductRow, ("product_id", "name", "category", "spec_version")),
    "operators": (OperatorRow, ("operator_id", "name", "role")),
    "process_steps": (
        ProcessStepRow,
        (
            "product_id",
            "step_id",
            "step_name",
            "assigned_machine",
            "assigned_operators",
            "estimated_time",
            "dependency_step_id",
        ),
    ),
    "production_log": (
        ProductionLogRow,
        (
            "timestamp",
            "start_time",
            "end_time",
            "line_id",
            "product_id",
            "step_id",
            "quantity",
            "status",
        ),
    ),
    "machine_metrics": (
        MachineMetricRow,
        ("timestamp", "machine_id", "metric_type", "metric_value"),
    ),
    "quality_checks": (
        QualityCheckRow,
        ("timestamp", "product_id", "check_type", "result", "inspector_id"),
    ),
    # Strictly run-based targets: require run_id and target_qty (product_id optional/informational).
    "production_targets": (ProductionTargetRow, ("run_id", "target_qty")),
    # Preferred runs table: planned quantities per run (supersedes production_targets)
    "runs": (RunsRow, ("run_id", "planned_qty")),
}


def validate_dataframe(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """
    Row-level validation using Pydantic. Returns a normalized DataFrame.
    Raises ValueError with aggregated, friendly messages if any row is invalid.
    """
    if table not in TABLE_REGISTRY:
        raise ValueError(f"Unknown table '{table}'")

    model, req_cols = TABLE_REGISTRY[table]

    # Check required columns early for clearer errors
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing required columns: {missing}")

    records = df.to_dict(orient="records")
    normalized: list[dict] = []
    errors: list[str] = []

    for i, rec in enumerate(records, start=1):
        try:
            obj = model(**rec)
            normalized.append(obj.model_dump())
        except ValidationError as ve:
            for err in ve.errors():
                loc = ".".join(str(x) for x in err.get("loc", []) if x is not None)
                msg = err.get("msg", "invalid")
                bad = rec.get(loc, None)
                errors.append(f"{table}: row {i} field '{loc}': {msg} (value={bad!r})")

    if errors:
        # Limit extremely noisy output
        head = "\n".join(errors[:100])
        more = "" if len(errors) <= 100 else f"\n... and {len(errors)-100} more"
        raise ValueError(f"Validation failed for {table}:\n{head}{more}")

    return pd.DataFrame.from_records(normalized)


def check_uniques_and_fks(tables: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Table-level validation: uniqueness and cross-table foreign keys.
    Returns a list of error strings (empty if OK).
    """
    errs: list[str] = []

    def dup_errors(df: pd.DataFrame, cols: list[str], label: str):
        if not set(cols).issubset(df.columns):
            return
        dups = df.duplicated(subset=cols, keep=False)
        if dups.any():
            bad = df.loc[dups, cols].astype(str).drop_duplicates()
            for _, r in bad.iterrows():
                key = ", ".join(f"{c}={r[c]!r}" for c in cols)
                errs.append(f"DUP_KEY {label}: {key}")

    # Uniques
    if "machines" in tables:
        dup_errors(tables["machines"], ["machine_id"], "machines.machine_id")
    if "production_lines" in tables:
        dup_errors(tables["production_lines"], ["line_id"], "production_lines.line_id")
    if "products" in tables:
        dup_errors(tables["products"], ["product_id"], "products.product_id")
    if "operators" in tables:
        dup_errors(tables["operators"], ["operator_id"], "operators.operator_id")
    if "process_steps" in tables:
        dup_errors(
            tables["process_steps"],
            ["product_id", "step_id"],
            "process_steps.(product_id,step_id)",
        )
    if "production_targets" in tables:
        dup_errors(
            tables["production_targets"],
            ["run_id"],
            "production_targets.run_id",
        )
    if "runs" in tables:
        dup_errors(
            tables["runs"],
            ["run_id"],
            "runs.run_id",
        )

    # FKs
    if {"production_log", "process_steps"} <= tables.keys():
        steps_keys = set(
            zip(
                tables["process_steps"]["product_id"].astype(str),
                tables["process_steps"]["step_id"].astype(str),
            )
        )
        for i, r in tables["production_log"].reset_index(drop=True).iterrows():
            key = (str(r.get("product_id", "")), str(r.get("step_id", "")))
            if key not in steps_keys:
                errs.append(
                    f"FK production_log row {i+1}: step not found in process_steps (product_id={key[0]!r}, step_id={key[1]!r})"
                )

    if {"production_targets", "production_log"} <= tables.keys():
        runset = set(
            tables["production_log"]["run_id"].dropna().astype(str).unique().tolist()
        )
        for i, r in tables["production_targets"].reset_index(drop=True).iterrows():
            rid = str(r.get("run_id", "") or "")
            if rid not in runset:
                errs.append(
                    f"FK production_targets row {i+1}: run_id {rid!r} not found in production_log"
                )
    if {"runs", "production_log"} <= tables.keys():
        runset = set(
            tables["production_log"]["run_id"].dropna().astype(str).unique().tolist()
        )
        for i, r in tables["runs"].reset_index(drop=True).iterrows():
            rid = str(r.get("run_id", "") or "")
            if rid not in runset:
                errs.append(
                    f"FK runs row {i+1}: run_id {rid!r} not found in production_log"
                )

    # Unit-mode quantity guard: prefer runs table; fallback to production_targets
    if {"runs", "production_log"} <= tables.keys():
        try:
            rn = tables["runs"]["run_id"].astype(str)
            pq = (
                pd.to_numeric(tables["runs"]["planned_qty"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            unit_runs = set(rn[pq == 1].tolist())
            if unit_runs:
                pl = tables["production_log"][["run_id", "quantity"]].copy()
                pl["quantity"] = (
                    pd.to_numeric(pl["quantity"], errors="coerce").fillna(0).astype(int)
                )
                bad = pl[
                    pl["run_id"].astype(str).isin(unit_runs) & (pl["quantity"] != 1)
                ]
                if not bad.empty:
                    bad_runs = bad["run_id"].astype(str).unique().tolist()
                    errs.append(
                        f"UNIT_MODE_QUANTITY: runs planned_qty=1 must have event quantity==1 (violations in run_id(s): {bad_runs[:20]})"
                    )
        except Exception:
            pass
    elif {"production_targets", "production_log"} <= tables.keys():
        try:
            pt = tables["production_targets"][["run_id", "target_qty"]].copy()
            pt["target_qty"] = (
                pd.to_numeric(pt["target_qty"], errors="coerce").fillna(0).astype(int)
            )
            unit_runs = set(
                pt.loc[pt["target_qty"] == 1, "run_id"].dropna().astype(str)
            )
            if unit_runs:
                pl = tables["production_log"][["run_id", "quantity"]].copy()
                pl["quantity"] = (
                    pd.to_numeric(pl["quantity"], errors="coerce").fillna(0).astype(int)
                )
                bad = pl[
                    pl["run_id"].astype(str).isin(unit_runs) & (pl["quantity"] != 1)
                ]
                if not bad.empty:
                    bad_runs = bad["run_id"].astype(str).unique().tolist()
                    errs.append(
                        f"UNIT_MODE_QUANTITY: runs planned_qty=1 must have event quantity==1 (violations in run_id(s): {bad_runs[:20]})"
                    )
        except Exception:
            # non-fatal; schema-level validation shouldn't crash if optional tables or columns shift
            pass

    if {"machine_metrics", "machines"} <= tables.keys():
        mset = set(tables["machines"]["machine_id"].astype(str))
        for i, r in tables["machine_metrics"].reset_index(drop=True).iterrows():
            mid = str(r.get("machine_id", ""))
            if mid not in mset:
                errs.append(
                    f"FK machine_metrics row {i+1}: machine_id {mid!r} not found in machines"
                )

    if {"quality_checks", "products"} <= tables.keys():
        pset = set(tables["products"]["product_id"].astype(str))
        for i, r in tables["quality_checks"].reset_index(drop=True).iterrows():
            pid = str(r.get("product_id", ""))
            if pid not in pset:
                errs.append(
                    f"FK quality_checks row {i+1}: product_id {pid!r} not found in products"
                )

    # Reuse workflow dependency validator if present
    try:
        from workflow.validator import steps_from_dataframe, validate_dependencies

        if "process_steps" in tables:
            steps = steps_from_dataframe(tables["process_steps"])
            dep_errs = validate_dependencies(steps)
            errs.extend(f"WORKFLOW {e}" for e in dep_errs)
    except Exception:
        # Don’t hard-fail if workflow module changes; schema layer should still work.
        pass

    return sorted(errs)
