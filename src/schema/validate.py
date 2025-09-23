from __future__ import annotations
from typing import Iterable, Type, Tuple, Dict, List
import pandas as pd
from datetime import datetime, timezone, timedelta
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
    RunsRow,
    LaborActivityRow,  # <-- add
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
            "requires_machine",
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
    # Preferred runs table: planned quantities per run (supersedes production_targets)
    "runs": (RunsRow, ("run_id", "planned_qty")),
    "labor_activities": (
        LaborActivityRow,
        (
            "activity_id",
            "operator_id",
            "product_id",
            "step_id",
            "run_id",
            "line_id",
            "start_time",
            "end_time",
            "activity_type",
        ),
    ),
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
    # Backward-compat: if process_steps omits 'requires_machine', inject default True so the
    # required-column check passes but callers are warned via downstream checks if needed.
    if table == "process_steps" and "requires_machine" not in df.columns:
        df = df.copy()
        df["requires_machine"] = True

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
    if "runs" in tables:
        dup_errors(
            tables["runs"],
            ["run_id"],
            "runs.run_id",
        )
    if "labor_activities" in tables:  # uniqueness
        dup_errors(
            tables["labor_activities"],
            ["activity_id"],
            "labor_activities.activity_id",
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

    # Unit-mode quantity guard based on runs
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

    if {"labor_activities", "operators"} <= tables.keys():
        op_set = set(tables["operators"]["operator_id"].astype(str))
        for i, r in tables["labor_activities"].reset_index(drop=True).iterrows():
            oid = str(r.get("operator_id", ""))
            if oid not in op_set:
                errs.append(
                    f"FK labor_activities row {i+1}: operator_id {oid!r} not found in operators"
                )

    if {"labor_activities", "products"} <= tables.keys():
        pset = set(tables["products"]["product_id"].astype(str))
        for i, r in tables["labor_activities"].reset_index(drop=True).iterrows():
            pid = str(r.get("product_id", ""))
            if pid not in pset:
                errs.append(
                    f"FK labor_activities row {i+1}: product_id {pid!r} not found in products"
                )

    if {"labor_activities", "process_steps"} <= tables.keys():
        step_keys = set(
            zip(
                tables["process_steps"]["product_id"].astype(str),
                tables["process_steps"]["step_id"].astype(str),
            )
        )
        for i, r in tables["labor_activities"].reset_index(drop=True).iterrows():
            key = (str(r.get("product_id", "")), str(r.get("step_id", "")))
            if key not in step_keys:
                errs.append(
                    f"FK labor_activities row {i+1}: step not found in process_steps (product_id={key[0]!r}, step_id={key[1]!r})"
                )

    if {"labor_activities", "runs"} <= tables.keys():
        run_set = set(tables["runs"]["run_id"].astype(str))
        for i, r in tables["labor_activities"].reset_index(drop=True).iterrows():
            rid = r.get("run_id")
            if pd.notna(rid) and str(rid) not in run_set:
                errs.append(
                    f"FK labor_activities row {i+1}: run_id {str(rid)!r} not found in runs"
                )

    if {"labor_activities", "production_lines"} <= tables.keys():
        line_set = set(tables["production_lines"]["line_id"].astype(str))
        for i, r in tables["labor_activities"].reset_index(drop=True).iterrows():
            lid = r.get("line_id")
            if pd.notna(lid) and str(lid) not in line_set:
                errs.append(
                    f"FK labor_activities row {i+1}: line_id {str(lid)!r} not found in production_lines"
                )

    # --- Planning sanity & extra warnings ------------------------------------------------
    # Process steps: requires_machine / assigned_machine consistency + assigned_operators existence
    if "process_steps" in tables:
        try:
            ps = tables["process_steps"].reset_index(drop=True)
            op_set = (
                set(tables["operators"]["operator_id"].astype(str))
                if "operators" in tables
                else set()
            )
            for i, r in ps.iterrows():
                requires = bool(r.get("requires_machine", True))
                assigned = r.get("assigned_machine")
                # assigned might be NaN -> normalize
                # assigned might be NaN or the literal string 'nan' from CSVs -> normalize
                if pd.isna(assigned) or (
                    isinstance(assigned, str) and assigned.strip().lower() == "nan"
                ):
                    assigned = None
                if not requires and assigned:
                    errs.append(
                        f"WARN process_steps row {i+1}: requires_machine=False but assigned_machine is set ({assigned!r})"
                    )
                if requires and not assigned:
                    errs.append(
                        f"ERR process_steps row {i+1}: requires_machine=True but assigned_machine is empty"
                    )
                # assigned_operators must exist -> warning
                ops = r.get("assigned_operators", [])
                # accept lists or comma strings
                if isinstance(ops, str):
                    ops_list = [x.strip() for x in ops.split(",") if x.strip()]
                elif isinstance(ops, (list, tuple)):
                    ops_list = [str(x).strip() for x in ops if str(x).strip()]
                else:
                    ops_list = []
                for op in ops_list:
                    if op and op_set and op not in op_set:
                        errs.append(
                            f"WARN process_steps row {i+1}: assigned operator {op!r} not found in operators"
                        )
        except Exception:
            pass

    # --- Overlap checks -------------------------------------------------------------
    # Operators: overlapping labor_activity intervals per operator -> default ERROR
    if "labor_activities" in tables:
        try:
            la = tables["labor_activities"].reset_index(drop=True)
            # group by operator
            for op, grp in la.groupby(la["operator_id"].astype(str), dropna=False):
                intervals: list[tuple[datetime, datetime, int, str]] = []
                for idx, row in grp.reset_index(drop=True).iterrows():
                    start = row.get("start_time")
                    end = row.get("end_time")
                    # normalize pd.NaT etc.
                    if pd.isna(start):
                        continue
                    # If end_time is missing, approximate as start + 1 hour to avoid marking
                    # benign in-progress or legacy-missing rows as permanently open intervals.
                    # This reduces false positive overlaps in mock/generated datasets.
                    if pd.isna(end) or end is None:
                        end = start + timedelta(hours=1)
                    intervals.append(
                        (
                            start,
                            end,
                            int(row.name) + 1 if hasattr(row, "name") else idx + 1,
                            row.get("activity_id"),
                        )
                    )
                if len(intervals) < 2:
                    continue
                intervals.sort(key=lambda x: x[0])
                prev_s, prev_e, prev_rownum, prev_aid = intervals[0]
                for cur_s, cur_e, cur_rownum, cur_aid in intervals[1:]:
                    if cur_s < prev_e:
                        errs.append(
                            f"OVERLAP operator {op!r}: activity rows {prev_rownum}({prev_aid}) and {cur_rownum}({cur_aid}) have overlapping intervals"
                        )
                    if cur_e > prev_e:
                        prev_s, prev_e, prev_rownum, prev_aid = (
                            cur_s,
                            cur_e,
                            cur_rownum,
                            cur_aid,
                        )
        except Exception:
            pass

    # Machines: derive machine_id per production_log row (actual_machine_id or assigned_machine from process_steps)
    if "production_log" in tables:
        try:
            pl = tables["production_log"].reset_index(drop=True)
            # build mapping from (product_id,step_id) to assigned_machine
            step_map = {}
            if "process_steps" in tables:
                for _, r in tables["process_steps"].reset_index(drop=True).iterrows():
                    key = (str(r.get("product_id") or ""), str(r.get("step_id") or ""))
                    am = r.get("assigned_machine")
                    if pd.isna(am):
                        am = None
                    step_map[key] = am
            machine_intervals: Dict[str, list[tuple[datetime, datetime, int]]] = {}
            for idx, r in pl.iterrows():
                start = r.get("start_time")
                end = r.get("end_time")
                if pd.isna(start):
                    continue
                # If end_time is missing, approximate to start + 1 hour to avoid treating it
                # as an infinite open interval which would overlap everything.
                if pd.isna(end) or end is None:
                    end = start + timedelta(hours=1)
                mach = r.get("actual_machine_id")
                if pd.isna(mach) or mach is None:
                    # fallback to process_steps.assigned_machine
                    key = (str(r.get("product_id") or ""), str(r.get("step_id") or ""))
                    mach = step_map.get(key)
                if mach is None:
                    # no machine assigned/detected; skip but could warn
                    continue
                # Normalize machines that may be the string 'nan'
                if isinstance(mach, str) and mach.strip().lower() == "nan":
                    continue
                mach = str(mach)
                machine_intervals.setdefault(mach, []).append((start, end, idx + 1))
            # detect overlaps per machine
            for mach, ivs in machine_intervals.items():
                ivs.sort(key=lambda x: x[0])
                prev_s, prev_e, prev_row = ivs[0]
                for cur_s, cur_e, cur_row in ivs[1:]:
                    if cur_s < prev_e:
                        errs.append(
                            f"OVERLAP machine {mach!r}: production_log rows {prev_row} and {cur_row} have overlapping intervals"
                        )
                    if cur_e > prev_e:
                        prev_s, prev_e, prev_row = cur_s, cur_e, cur_row
        except Exception:
            pass

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
