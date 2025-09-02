import pandas as pd
from typing import List, Dict, Set
from .step import Step


def validate_dependencies(steps: List[Step]) -> List[str]:
    """
    Validate workflow dependencies within each product.

    Rules:
    - Dependencies are scoped by product_id (no cross-product deps).
    - Each step may have dependency_step_id = None (start) or a single parent step_id.
    - Report:
        * Missing refs: dependency points to a step_id not present in the same product.
        * Cycles: any cycle in the dependency graph (e.g., S1->S2->S1).
    - Deterministic output: errors are sorted for stable test expectations.

    Returns:
        List[str]: human-readable error messages. Empty list means valid.
    """
    errors: List[str] = []

    # Group steps by product for isolated validation
    by_product: Dict[str, List[Step]] = {}
    for s in steps:
        # Guard: skip steps missing essentials (optional, but keeps DFS safe)
        if not getattr(s, "product_id", None) or not getattr(s, "step_id", None):
            errors.append(f"MISSING_KEY: product_id/step_id missing on step={s!r}")
            continue
        by_product.setdefault(s.product_id, []).append(s)

    # Validate each product independently
    for product_id, plist in by_product.items():
        # Map: step_id -> Step (for quick existence checks)
        nodes: Dict[str, Step] = {str(p.step_id): p for p in plist}

        # Prepare adjacency: child -> parent (single-parent edges)
        # Also collect missing dependency references
        edges: Dict[str, str] = {}
        for p in plist:
            dep = getattr(p, "dependency_step_id", None)
            if dep is None or (isinstance(dep, str) and dep.strip() == ""):
                continue  # Start step (no dependency)
            dep_id = str(dep)
            child_id = str(p.step_id)
            if dep_id not in nodes:
                errors.append(
                    f"MISSING_DEP: product={product_id} step={child_id} depends_on={dep_id} (not found)"
                )
            else:
                edges[child_id] = dep_id

        # DFS-based cycle detection
        # states: 0 = UNVISITED, 1 = VISITING, 2 = VISITED
        state: Dict[str, int] = {sid: 0 for sid in nodes.keys()}

        # To reconstruct cycles, keep a parent pointer (reverse edge we traverse)
        parent: Dict[str, str] = {}

        def dfs(u: str) -> None:
            state[u] = 1  # VISITING
            if u in edges:
                v = edges[u]  # u -> v
                if v not in state:
                    # If the parent node is missing, we already reported MISSING_DEP above; skip.
                    pass
                else:
                    if state[v] == 0:  # UNVISITED
                        parent[v] = u
                        dfs(v)
                    elif state[v] == 1:
                        # Found a back-edge (cycle). Reconstruct cycle path u -> ... -> v -> u
                        cycle = [v]
                        cur = u
                        while cur != v and cur in parent:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.append(v)  # close the loop for readability
                        cycle_path = "->".join(cycle[::-1])  # start at v, end back at v
                        errors.append(
                            f"CYCLE: product={product_id} path={cycle_path}"
                        )
            state[u] = 2  # VISITED

        for sid in nodes.keys():
            if state[sid] == 0:
                dfs(sid)

    # Deterministic ordering of messages (useful for tests)
    errors.sort()
    return errors

def steps_from_dataframe(df: pd.DataFrame) -> List[Step]:
    """
    Convert a process_steps DataFrame into a list of Step objects.

    Rules:
    - assigned_operators: split on commas, strip whitespace, drop empties, dedupe while preserving order.
    - estimated_time: coerced to int >= 0 (invalid or negative -> 0).
    - dependency_step_id: empty/NaN/whitespace -> None.
    - product_id and step_id: must be non-empty strings; rows missing them are skipped.
    - step_name: falls back to step_id if blank.
    - Keeps first instance if duplicate (product_id, step_id) rows appear.

    Args:
        df: DataFrame with columns
            [product_id, step_id, step_name, assigned_machine,
             assigned_operators, estimated_time, dependency_step_id]

    Returns:
        List[Step]: normalized Step objects.
    """
    if df is None or df.empty:
        return []

    steps: List[Step] = []
    seen_keys = set()

    for _, row in df.iterrows():
        product_id = str(row.get("product_id", "")).strip()
        step_id = str(row.get("step_id", "")).strip()

        # Skip rows missing essential keys
        if not product_id or not step_id:
            continue

        # Deduplicate on (product_id, step_id)
        key = (product_id, step_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Step name (fallback to step_id)
        step_name = str(row.get("step_name", "")).strip()
        if not step_name:
            step_name = step_id

        # Assigned machine
        assigned_machine = str(row.get("assigned_machine", "")).strip()

        # Operators: split, strip, dedupe while preserving order
        ops_raw = str(row.get("assigned_operators", "") or "")
        ops = []
        for op in [o.strip() for o in ops_raw.split(",")]:
            if op and op not in ops:
                ops.append(op)

        # Estimated time: coerce to int >= 0
        try:
            est_time = int(float(row.get("estimated_time", 0)))
            if est_time < 0:
                est_time = 0
        except Exception:
            est_time = 0

        # Dependency: normalize to None if blank
        dep_raw = row.get("dependency_step_id", None)
        dep = None
        if pd.notna(dep_raw):
            dep_str = str(dep_raw).strip()
            dep = dep_str if dep_str else None

        steps.append(
            Step(
                product_id=product_id,
                step_id=step_id,
                step_name=step_name,
                assigned_machine=assigned_machine,
                assigned_operators=ops,
                estimated_time=est_time,
                dependency_step_id=dep,
            )
        )

    return steps
