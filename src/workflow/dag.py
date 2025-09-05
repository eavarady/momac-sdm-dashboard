import pandas as pd
from typing import Optional
from .validator import steps_from_dataframe, validate_dependencies


def planned_finish_offsets(process_steps: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute planned finish offsets (seconds) per (product_id, step_id) using Kahn's algorithm.

    Returns None if the dependency graph is invalid per workflow.validator.
    """
    if process_steps is None or process_steps.empty:
        return pd.DataFrame(
            columns=["product_id", "step_id", "planned_finish_offset_sec"]
        )

    # Validate DAG first using existing validator utilities
    steps = steps_from_dataframe(process_steps)
    errors = validate_dependencies(steps)
    if errors:
        return None

    ps = process_steps[
        ["product_id", "step_id", "estimated_time", "dependency_step_id"]
    ].copy()
    ps["planned_dur_sec"] = (
        pd.to_numeric(ps["estimated_time"], errors="coerce").fillna(0.0) * 3600.0
    )

    out_frames = []
    for product_id, g in ps.groupby("product_id"):
        g = g.reset_index(drop=True)
        dur = dict(zip(g["step_id"], g["planned_dur_sec"]))

        # Build graph: parent -> children, indegree counts
        children: dict[str, list[str]] = {}
        indeg: dict[str, int] = {str(sid): 0 for sid in g["step_id"]}

        for _, row in g.iterrows():
            child = str(row["step_id"])
            parent = row.get("dependency_step_id", None)
            if (
                pd.isna(parent)
                or (isinstance(parent, str) and parent.strip() == "")
                or parent is None
            ):
                continue
            parent = str(parent)
            children.setdefault(parent, []).append(child)
            indeg[child] = indeg.get(child, 0) + 1
            indeg.setdefault(parent, indeg.get(parent, 0))

        # Kahn queue and longest-path accumulation (finish offsets)
        queue = [sid for sid, d in indeg.items() if d == 0]
        finish_offset: dict[str, float] = {
            sid: float(dur.get(sid, 0.0)) for sid in queue
        }

        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in children.get(u, []):
                cand = float(finish_offset.get(u, 0.0)) + float(dur.get(v, 0.0))
                finish_offset[v] = max(float(finish_offset.get(v, 0.0)), cand)
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)

        # Ensure every node has an entry
        for sid in g["step_id"]:
            s = str(sid)
            finish_offset.setdefault(s, float(dur.get(sid, 0.0)))

        out_frames.append(
            pd.DataFrame(
                {
                    "product_id": product_id,
                    "step_id": list(finish_offset.keys()),
                    "planned_finish_offset_sec": list(finish_offset.values()),
                }
            )
        )

    return (
        pd.concat(out_frames, ignore_index=True)
        if out_frames
        else pd.DataFrame(
            columns=["product_id", "step_id", "planned_finish_offset_sec"]
        )
    )
