from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from graphviz import Digraph
from workflow.validator import steps_from_dataframe, validate_dependencies


def _wrap_label(text: str, max_chars: int = 16) -> str:
    try:
        words = str(text or "").split()
        if not words:
            return ""
        lines: List[str] = []
        cur = words[0]
        for w in words[1:]:
            if len(cur) + 1 + len(w) <= max_chars:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return "\n".join(lines)
    except Exception:
        return str(text or "")


def build_step_dependency_graph(process_steps_df: pd.DataFrame) -> Tuple[Digraph, List[str]]:
    """Build a compact, readable dependency DAG for process steps.

    - Components are stacked vertically (one column).
    - Steps inside each component flow left-to-right.
    - Long titles are wrapped at word boundaries.

    Returns: (graphviz.Digraph, dependency_warnings)
    """
    if process_steps_df is None or process_steps_df.empty:
        raise ValueError("process_steps dataframe is empty")

    # Layout constants tuned for readability; adjust if needed
    COMPONENTS_RANKDIR = "TB"  # stack disconnected components vertically
    STEPS_RANKDIR = "LR"       # left-to-right flow inside each component
    NODESEP = 0.12              # inches between nodes in the same rank
    RANKSEP = 0.40              # inches between ranks
    GRAPH_W_IN, GRAPH_H_IN = (4.5, 3.0)
    PACK_MARGIN = 0.20          # inches between components
    FONT_SIZE = 7
    NODE_H, NODE_W = (0.60, 0.90)
    LABEL_TWO_LINES = True
    LABEL_WRAP = 16

    steps = steps_from_dataframe(process_steps_df)
    dep_warnings = validate_dependencies(steps)

    g = Digraph(comment="Process Steps")
    g.attr(
        rankdir=COMPONENTS_RANKDIR,
        bgcolor="transparent",
        pad="0.1",
        margin="0.02",
        nodesep=str(NODESEP),
        ranksep=str(RANKSEP),
        size=f"{GRAPH_W_IN},{GRAPH_H_IN}",
        ratio="compress",
        pack="true",
        packmode="graph",
        packmargin=str(PACK_MARGIN),
    )
    g.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#72BCFF",
        color="white",
        fontsize=str(FONT_SIZE),
        height=str(NODE_H),
        width=str(NODE_W),
        margin="0.02,0.01",
    )
    g.attr("edge", color="white", arrowsize="0.5", penwidth="1.0")

    # Group nodes by product (component)
    from collections import defaultdict

    nodes_by_product = defaultdict(list)
    step_to_pid = {}
    for s in steps:
        label = (
            f"{_wrap_label(s.step_name, LABEL_WRAP)}\n({s.step_id})"
            if LABEL_TWO_LINES
            else f"{_wrap_label(s.step_name, LABEL_WRAP)} ({s.step_id})"
        )
        node_id = f"{s.product_id}__{s.step_id}"
        nodes_by_product[s.product_id].append((node_id, label))
        step_to_pid[s.step_id] = s.product_id

    for pid, items in sorted(nodes_by_product.items()):
        with g.subgraph(name=f"cluster_{pid}") as c:
            c.attr(label="", color="transparent", penwidth="0", rankdir=STEPS_RANKDIR)
            for node_id, label in items:
                c.node(node_id, label)

    # Edges
    for s in steps:
        if s.dependency_step_id:
            dep_pid = step_to_pid.get(s.dependency_step_id, s.product_id)
            src = f"{dep_pid}__{s.dependency_step_id}"
            dst = f"{s.product_id}__{s.step_id}"
            g.edge(src, dst)

    return g, dep_warnings
