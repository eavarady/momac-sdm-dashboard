from __future__ import annotations
from typing import Optional, Literal, Mapping
import pandas as pd
import plotly.express as px
from workflow.dag import planned_finish_offsets  # NEW


class GanttChart:
    """Build Plotly Gantt charts from normalized tables.

    Methods return Plotly Figure objects (or None if insufficient data).
    Keep this module UI-agnostic; Streamlit rendering happens in the dashboard.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _now_utc() -> pd.Timestamp:
        """Return a UTC-aware Timestamp robustly across pandas versions.

        If utcnow() returns a naive timestamp, localize to UTC.
        If it returns tz-aware, convert to UTC (no-op if already UTC).
        """
        ts = pd.Timestamp.utcnow()
        if ts.tz is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def actual_gantt(
        self,
        production_log: pd.DataFrame,
        product_names: Optional[pd.DataFrame | Mapping[str, str]] = None,
        step_names: Optional[pd.DataFrame | Mapping[str, str]] = None,
        view: Optional[Literal["by_run", "by_step"]] = None,
    ) -> Optional["px.Figure"]:
        """Actual Gantt using start_time/end_time from production_log.

        Requirements: columns [product_id, step_id, status, start_time], optional end_time.
        In-progress rows with no end_time get a temporary end at 'now' for visualization only.
        """
        if production_log is None or production_log.empty:
            return None
        required = {"product_id", "step_id", "status", "start_time"}
        if not required.issubset(production_log.columns):
            return None

        df = production_log.copy()
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
        if "end_time" in df.columns:
            df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
        else:
            df["end_time"] = pd.NaT

        # Initialize default label first
        df["product_label"] = df["product_id"].astype(str)

        # Map product_id -> display name (optional)
        if product_names is not None:
            if isinstance(product_names, pd.DataFrame) and {
                "product_id",
                "name",
            }.issubset(product_names.columns):
                df = df.merge(
                    product_names[["product_id", "name"]],
                    on="product_id",
                    how="left",
                )
                # Overwrite label where a name is available
                df["product_label"] = df["name"].fillna(df["product_label"])
                df = df.drop(columns=["name"])
            elif isinstance(product_names, Mapping):
                df["product_label"] = (
                    df["product_id"]
                    .map(lambda x: product_names.get(x))
                    .fillna(df["product_label"])
                )

        # Map step_id -> step_name (optional)
        if step_names is not None:
            if isinstance(step_names, pd.DataFrame) and {
                "step_id",
                "step_name",
            }.issubset(step_names.columns):
                df = df.merge(
                    step_names[["step_id", "step_name"]], on="step_id", how="left"
                )
            elif isinstance(step_names, Mapping):
                df["step_name"] = df["step_id"].map(lambda x: step_names.get(x))

        # For in-progress rows, fill end_time with now (for visualization only)
        now_utc = self._now_utc()
        mask_inp = (df["status"].astype(str).str.lower() == "in_progress") & (
            df["end_time"].isna()
        )
        df.loc[mask_inp, "end_time"] = now_utc

        # Keep rows with valid intervals
        df = df.dropna(subset=["start_time", "end_time"]).reset_index(drop=True)
        if df.empty:
            return None

        # Determine desired view
        desired = view
        if desired is None:
            desired = "by_run" if "run_id" in df.columns else "by_step"

        # Group; include run_id when available
        if desired == "by_run" and "run_id" in df.columns:
            agg = (
                df.groupby(
                    ["product_id", "product_label", "run_id", "step_id", "status"],
                    dropna=False,
                )
                .agg(start_time=("start_time", "min"), end_time=("end_time", "max"))
                .reset_index()
            )
            # Prefer step_name when available
            if "step_name" not in agg.columns and "step_name" in df.columns:
                # bring through a representative step_name per (product, run, step)
                names = (
                    df.dropna(subset=["step_id"])  # avoid NaN keys
                    .groupby(["product_id", "run_id", "step_id"], dropna=False)[
                        "step_name"
                    ]
                    .first()
                    .reset_index()
                )
                agg = agg.merge(
                    names, on=["product_id", "run_id", "step_id"], how="left"
                )

            # Per-run Y-axis label: one row per (product, run)
            run_label = "(" + agg["run_id"].astype(str) + ")"
            agg["y_label"] = agg["product_label"].astype(str) + " . " + run_label

            # Color by step (name preferred)
            color_col = "step_name" if "step_name" in agg.columns else "step_id"

            hover_cols = [
                c
                for c in [
                    "product_id",
                    "product_label",
                    "run_id",
                    "step_id",
                    "step_name",
                    "status",
                ]
                if c in agg.columns
            ]

            fig = px.timeline(
                agg,
                x_start="start_time",
                x_end="end_time",
                y="y_label",
                color=color_col,
                hover_data=hover_cols,
                title="Actual Gantt (by run)",
            )
            fig.update_yaxes(autorange="reversed")
            return fig
        else:
            agg = (
                df.groupby(
                    ["product_id", "product_label", "step_id", "status"], dropna=False
                )
                .agg(start_time=("start_time", "min"), end_time=("end_time", "max"))
                .reset_index()
            )
            # Prefer step_name when available (if step_names mapping was provided via df)
            if "step_name" not in agg.columns and "step_name" in df.columns:
                names = (
                    df.dropna(subset=["step_id"])  # avoid NaN keys
                    .groupby(["product_id", "step_id"], dropna=False)["step_name"]
                    .first()
                    .reset_index()
                )
                agg = agg.merge(names, on=["product_id", "step_id"], how="left")

            step_label = agg.get("step_name", agg.get("step_id")).astype(str)
            agg["task"] = agg["product_label"].astype(str) + " . " + step_label
            hover_cols = [
                c
                for c in [
                    "product_id",
                    "product_label",
                    "step_id",
                    "step_name",
                    "status",
                ]
                if c in agg.columns
            ]

        fig = px.timeline(
            agg,
            x_start="start_time",
            x_end="end_time",
            y="task",
            color="status",
            hover_data=hover_cols,
            title="Actual Gantt (by step)",
        )
        fig.update_yaxes(autorange="reversed")
        return fig

    def planned_gantt(
        self,
        process_steps: pd.DataFrame,
        production_log: Optional[pd.DataFrame] = None,
        anchor: Literal["now", "run_start"] = "now",
        product_names: Optional[pd.DataFrame | Mapping[str, str]] = None,
        view: Optional[Literal["by_run", "by_step"]] = None,
    ) -> Optional["px.Figure"]:
        """Planned Gantt using dependency DAG and estimated_time (hours).

        anchor:
          - "now": anchor all products at current UTC time (preview).
          - "run_start": anchor each (product_id, run_id) at earliest actual start_time.
        """
        if process_steps is None or process_steps.empty:
            return None
        req = {"product_id", "step_id", "estimated_time"}
        if not req.issubset(process_steps.columns):
            return None

        df = process_steps.copy()
        df["estimated_time"] = pd.to_numeric(df["estimated_time"], errors="coerce")
        df["planned_dur_sec"] = df["estimated_time"].fillna(0.0).astype(float) * 3600.0

        # Map product_id -> display name for planned data
        if product_names is not None:
            if isinstance(product_names, pd.DataFrame) and {
                "product_id",
                "name",
            }.issubset(product_names.columns):
                df = df.merge(
                    product_names[["product_id", "name"]],
                    on="product_id",
                    how="left",
                )
                df = df.rename(columns={"name": "product_label"})
            elif isinstance(product_names, Mapping):
                df["product_label"] = df["product_id"].map(
                    lambda x: product_names.get(x)
                )
        df["product_label"] = df.get("product_label", pd.Series(dtype=object)).fillna(
            df["product_id"].astype(str)
        )

        offsets = planned_finish_offsets(df)
        if offsets is None or offsets.empty:
            seq = (
                df.sort_values(["product_id", "step_id"])
                .groupby("product_id", as_index=False, group_keys=False)
                .apply(
                    lambda g: g.assign(
                        planned_finish_offset_sec=g["planned_dur_sec"].cumsum()
                    )
                )
            )
            merged = seq
        else:
            merged = df.merge(offsets, on=["product_id", "step_id"], how="left")

        merged["planned_start_offset_sec"] = (
            merged["planned_finish_offset_sec"].fillna(0.0)
            - merged["planned_dur_sec"].fillna(0.0)
        ).clip(lower=0)

        # Anchor by run start (if requested and data available)
        if (
            anchor == "run_start"
            and production_log is not None
            and not production_log.empty
            and {"product_id", "run_id", "start_time"}.issubset(production_log.columns)
        ):
            plog = production_log.copy()
            plog["start_time"] = pd.to_datetime(
                plog["start_time"], utc=True, errors="coerce"
            )
            anchors = (
                plog.dropna(subset=["start_time"])
                .groupby(["product_id", "run_id"], as_index=False)["start_time"]
                .min()
                .rename(columns={"start_time": "anchor_start"})
            )
            planned = anchors.merge(
                merged[
                    [
                        "product_id",
                        "product_label",
                        "step_id",
                        "step_name",
                        "planned_start_offset_sec",
                        "planned_finish_offset_sec",
                    ]
                ],
                on="product_id",
                how="left",
            )
            planned["start_time"] = planned["anchor_start"] + pd.to_timedelta(
                planned["planned_start_offset_sec"], unit="s"
            )
            planned["end_time"] = planned["anchor_start"] + pd.to_timedelta(
                planned["planned_finish_offset_sec"].fillna(0.0), unit="s"
            )
            step_label = planned.get("step_name", planned.get("step_id")).astype(str)
            run_label = "(" + planned["run_id"].astype(str) + ")"
            # Build step-level task label (for by-step view)
            planned["task"] = (
                planned["product_label"].astype(str)
                + " . "
                + run_label
                + " . "
                + step_label
            )
            # Default color for by-step: run_id; for by-run we will color by step
            color_col = "run_id"
            hover_cols = [
                c
                for c in [
                    "product_id",
                    "product_label",
                    "run_id",
                    "step_id",
                    "step_name",
                ]
                if c in planned.columns
            ]
            # Decide Y-axis based on view
            desired = view or "by_step"
            if desired == "by_run":
                planned["y_label"] = (
                    planned["product_label"].astype(str) + " . " + run_label
                )
                y_col = "y_label"
                color_col = "step_name" if "step_name" in planned.columns else "step_id"
                title = "Planned Gantt (by run)"
            else:
                y_col = "task"
                title = "Planned Gantt (by step)"
        else:
            base = self._now_utc()
            merged["start_time"] = base + pd.to_timedelta(
                merged["planned_start_offset_sec"], unit="s"
            )
            merged["end_time"] = base + pd.to_timedelta(
                merged["planned_finish_offset_sec"].fillna(0.0), unit="s"
            )
            step_label = merged.get("step_name", merged.get("step_id")).astype(str)
            merged["task"] = merged["product_label"].astype(str) + " . " + step_label
            planned = merged
            color_col = "product_id"
            hover_cols = [
                c
                for c in [
                    "product_id",
                    "product_label",
                    "step_id",
                    "step_name",
                    "assigned_machine",
                    "estimated_time",
                ]
                if c in planned.columns
            ]
            y_col = "task"
            title = "Planned Gantt (DAG-derived)"

        planned = planned.dropna(subset=["start_time", "end_time"])
        if planned.empty:
            return None

        fig = px.timeline(
            planned,
            x_start="start_time",
            x_end="end_time",
            y=y_col,
            color=color_col,
            hover_data=hover_cols,
            title=title,
        )
        fig.update_yaxes(autorange="reversed")
        return fig
