from __future__ import annotations
from typing import Optional
import pandas as pd


def compute_time_per_step(
    production_log: pd.DataFrame,
    process_steps: Optional[pd.DataFrame] = None,
    products: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute average duration (hours) per (product_id, step_id).

    Returns a DataFrame with columns:
      - product_id, step_id
      - product_label (falls back to product_id)
      - step_label (falls back to step_id)
      - avg_duration_hours (float)
      - events (int) — number of completed events contributing to the average
    """
    if production_log is None or production_log.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "step_id",
                "product_label",
                "step_label",
                "avg_duration_hours",
                "events",
            ]
        )

    df = production_log.copy()
    needed = {"start_time", "end_time", "status", "product_id", "step_id"}
    if not needed.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "product_id",
                "step_id",
                "product_label",
                "step_label",
                "avg_duration_hours",
                "events",
            ]
        )

    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["end_time"] = pd.to_datetime(df.get("end_time"), utc=True, errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "step_id",
                "product_label",
                "step_label",
                "avg_duration_hours",
                "events",
            ]
        )

    # Only completed events, end >= start
    df = df[df["status"].astype(str).str.lower() == "complete"].copy()
    df = df[df["end_time"] >= df["start_time"]]
    if df.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "step_id",
                "product_label",
                "step_label",
                "avg_duration_hours",
                "events",
            ]
        )

    df["duration_hours"] = (
        df["end_time"] - df["start_time"]
    ).dt.total_seconds() / 3600.0

    agg = (
        df.groupby(["product_id", "step_id"], dropna=False)
        .agg(
            avg_duration_hours=("duration_hours", "mean"),
            events=("duration_hours", "count"),
        )
        .reset_index()
    )

    # Optional enrichments
    if (
        process_steps is not None
        and not process_steps.empty
        and {"step_id", "step_name"}.issubset(process_steps.columns)
    ):
        step_names_map = process_steps[["step_id", "step_name"]].drop_duplicates()
        agg = agg.merge(step_names_map, on="step_id", how="left")
    else:
        agg["step_name"] = pd.NA

    if (
        products is not None
        and not products.empty
        and {"product_id", "name"}.issubset(products.columns)
    ):
        prod_names_map = products[["product_id", "name"]].rename(
            columns={"name": "product_label"}
        )
        agg = agg.merge(prod_names_map, on="product_id", how="left")
    else:
        agg["product_label"] = pd.NA

    agg["product_label"] = agg["product_label"].fillna(agg["product_id"].astype(str))
    agg["step_label"] = agg["step_name"].fillna(agg["step_id"].astype(str))
    agg["avg_duration_hours"] = agg["avg_duration_hours"].astype(float)

    return agg[
        [
            "product_id",
            "step_id",
            "product_label",
            "step_label",
            "avg_duration_hours",
            "events",
        ]
    ]
