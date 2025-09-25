import pandas as pd


def compute_avg_cycle_time_mean(production_log: pd.DataFrame) -> float:
    if production_log is None or production_log.empty:
        return 0.0
    df = production_log.copy()
    if not {"start_time", "end_time"}.issubset(df.columns):
        return 0.0
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "complete"]
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    if df.empty:
        return 0.0
    avg_seconds = (df["end_time"] - df["start_time"]).dt.total_seconds().mean()
    return float(avg_seconds / 3600.0)


def compute_avg_cycle_time_median(production_log: pd.DataFrame) -> float:
    if production_log is None or production_log.empty:
        return 0.0
    df = production_log.copy()
    if not {"start_time", "end_time"}.issubset(df.columns):
        return 0.0
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "complete"]
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    if df.empty:
        return 0.0
    med_seconds = (df["end_time"] - df["start_time"]).dt.total_seconds().median()
    return float(med_seconds / 3600.0)
