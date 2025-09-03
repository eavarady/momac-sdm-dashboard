from __future__ import annotations
from typing import List, Literal, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, field_validator
from dateutil import parser as dateparser


def _parse_utc(ts: str | datetime) -> datetime:
    if isinstance(ts, datetime):
        dt = ts
    else:
        dt = dateparser.isoparse(str(ts))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


class _Row(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


# ---- Dimension tables -------------------------------------------------------

class MachineRow(_Row):
    machine_id: str = Field(min_length=1)
    line_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    status: Literal["online", "offline", "maintenance"]

    @field_validator("machine_id", "line_id", "type", mode="before")
    @classmethod
    def _strip(cls, v): return str(v).strip()

    @field_validator("status", mode="before")
    @classmethod
    def _status_norm(cls, v): return str(v).strip().lower()


class ProductionLineRow(_Row):
    line_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    shift: str = Field(min_length=1)

    @field_validator("line_id", "name", "shift", mode="before")
    @classmethod
    def _strip(cls, v): return str(v).strip()


class ProductRow(_Row):
    product_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(default="")
    spec_version: str = Field(default="")

    @field_validator("product_id", "name", "category", "spec_version", mode="before")
    @classmethod
    def _strip(cls, v): return str(v).strip()


class OperatorRow(_Row):
    operator_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role: str = Field(default="")

    @field_validator("operator_id", "name", "role", mode="before")
    @classmethod
    def _strip(cls, v): return str(v).strip()


# ---- Process steps ----------------------------------------------------------

class ProcessStepRow(_Row):
    product_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    step_name: str = Field(default="")
    assigned_machine: str = Field(default="")
    assigned_operators: List[str] = Field(default_factory=list)
    estimated_time: int = Field(ge=0, default=0)
    dependency_step_id: Optional[str] = None

    @field_validator("product_id", "step_id", "step_name", "assigned_machine", mode="before")
    @classmethod
    def _strip(cls, v):
        s = "" if v is None else str(v)
        return s.strip()

    @field_validator("assigned_operators", mode="before")
    @classmethod
    def _ops_parse(cls, v):
        if v is None: return []
        if isinstance(v, list): items = v
        else: items = str(v).split(",")
        out: list[str] = []
        for it in items:
            it = str(it).strip()
            if it and it not in out:
                out.append(it)
        return out

    @field_validator("estimated_time", mode="before")
    @classmethod
    def _et_int(cls, v):
        if v is None: return 0
        try:
            iv = int(float(v))
            return max(0, iv)
        except Exception:
            return 0

    @field_validator("dependency_step_id", mode="before")
    @classmethod
    def _dep_norm(cls, v):
        if v is None: return None
        s = str(v).strip()
        return s or None

    @field_validator("step_name")
    @classmethod
    def _fallback_name(cls, v, info):
        if v: return v
        step_id = info.data.get("step_id", "")
        return step_id


# ---- Fact tables ------------------------------------------------------------

class ProductionLogRow(_Row):
    timestamp: datetime
    line_id: str
    product_id: str
    step_id: str
    quantity: int = Field(ge=0)
    status: Literal["in_progress", "complete"]

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ts_parse(cls, v): return _parse_utc(v)

    @field_validator("line_id", "product_id", "step_id", mode="before")
    @classmethod
    def _strip(cls, v): return str(v).strip()

    @field_validator("status", mode="before")
    @classmethod
    def _status_norm(cls, v): return str(v).strip().lower()

    @field_validator("quantity", mode="before")
    @classmethod
    def _qty_int(cls, v):
        if v is None: return 0
        try:
            return max(0, int(float(v)))
        except Exception:
            return 0


class MachineMetricRow(_Row):
    timestamp: datetime
    machine_id: str
    metric_type: str
    metric_value: float

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ts_parse(cls, v): return _parse_utc(v)

    @field_validator("machine_id", "metric_type", mode="before")
    @classmethod
    def _strip(cls, v): return str(v).strip()


class QualityCheckRow(_Row):
    timestamp: datetime
    product_id: str
    check_type: str
    result: Literal["pass", "fail", "rework"]
    inspector_id: Optional[str] = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ts_parse(cls, v): return _parse_utc(v)

    @field_validator("product_id", "check_type", "result", "inspector_id", mode="before")
    @classmethod
    def _strip(cls, v):
        if v is None: return None
        return str(v).strip() or None