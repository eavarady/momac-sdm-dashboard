from typing import List, Optional
from pydantic import BaseModel, Field


class Step(BaseModel):
    product_id: str
    step_id: str
    step_name: str
    assigned_machine: Optional[str] = None
    assigned_operators: List[str] = Field(default_factory=list)
    estimated_time: Optional[int] = None  # hours
    dependency_step_id: Optional[str] = None
