from typing import List
from .step import Step
from .validator import validate_dependencies


class Workflow:
    def __init__(self, steps: List[Step]):
        self.steps = steps

    def validate(self) -> List[str]:
        return validate_dependencies(self.steps)

    @property
    def step_ids(self) -> List[str]:
        return [s.step_id for s in self.steps]
