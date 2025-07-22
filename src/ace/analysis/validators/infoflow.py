from typing import Any, Optional

from ...schema.infoflow import MemoryModel
from ..analyzer import BaseValidator
from ..exceptions import AnalysisError


class InfoFlowViolationError(AnalysisError):
    def __init__(self, message: str, flow_violations: Optional[list[Any]] = None):
        super().__init__(message)
        self.flow_violations = flow_violations or []


class InfoFlowValidator(BaseValidator):
    def __init__(self, memory_model: MemoryModel):
        self.memory_model = memory_model

    @property
    def name(self) -> str:
        return "infoflow_validator"
