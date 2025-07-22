from ..analyzer import BaseValidator
from ..exceptions import AnalysisError


class TypeValidationError(AnalysisError):
    def __init__(self, message: str):
        super().__init__(message)


class TypeValidator(BaseValidator):
    @property
    def name(self) -> str:
        return "type_validator"
