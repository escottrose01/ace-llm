import ast

from ...schema.abstract import AbstractPlan
from ..analyzer import BaseValidator
from ..exceptions import AnalysisError


class GrammarValidationError(AnalysisError):
    def __init__(self, message: str):
        super().__init__(message)


class GrammarValidator(BaseValidator):
    @property
    def name(self) -> str:
        return "grammar_validator"

    def validate_abstract_plan(self, plan: AbstractPlan) -> None:
        try:
            ast.parse(plan.script)
        except SyntaxError as e:
            raise GrammarValidationError(f"Invalid Python syntax: {e}")

        self._validate_custom_rules(plan)

    def _validate_custom_rules(self, plan: AbstractPlan) -> None:
        pass
