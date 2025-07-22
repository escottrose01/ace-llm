from abc import ABC, abstractmethod
from typing import Optional

from ..schema.abstract import AbstractPlan, AbstractTool
from ..schema.concrete import ConcreteToolBase
from .exceptions import AnalysisError


class BaseValidator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def validate_abstract_tools(self, tools: list[AbstractTool]) -> None:
        pass

    def validate_abstract_plan(self, plan: AbstractPlan) -> None:
        pass

    def validate_tool_mapping(self, abstract_tool: AbstractTool, concrete_tool: ConcreteToolBase) -> None:
        pass

    def validate_concrete_plan(
        self,
        abstract_plan: AbstractPlan,
        concrete_plan: dict[str, ConcreteToolBase],
    ) -> None:
        pass


class ValidatorRegistry:
    def __init__(self):
        self._validators: list[BaseValidator] = []

    def register(self, validator: BaseValidator) -> None:
        self._validators.append(validator)

    def unregister(self, validator_name: str) -> None:
        self._validators[:] = [v for v in self._validators if v.name != validator_name]

    def get_validators(self) -> list[BaseValidator]:
        return self._validators.copy()


class Analyzer:
    def __init__(self, registry: Optional[ValidatorRegistry] = None):
        self.registry = registry or ValidatorRegistry()

    def register(self, validator: BaseValidator) -> None:
        self.registry.register(validator)

    def unregister(self, validator_name: str) -> None:
        self.registry.unregister(validator_name)

    def analyze_post_abstract_tools(self, tools: list[AbstractTool]) -> None:
        validators = self.registry.get_validators()
        errors = []
        for validator in validators:
            try:
                validator.validate_abstract_tools(tools)
            except AnalysisError as e:
                errors.append(e)
        self._handle_errors(errors)

    def analyze_post_abstract_plan(self, plan: AbstractPlan) -> None:
        validators = self.registry.get_validators()
        errors = []
        for validator in validators:
            try:
                validator.validate_abstract_plan(plan)
            except AnalysisError as e:
                errors.append(e)

        self._handle_errors(errors)

    def analyze_post_tool_mapping(
        self,
        abstract_tool: AbstractTool,
        concrete_tool: ConcreteToolBase,
    ) -> None:
        validators = self.registry.get_validators()
        errors = []
        for validator in validators:
            try:
                validator.validate_tool_mapping(abstract_tool, concrete_tool)
            except AnalysisError as e:
                errors.append(e)

        self._handle_errors(errors)

    def analyze_post_concrete_plan(
        self,
        abstract_plan: AbstractPlan,
        concrete_plan: dict[str, ConcreteToolBase],
    ) -> None:
        validators = self.registry.get_validators()
        errors = []
        for validator in validators:
            try:
                validator.validate_concrete_plan(abstract_plan, concrete_plan)
            except AnalysisError as e:
                errors.append(e)

        self._handle_errors(errors)

    def _handle_errors(self, errors: list[AnalysisError]) -> None:
        if errors:
            if len(errors) == 1:
                raise errors[0]
            else:
                messages = [str(e) for e in errors]
                combined_message = f"Multiple validation errors: {'; '.join(messages)}"
                raise AnalysisError(combined_message)
