"""
Event system for decoupling agent execution from frontend display.

This module provides event classes and callback interfaces that allow
the agent to emit events during query processing without being coupled
to any specific frontend (CLI, web, etc.).
"""

from dataclasses import dataclass
from typing import Any, Protocol

from ..schema.abstract import AbstractPlan, AbstractTool
from ..schema.concrete import ConcreteToolBase


@dataclass
class AbstractToolsGeneratedEvent:
    """Event emitted when abstract tools are generated."""

    tools: list[AbstractTool]


@dataclass
class PlanGeneratedEvent:
    """Event emitted when an abstract plan is generated."""

    plan: AbstractPlan


@dataclass
class ToolMappingGeneratedEvent:
    """Event emitted when tool mapping is completed."""

    mapping: dict[str, ConcreteToolBase]


@dataclass
class ToolMappingFailedEvent:
    """Event emitted when tool mapping fails."""

    error: Exception
    attempted_mapping: dict[str, ConcreteToolBase]


@dataclass
class ExecutionStartedEvent:
    """Event emitted when plan execution begins."""

    plan: AbstractPlan
    tools: dict[str, ConcreteToolBase]


@dataclass
class ExecutionOutputEvent:
    """Event emitted when the executor produces output."""

    message: str


@dataclass
class ExecutionCompletedEvent:
    """Event emitted when plan execution completes."""

    result: Any


class AgentEventHandler(Protocol):
    """Protocol for handling agent events."""

    def on_abstract_tools_generated(self, event: AbstractToolsGeneratedEvent) -> None:
        """Handle abstract tools generated event."""
        ...

    def on_plan_generated(self, event: PlanGeneratedEvent) -> None:
        """Handle plan generated event."""
        ...

    def on_tool_mapping_generated(self, event: ToolMappingGeneratedEvent) -> None:
        """Handle tool mapping generated event."""
        ...

    def on_tool_mapping_failed(self, event: ToolMappingFailedEvent) -> None:
        """Handle tool mapping failed event."""
        ...

    def on_execution_started(self, event: ExecutionStartedEvent) -> None:
        """Handle execution started event."""
        ...

    def on_execution_output(self, event: ExecutionOutputEvent) -> None:
        """Handle execution output event."""
        ...

    def on_execution_completed(self, event: ExecutionCompletedEvent) -> None:
        """Handle execution completed event."""
        ...


class NullEventHandler:
    """Event handler that does nothing - useful for headless operation."""

    def on_abstract_tools_generated(self, event: AbstractToolsGeneratedEvent) -> None:
        pass

    def on_plan_generated(self, event: PlanGeneratedEvent) -> None:
        pass

    def on_tool_mapping_generated(self, event: ToolMappingGeneratedEvent) -> None:
        pass

    def on_tool_mapping_failed(self, event: ToolMappingFailedEvent) -> None:
        pass

    def on_execution_started(self, event: ExecutionStartedEvent) -> None:
        pass

    def on_execution_output(self, event: ExecutionOutputEvent) -> None:
        pass

    def on_execution_completed(self, event: ExecutionCompletedEvent) -> None:
        pass
