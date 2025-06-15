"""
Event handler registry for composable event handling.

This module provides a registry system that allows multiple event handlers
to be registered and executed for each event, enabling composition of
different behaviors (CLI, logging, metrics, debugging, etc.).
"""

from typing import Any

from ..logging_config import get_logger
from .events import (
    AbstractToolsGeneratedEvent,
    AgentEventHandler,
    ExecutionCompletedEvent,
    ExecutionOutputEvent,
    ExecutionStartedEvent,
    PlanGeneratedEvent,
    ToolMappingFailedEvent,
    ToolMappingGeneratedEvent,
)

logger = get_logger(__name__)


class HandlerRegistry:
    """Registry for managing multiple event handlers with error isolation."""

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self._handlers: list[AgentEventHandler] = []

    def register(self, handler: AgentEventHandler) -> None:
        """Register an event handler."""
        self._handlers.append(handler)
        logger.debug(f"Registered handler {type(handler).__name__} in registry '{self.name}'")

    def unregister(self, handler: AgentEventHandler) -> None:
        """Unregister an event handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)
            logger.debug(f"Unregistered handler {type(handler).__name__} from registry '{self.name}'")

    def clear(self) -> None:
        """Clear all registered handlers."""
        count = len(self._handlers)
        self._handlers.clear()
        logger.debug(f"Cleared {count} handlers from registry '{self.name}'")

    def _safe_notify(self, event_name: str, event: Any, handler_method: str) -> None:
        """Safely notify all handlers of an event with error isolation."""
        if not self._handlers:
            return

        logger.debug(f"Registry '{self.name}': Notifying {len(self._handlers)} handlers of {event_name}")

        for handler in self._handlers:
            try:
                method = getattr(handler, handler_method)
                method(event)
            except Exception as e:
                # Log error but don't stop other handlers
                handler_name = type(handler).__name__
                logger.error(f"Error in handler {handler_name}.{handler_method}: {e}", exc_info=True)

    def on_abstract_tools_generated(self, event: AbstractToolsGeneratedEvent) -> None:
        """Notify all handlers of abstract tools generated event."""
        self._safe_notify("AbstractToolsGenerated", event, "on_abstract_tools_generated")

    def on_plan_generated(self, event: PlanGeneratedEvent) -> None:
        """Notify all handlers of plan generated event."""
        self._safe_notify("PlanGenerated", event, "on_plan_generated")

    def on_tool_mapping_generated(self, event: ToolMappingGeneratedEvent) -> None:
        """Notify all handlers of tool mapping generated event."""
        self._safe_notify("ToolMappingGenerated", event, "on_tool_mapping_generated")

    def on_tool_mapping_failed(self, event: ToolMappingFailedEvent) -> None:
        """Notify all handlers of tool mapping failed event."""
        self._safe_notify("ToolMappingFailed", event, "on_tool_mapping_failed")

    def on_execution_started(self, event: ExecutionStartedEvent) -> None:
        """Notify all handlers of execution started event."""
        self._safe_notify("ExecutionStarted", event, "on_execution_started")

    def on_execution_output(self, event: ExecutionOutputEvent) -> None:
        """Notify all handlers of execution output event."""
        self._safe_notify("ExecutionOutput", event, "on_execution_output")

    def on_execution_completed(self, event: ExecutionCompletedEvent) -> None:
        """Notify all handlers of execution completed event."""
        self._safe_notify("ExecutionCompleted", event, "on_execution_completed")

    def __len__(self) -> int:
        """Return the number of registered handlers."""
        return len(self._handlers)

    def __str__(self) -> str:
        """String representation of the registry."""
        handler_names = [type(h).__name__ for h in self._handlers]
        return f"HandlerRegistry('{self.name}', {len(self._handlers)} handlers: {handler_names})"


# Global registry for decorator-based registration
_global_registry = HandlerRegistry("global")


def register_handler(handler_class: type[AgentEventHandler]) -> type[AgentEventHandler]:
    """
    Decorator to register a handler class in the global registry.

    Usage:
        @register_handler
        class MyHandler:
            def on_plan_generated(self, event): ...
    """
    # Register an instance of the handler
    instance = handler_class()
    _global_registry.register(instance)
    logger.info(f"Auto-registered handler {handler_class.__name__} via decorator")
    return handler_class


def get_global_registry() -> HandlerRegistry:
    """Get the global handler registry."""
    return _global_registry


def create_registry(name: str = "custom", *handlers: AgentEventHandler) -> HandlerRegistry:
    """Create a new registry with the specified handlers."""
    registry = HandlerRegistry(name)
    for handler in handlers:
        registry.register(handler)
    return registry


def create_development_registry(*additional_handlers: AgentEventHandler) -> HandlerRegistry:
    """Create a registry with common development handlers."""
    from ..cli.event_handler import CLIEventHandler
    from ..cli.formatter import CLIFormatter

    registry = HandlerRegistry("development")

    # Always include CLI for development
    cli_handler = CLIEventHandler(CLIFormatter())
    registry.register(cli_handler)

    # Add development-specific handlers
    registry.register(LoggingEventHandler())
    registry.register(DebugEventHandler())

    # Add any additional handlers
    for handler in additional_handlers:
        registry.register(handler)

    return registry


def create_production_registry(*additional_handlers: AgentEventHandler) -> HandlerRegistry:
    """Create a registry with common production handlers."""
    registry = HandlerRegistry("production")

    # Production typically has minimal UI and focuses on metrics/monitoring
    registry.register(MetricsEventHandler())

    # Add any additional handlers
    for handler in additional_handlers:
        registry.register(handler)

    return registry


# Built-in handlers for common use cases
@register_handler
class LoggingEventHandler:
    """Handler that logs all events at appropriate levels."""

    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    def on_abstract_tools_generated(self, event: AbstractToolsGeneratedEvent) -> None:
        self.logger.info(f"Generated {len(event.tools)} abstract tools: {[t.name for t in event.tools]}")

    def on_plan_generated(self, event: PlanGeneratedEvent) -> None:
        self.logger.info(f"Generated plan with {len(event.plan.script)} characters, {len(event.plan.abs_tools)} tools")

    def on_tool_mapping_generated(self, event: ToolMappingGeneratedEvent) -> None:
        mappings = {abs_name: concrete.name for abs_name, concrete in event.mapping.items()}
        self.logger.info(f"Tool mapping successful: {mappings}")

    def on_tool_mapping_failed(self, event: ToolMappingFailedEvent) -> None:
        self.logger.info(f"Tool mapping failed: {event.message}")

    def on_execution_started(self, event: ExecutionStartedEvent) -> None:
        self.logger.info(f"Execution started for plan with {len(event.tools)} tools")

    def on_execution_output(self, event: ExecutionOutputEvent) -> None:
        self.logger.info(f"Execution output: {event.message}")

    def on_execution_completed(self, event: ExecutionCompletedEvent) -> None:
        result_str = str(event.result)[:100] + "..." if len(str(event.result)) > 100 else str(event.result)
        self.logger.info(f"Execution completed with result: {result_str}")


@register_handler
class MetricsEventHandler:
    """Handler that collects metrics for monitoring and analysis."""

    def __init__(self):
        self.metrics = {
            "tools_generated": 0,
            "plans_generated": 0,
            "successful_mappings": 0,
            "failed_mappings": 0,
            "executions_started": 0,
            "executions_completed": 0,
            "output_messages": 0,
            "total_script_length": 0,
        }
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    def on_abstract_tools_generated(self, event: AbstractToolsGeneratedEvent) -> None:
        self.metrics["tools_generated"] += len(event.tools)

    def on_plan_generated(self, event: PlanGeneratedEvent) -> None:
        self.metrics["plans_generated"] += 1
        self.metrics["total_script_length"] += len(event.plan.script)

    def on_tool_mapping_generated(self, event: ToolMappingGeneratedEvent) -> None:
        self.metrics["successful_mappings"] += 1

    def on_tool_mapping_failed(self, event: ToolMappingFailedEvent) -> None:
        self.metrics["failed_mappings"] += 1

    def on_execution_started(self, event: ExecutionStartedEvent) -> None:
        self.metrics["executions_started"] += 1

    def on_execution_output(self, event: ExecutionOutputEvent) -> None:
        self.metrics["output_messages"] += 1

    def on_execution_completed(self, event: ExecutionCompletedEvent) -> None:
        self.metrics["executions_completed"] += 1
        self.logger.debug(f"Current metrics: {self.metrics}")

    def get_metrics(self) -> dict:
        """Get a copy of current metrics."""
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        for key in self.metrics:
            self.metrics[key] = 0


@register_handler
class DebugEventHandler:
    """Handler that provides detailed debug information during development."""

    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.event_count = 0

    def on_abstract_tools_generated(self, event: AbstractToolsGeneratedEvent) -> None:
        self.event_count += 1
        self.logger.debug(f"[{self.event_count}] AbstractToolsGenerated: {len(event.tools)} tools")
        for i, tool in enumerate(event.tools):
            self.logger.debug(f"  Tool {i + 1}: {tool.name} - {tool.description[:50]}...")

    def on_plan_generated(self, event: PlanGeneratedEvent) -> None:
        self.event_count += 1
        lines = event.plan.script.split("\n")
        self.logger.debug(f"[{self.event_count}] PlanGenerated: {len(lines)} lines")
        for i, line in enumerate(lines[:5]):  # Show first 5 lines
            self.logger.debug(f"  {i + 1}: {line}")
        if len(lines) > 5:
            self.logger.debug(f"  ... and {len(lines) - 5} more lines")

    def on_tool_mapping_generated(self, event: ToolMappingGeneratedEvent) -> None:
        self.event_count += 1
        self.logger.debug(f"[{self.event_count}] ToolMappingGenerated: {len(event.mapping)} mappings")
        for abs_name, concrete in event.mapping.items():
            self.logger.debug(f"  {abs_name} -> {concrete.name} ({concrete.provider})")

    def on_tool_mapping_failed(self, event: ToolMappingFailedEvent) -> None:
        self.event_count += 1
        self.logger.debug(f"[{self.event_count}] ToolMappingFailed: {event.message}")
        self.logger.debug(f"  Details: {event.details}")

    def on_execution_started(self, event: ExecutionStartedEvent) -> None:
        self.event_count += 1
        self.logger.debug(f"[{self.event_count}] ExecutionStarted")
        self.logger.debug(f"  Plan tools: {len(event.plan.abs_tools)}")
        self.logger.debug(f"  Mapped tools: {len(event.tools)}")

    def on_execution_output(self, event: ExecutionOutputEvent) -> None:
        self.event_count += 1
        self.logger.debug(f"[{self.event_count}] ExecutionOutput: '{event.message}'")

    def on_execution_completed(self, event: ExecutionCompletedEvent) -> None:
        self.event_count += 1
        result_type = type(event.result).__name__
        self.logger.debug(f"[{self.event_count}] ExecutionCompleted: {result_type}")
        self.logger.debug(f"  Total events processed: {self.event_count}")


class CaptureEventHandler:
    """Handler that captures all events for testing purposes."""

    def __init__(self):
        self.captured_events = []

    def _capture(self, event_name: str, event: Any) -> None:
        self.captured_events.append((event_name, event))

    def on_abstract_tools_generated(self, event: AbstractToolsGeneratedEvent) -> None:
        self._capture("abstract_tools_generated", event)

    def on_plan_generated(self, event: PlanGeneratedEvent) -> None:
        self._capture("plan_generated", event)

    def on_tool_mapping_generated(self, event: ToolMappingGeneratedEvent) -> None:
        self._capture("tool_mapping_generated", event)

    def on_tool_mapping_failed(self, event: ToolMappingFailedEvent) -> None:
        self._capture("tool_mapping_failed", event)

    def on_execution_started(self, event: ExecutionStartedEvent) -> None:
        self._capture("execution_started", event)

    def on_execution_output(self, event: ExecutionOutputEvent) -> None:
        self._capture("execution_output", event)

    def on_execution_completed(self, event: ExecutionCompletedEvent) -> None:
        self._capture("execution_completed", event)

    def get_events_by_type(self, event_type: str) -> list:
        """Get all captured events of a specific type."""
        return [event for name, event in self.captured_events if name == event_type]

    def clear(self) -> None:
        """Clear all captured events."""
        self.captured_events.clear()
