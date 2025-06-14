"""
CLI event handler that bridges agent events to CLI formatting.

This module provides a CLI-specific implementation of the AgentEventHandler
that translates agent events into appropriate CLI display calls.
"""

from ..schema.events import (
    AbstractToolsGeneratedEvent,
    ExecutionCompletedEvent,
    ExecutionOutputEvent,
    ExecutionStartedEvent,
    PlanGeneratedEvent,
    ToolMappingFailedEvent,
    ToolMappingGeneratedEvent,
)
from .formatter import CLIFormatter


class CLIEventHandler:
    """Event handler that translates agent events to CLI display."""

    def __init__(self, formatter: CLIFormatter):
        self.formatter = formatter

    def on_abstract_tools_generated(self, event: AbstractToolsGeneratedEvent) -> None:
        """Display abstract tools in CLI."""
        self.formatter.print_section_header("ABSTRACT TOOLS", "bright_cyan")
        self.formatter.print_abstract_tools(event.tools)

    def on_plan_generated(self, event: PlanGeneratedEvent) -> None:
        """Display generated plan in CLI."""
        self.formatter.print_section_header("PLAN", "bright_green")
        self.formatter.print_plan(event.plan.script)

    def on_tool_mapping_generated(self, event: ToolMappingGeneratedEvent) -> None:
        """Display tool mapping in CLI."""
        self.formatter.print_section_header("TOOL MAPPING", "bright_yellow")
        self.formatter.print_tool_mapping(event.mapping)

    def on_tool_mapping_failed(self, event: ToolMappingFailedEvent) -> None:
        """Display tool mapping failure in CLI."""
        self.formatter.print_section_header("TOOL MAPPING", "bright_yellow")
        self.formatter.print_tool_mapping(event.attempted_mapping)

    def on_execution_started(self, event: ExecutionStartedEvent) -> None:
        """Display execution start in CLI."""
        self.formatter.print_execution_start()

    def on_execution_output(self, event: ExecutionOutputEvent) -> None:
        """Display execution output in CLI."""
        self.formatter.print_execution_output(event.message)

    def on_execution_completed(self, event: ExecutionCompletedEvent) -> None:
        """Display execution completion in CLI."""
        self.formatter.print_execution_complete(event.result)
