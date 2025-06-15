from typing import Any

from ..execute.orchestrator import PlanOrchestrator
from ..logging_config import get_logger
from ..plan.abstract import AbstractPlanner
from ..plan.concrete import ConcretePlannerBase
from ..schema.events import (
    AbstractToolsGeneratedEvent,
    AgentEventHandler,
    ExecutionCompletedEvent,
    ExecutionStartedEvent,
    NullEventHandler,
    PlanGeneratedEvent,
    ToolMappingFailedEvent,
    ToolMappingGeneratedEvent,
)
from ..schema.handler_registry import HandlerRegistry

logger = get_logger(__name__)


class AceAgent:
    abstract_planner: AbstractPlanner
    concrete_planner: ConcretePlannerBase
    tool_use_history: list[str]
    output_log: list[Any]
    event_registry: HandlerRegistry

    def __init__(
        self,
        abstract_planner: AbstractPlanner,
        concrete_planner: ConcretePlannerBase,
        event_registry: HandlerRegistry | AgentEventHandler | None = None,
    ):
        logger.info("Initializing AceAgent")
        self.abstract_planner = abstract_planner
        self.concrete_planner = concrete_planner
        self.tool_use_history = []
        self.output_log = []

        # Support both HandlerRegistry and single AgentEventHandler for backward compatibility
        if isinstance(event_registry, HandlerRegistry):
            self.event_registry = event_registry
        elif event_registry is not None:
            # Convert single handler to registry
            self.event_registry = HandlerRegistry("compat")
            self.event_registry.register(event_registry)
        else:
            # Create empty registry with null handler
            self.event_registry = HandlerRegistry("empty")
            self.event_registry.register(NullEventHandler())

        logger.debug("AceAgent initialization completed")
        logger.debug(f"Abstract planner: {type(abstract_planner).__name__}")
        logger.debug(f"Concrete planner: {type(concrete_planner).__name__}")
        logger.debug(f"Event registry: {self.event_registry}")

    def run_query(self, query: str) -> Any:
        logger.log_query(query)

        # Phase 1: Generate abstract plan
        logger.info("Phase 1: Generating abstract plan")

        abstract_plan = self.abstract_planner.generate_abstract_plan(query)
        logger.info(f"Abstract plan generated with {len(abstract_plan.abs_tools)} tools")
        logger.debug(f"Abstract tools: {[tool.name for tool in abstract_plan.abs_tools]}")
        logger.debug(f"Abstract script length: {len(abstract_plan.script)} characters")

        # Always emit events for the generated abstract tools and plan
        self.event_registry.on_abstract_tools_generated(AbstractToolsGeneratedEvent(abstract_plan.abs_tools))
        self.event_registry.on_plan_generated(PlanGeneratedEvent(abstract_plan))
        logger.log_plan(abstract_plan.script, abstract_plan.abs_tools)

        # Phase 2: Generate concrete plan (tool mapping)
        logger.info("Phase 2: Generating concrete plan")
        try:
            tool_mapping = self.concrete_planner.implement_plan(abstract_plan)
            if tool_mapping is None:
                # TODO: refactor this. Should not be an error object. Not needed.
                logger.info("No valid tool mapping found for the given plan")
                self.event_registry.on_tool_mapping_failed(
                    ToolMappingFailedEvent("No valid tool mapping found for the given plan", {})
                )

                # TODO: what is the best way to handle this?
                #       in the case that this happens, shouldn't just return with "success" result . . .
                # TODO: create a custom return type to handle run output.
                #       has .result, .status, .tool_use_history, .output_log, etc.
                return None

            logger.info(f"Tool mapping generated with {len(tool_mapping)} concrete tools")
            logger.debug(f"Concrete tools: {list(tool_mapping.keys())}")
        except Exception as e:
            logger.error(f"Failed to generate tool mapping: {e}", exc_info=True)
            raise

        self.event_registry.on_tool_mapping_generated(ToolMappingGeneratedEvent(tool_mapping))
        for tool_name, tool in tool_mapping.items():
            logger.log_tool_match(tool_name, tool.name)

        self.event_registry.on_execution_started(ExecutionStartedEvent(abstract_plan, tool_mapping))

        # Phase 3: Execute the plan
        logger.info("Phase 3: Executing plan")
        try:
            # Pass the event registry directly to the orchestrator instead of creating a callback
            with PlanOrchestrator(abstract_plan, tool_mapping, event_handler=self.event_registry) as orchestrator:
                logger.debug("Plan orchestrator created")
                orchestrator.launch()
                logger.debug("Plan orchestrator launched")
                orchestrator.join()
                logger.debug("Plan orchestrator joined")

                result = orchestrator.result
                tool_use = orchestrator.tool_use_history

                logger.info("Plan execution completed successfully")
                logger.debug(f"Execution result type: {type(result).__name__}")
                logger.debug(f"Tools used: {tool_use}")
        except Exception as e:
            logger.error(f"Plan execution failed: {e}", exc_info=True)
            raise

        self.event_registry.on_execution_completed(ExecutionCompletedEvent(result))

        # Store results
        self.output_log.append(result)
        self.tool_use_history.extend(tool_use)

        logger.info("Query processing completed")
        logger.debug(f"Total output log entries: {len(self.output_log)}")
        logger.debug(f"Total tool use history: {len(self.tool_use_history)}")

        return result
