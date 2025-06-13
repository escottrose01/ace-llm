from typing import Any

from ..cli.formatter import CLIFormatter
from ..execute.orchestrator import PlanOrchestrator
from ..logging_config import get_logger
from ..plan.abstract import AbstractPlanner
from ..plan.concrete import ConcretePlannerBase

logger = get_logger(__name__)


class AceAgent:
    abstract_planner: AbstractPlanner
    concrete_planner: ConcretePlannerBase
    tool_use_history: list[str]
    output_log: list[Any]
    formatter: CLIFormatter

    def __init__(
        self,
        abstract_planner: AbstractPlanner,
        concrete_planner: ConcretePlannerBase,
    ):
        logger.info("Initializing AceAgent")
        self.abstract_planner = abstract_planner
        self.concrete_planner = concrete_planner
        self.tool_use_history = []
        self.output_log = []
        self.formatter = CLIFormatter()
        logger.debug("AceAgent initialization completed")
        logger.debug(f"Abstract planner: {type(abstract_planner).__name__}")
        logger.debug(f"Concrete planner: {type(concrete_planner).__name__}")

    def run_query(self, query: str) -> Any:
        logger.log_query(query)

        # Phase 1: Generate abstract plan
        logger.info("Phase 1: Generating abstract plan")
        try:
            abstract_plan = self.abstract_planner.generate_abstract_plan(query)
            logger.info(f"Abstract plan generated with {len(abstract_plan.abs_tools)} tools")
            logger.debug(f"Abstract tools: {[tool.name for tool in abstract_plan.abs_tools]}")
            logger.debug(f"Abstract script length: {len(abstract_plan.script)} characters")
        except Exception as e:
            logger.error(f"Failed to generate abstract plan: {e}", exc_info=True)
            raise

        # Always display the generated abstract tools and plan first
        self.formatter.print_section_header("ABSTRACT TOOLS", "bright_cyan")
        self.formatter.print_abstract_tools(abstract_plan.abs_tools)

        self.formatter.print_section_header("PLAN", "bright_green")
        self.formatter.print_plan(abstract_plan.script)
        logger.log_plan(abstract_plan.script, abstract_plan.abs_tools)

        # Phase 2: Generate concrete plan (tool mapping)
        logger.info("Phase 2: Generating concrete plan")
        try:
            tool_mapping = self.concrete_planner.implement_plan(abstract_plan)
            if tool_mapping is None:
                logger.error("No valid tool mapping found for the given plan")
                # Show what we tried to map before failing
                self.formatter.print_section_header("TOOL MAPPING", "bright_yellow")
                self.formatter.print_tool_mapping({})
                raise RuntimeError("No valid tool mapping found for the given plan")

            logger.info(f"Tool mapping generated with {len(tool_mapping)} concrete tools")
            logger.debug(f"Concrete tools: {list(tool_mapping.keys())}")
        except Exception as e:
            if "No valid tool mapping found" not in str(e):
                # Show empty mapping for other errors too
                self.formatter.print_section_header("TOOL MAPPING", "bright_yellow")
                self.formatter.print_tool_mapping({})
            logger.error(f"Failed to generate tool mapping: {e}", exc_info=True)
            raise

        self.formatter.print_section_header("TOOL MAPPING", "bright_yellow")
        self.formatter.print_tool_mapping(tool_mapping)
        for tool_name, tool in tool_mapping.items():
            logger.log_tool_match(tool_name, tool.name)

        self.formatter.print_execution_start()

        # Phase 3: Execute the plan
        logger.info("Phase 3: Executing plan")
        try:
            with PlanOrchestrator(abstract_plan, tool_mapping) as orchestrator:
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

        self.formatter.print_execution_complete(result)

        # Store results
        self.output_log.append(result)
        self.tool_use_history.extend(tool_use)

        logger.info("Query processing completed")
        logger.debug(f"Total output log entries: {len(self.output_log)}")
        logger.debug(f"Total tool use history: {len(self.tool_use_history)}")

        return result
