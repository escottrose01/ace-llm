import logging
from typing import Any

from .execute.orchestrator import PlanOrchestrator
from .plan.abstract import AbstractPlanner
from .plan.concrete import ConcretePlannerBase

logger = logging.getLogger(__name__)


class AceAgent:
    abstract_planner: AbstractPlanner
    concrete_planner: ConcretePlannerBase
    tool_use_history: list[str]
    output_log: list[Any]

    def __init__(
        self,
        abstract_planner: AbstractPlanner,
        concrete_planner: ConcretePlannerBase,
    ):
        self.abstract_planner = abstract_planner
        self.concrete_planner = concrete_planner
        self.tool_use_history = []
        self.output_log = []

    def run_query(self, query: str) -> str:
        logger.info(f"Running query: {query}")

        abstract_plan = self.abstract_planner.generate_abstract_plan(query)
        tool_mapping = self.concrete_planner.implement_plan(abstract_plan)

        # Display the generated abstract tools and plan, and the matching
        print("\n" + "=" * 10, "ABSTRACT TOOLS", "=" * 10)
        for tool in abstract_plan.abs_tools:
            print(f"{tool.name}: {tool.description}")
        print("\n" + "=" * 10, "PLAN", "=" * 10)
        print(abstract_plan.script)
        print("\n" + "=" * 10, "TOOL MAPPING", "=" * 10)
        for tool_name, tool in tool_mapping.items():
            print(f"{tool_name}: {tool.name}")
        print("\n" + "=" * 10, "BEGINNING EXECUTION", "=" * 10)

        with PlanOrchestrator(abstract_plan, tool_mapping) as orchestrator:
            orchestrator.launch()
            orchestrator.join()

            result = orchestrator.result
            tool_use = orchestrator.tool_use_history

        print("\n" + "=" * 10, "EXECUTION COMPLETED", "=" * 10 + "\n")
        logger.debug(f"Plan execution completed with result: {result}")
        logger.debug(f"Tool use history: {tool_use}")

        self.output_log.append(result)
        self.tool_use_history.extend(tool_use)

        return result
