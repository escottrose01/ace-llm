import sys
from typing import ClassVar
from unittest.mock import MagicMock, patch

sys.modules["ace.execute.orchestrator"] = MagicMock()

from src.ace.agent import AceAgent


class DummyAbstractPlanner:
    def __init__(self):
        self.last_query = None

    def generate_abstract_plan(self, query):
        self.last_query = query

        class DummyPlan:
            abs_tools: ClassVar = [type("Tool", (), {"name": "dummy", "description": "desc"})()]
            script = "def main(): pass"

        return DummyPlan()


class DummyConcretePlanner:
    def implement_plan(self, plan):
        return {"dummy": type("Tool", (), {"name": "dummy_concrete"})()}


class DummyOrchestrator:
    def __init__(self, plan, tool_mapping):
        self.plan = plan
        self.tool_mapping = tool_mapping
        self.result = "success"
        self.tool_use_history = ["dummy"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def launch(self):
        pass

    def join(self):
        pass


@patch("src.ace.agent.PlanOrchestrator", DummyOrchestrator)
def test_ace_agent_run_query():
    agent = AceAgent(DummyAbstractPlanner(), DummyConcretePlanner())
    result = agent.run_query("test query")
    assert result == "success"
    assert agent.output_log[-1] == "success"
    assert "dummy" in agent.tool_use_history
    assert agent.abstract_planner.last_query == "test query"
