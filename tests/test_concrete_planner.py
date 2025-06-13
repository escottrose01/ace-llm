from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field

from src.ace.plan.concrete import ConcretePlannerBase, InfoFlowPlanner, SchemaAdaptedTool
from src.ace.schema.abstract import AbstractPlan, AbstractTool
from src.ace.schema.concrete import ConcreteToolBase
from src.ace.schema.permissions import Permission


class MockInputSchema(BaseModel):
    query: str = Field(description="Search query")


class MockOutputSchema(BaseModel):
    result: str = Field(description="Search result")


class MockConcreteTool(ConcreteToolBase):
    def __init__(self, name="mock_tool", clearances=None, permissions=None):
        super().__init__(
            name=name,
            provider="test",
            description="Mock tool for testing",
            clearances=clearances or {"public"},
            permissions=permissions or {Permission.NETWORK},
            args_schema=MockInputSchema,
            output_schema=MockOutputSchema,
        )

    def generate_source(self) -> str:
        return """def main(query):
    return {"result": f"Mock result for {query}"}"""


class MockAbstractTool(AbstractTool):
    def __init__(self, name="abstract_tool"):
        super().__init__(
            name=name,
            description="Abstract tool for testing",
            args_schema=MockInputSchema,
            output_schema=MockOutputSchema,
        )


class MockAbstractPlan(AbstractPlan):
    def __init__(self, tools=None):
        super().__init__(script="def main(): return abstract_tool('test')", abs_tools=tools or [MockAbstractTool()])


class DummyConcretePlanner(ConcretePlannerBase):
    def implement_plan(self, plan):
        return {"foo": "bar"}

    def generate_matches_for_tool(self, abstract_tool):
        return []

    @property
    def results(self):
        return []


def test_implement_plan():
    planner = DummyConcretePlanner()
    plan = MockAbstractPlan()
    mapping = planner.implement_plan(plan)
    assert mapping == {"foo": "bar"}


def test_schema_adapted_tool():
    """Test that SchemaAdaptedTool correctly adapts schemas"""
    concrete_tool = MockConcreteTool()
    abstract_tool = MockAbstractTool()

    adapted_tool = SchemaAdaptedTool(
        name=concrete_tool.name,
        provider=concrete_tool.provider,
        description=concrete_tool.description,
        clearances=concrete_tool.clearances,
        permissions=concrete_tool.permissions,
        args_schema=abstract_tool.args_schema,
        output_schema=abstract_tool.output_schema,
        wrapped_tool=concrete_tool,
        input_mapping_source="def input_mapping(query): return {'query': query}",
        output_mapping_source="def output_mapping(inner_output): return inner_output['result']",
    )

    # Test that it generates valid source code
    source = adapted_tool.generate_source()
    assert "def main(" in source
    assert "input_mapping" in source
    assert "output_mapping" in source
    assert "_tool" in source


def test_generate_all_feasible_matches():
    """Test that concrete planner can generate matches for abstract tools"""
    planner = DummyConcretePlanner()

    # Override generate_matches_for_tool to return something useful
    planner.generate_matches_for_tool = lambda tool: [MockConcreteTool(f"match_{tool.name}")]

    abstract_tools = [MockAbstractTool("tool1"), MockAbstractTool("tool2")]
    matches = planner.generate_all_feasible_matches(abstract_tools)

    assert len(matches) == 2
    assert len(matches[0]) == 1
    assert len(matches[1]) == 1
    assert matches[0][0].name == "match_tool1"
    assert matches[1][0].name == "match_tool2"


@patch("src.ace.plan.concrete.MemoryModel")
@patch("src.ace.plan.concrete.FlowAnalyzer")
def test_info_flow_planner_security_validation(mock_analyzer_class, mock_memory_class):
    """Test that InfoFlowPlanner validates information flow constraints"""
    # Setup mocks
    mock_analyzer = MagicMock()
    mock_analyzer.valid = True
    mock_analyzer_class.return_value = mock_analyzer

    mock_memory = MagicMock()
    mock_memory_class.return_value = mock_memory

    # Create planner with mocked dependencies
    planner = InfoFlowPlanner.__new__(InfoFlowPlanner)  # Create without __init__
    planner.generate_matches_for_tool = lambda tool: [
        MockConcreteTool(clearances={"public"}),
        MockConcreteTool(clearances={"private"}),
    ]

    plan = MockAbstractPlan([MockAbstractTool("secure_tool")])

    # Test successful implementation (analyzer.valid = True)
    result = planner.implement_plan(plan)

    # Should return the first valid mapping
    assert result is not None
    assert "secure_tool" in result
    assert result["secure_tool"].clearances == {"public"}

    # Verify security analysis was performed
    mock_analyzer_class.assert_called()
    mock_analyzer.analyze_ast.assert_called()


@patch("src.ace.plan.concrete.MemoryModel")
@patch("src.ace.plan.concrete.FlowAnalyzer")
def test_info_flow_planner_rejects_insecure_plans(mock_analyzer_class, mock_memory_class):
    """Test that InfoFlowPlanner rejects plans that violate security constraints"""
    # Setup mocks to simulate security violation
    mock_analyzer = MagicMock()
    mock_analyzer.valid = False  # All combinations are insecure
    mock_analyzer_class.return_value = mock_analyzer

    mock_memory = MagicMock()
    mock_memory_class.return_value = mock_memory

    # Create planner
    planner = InfoFlowPlanner.__new__(InfoFlowPlanner)
    planner.generate_matches_for_tool = lambda tool: [MockConcreteTool(clearances={"restricted"})]

    plan = MockAbstractPlan([MockAbstractTool("insecure_tool")])

    # Test that insecure plan is rejected
    result = planner.implement_plan(plan)
    assert result is None  # Should return None when no secure implementation exists
