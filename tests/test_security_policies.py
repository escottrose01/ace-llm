import ast
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field

from src.ace.plan.concrete import InfoFlowPlanner
from src.ace.schema.abstract import AbstractPlan, AbstractTool
from src.ace.schema.concrete import ConcreteToolBase
from src.ace.schema.infoflow import ExplicitFlow, MemoryModel
from src.ace.schema.lattice import SubsetLattice
from src.ace.schema.permissions import Permission
from src.ace.security.infoflow import FlowAnalyzer, FlowParser


class MockInputSchema(BaseModel):
    content: str = Field(description="Content to process")


class MockOutputSchema(BaseModel):
    result: str = Field(description="Processing result")


class MockConcreteTool(ConcreteToolBase):
    def __init__(self, name, clearances, permissions=None):
        super().__init__(
            name=name,
            provider="test",
            description=f"Mock tool {name}",
            clearances=clearances,
            permissions=permissions or {Permission.NETWORK},
            args_schema=MockInputSchema,
            output_schema=MockOutputSchema,
        )

    def generate_source(self) -> str:
        return f'def main(content): return {{"result": "processed_{self.name}"}}'


def create_security_memory_model():
    """Create a memory model for security testing"""
    return MemoryModel(
        lattice_type=SubsetLattice,
        dynamic_vars={
            "user_data": SubsetLattice({"personal"}),
            "sensitive_info": SubsetLattice({"financial", "medical"}),
        },
        static_vars={
            "load_financial_data": SubsetLattice({"financial"}),
            "load_medical_data": SubsetLattice({"medical"}),
            "load_public_data": SubsetLattice({"public"}),
            "send_personal_email": SubsetLattice({"personal"}),
            "send_public_message": SubsetLattice({"public"}),
            "store_secure_data": SubsetLattice({"financial", "medical"}),
            "log_activity": SubsetLattice({"public"}),
        },
    )


class TestSecurityPolicyEnforcement:
    """Test suite for security policy enforcement in ACE"""

    def test_information_flow_violation_detection(self):
        """Test that information flow violations are properly detected"""
        # Code that violates security policy: financial data sent to personal email
        code = """
def main():
    financial_data = load_financial_data()
    send_personal_email(content=financial_data)
"""
        tree = ast.parse(code)
        memory = create_security_memory_model()
        analyzer = FlowAnalyzer(memory)
        parser = FlowParser()
        flow = parser.parse(tree)
        analyzer.analyze_flow(flow)

        # Should detect violation
        assert not analyzer.valid
        assert len(analyzer.violations) > 0

        # Check violation details
        violation = analyzer.violations[0]
        assert violation.flow.function == "send_personal_email"
        assert "financial" in str(violation.violating_inputs)

    def test_secure_information_flow_allowed(self):
        """Test that secure information flows are allowed"""
        # Code that follows security policy: public data to public channel
        code = """
def main():
    public_data = load_public_data()
    send_public_message(content=public_data)
"""
        tree = ast.parse(code)
        memory = create_security_memory_model()
        analyzer = FlowAnalyzer(memory)
        parser = FlowParser()
        flow = parser.parse(tree)
        analyzer.analyze_flow(flow)

        # Should be valid
        assert analyzer.valid
        assert len(analyzer.violations) == 0

    def test_same_level_information_flow_allowed(self):
        """Test that information flow at the same security level is allowed"""
        # Code that stores financial data in secure storage (both financial level)
        code = """
def main():
    financial_data = load_financial_data()
    store_secure_data(content=financial_data)
"""
        tree = ast.parse(code)
        memory = create_security_memory_model()
        analyzer = FlowAnalyzer(memory)
        parser = FlowParser()
        flow = parser.parse(tree)
        analyzer.analyze_flow(flow)

        # Should be valid
        assert analyzer.valid
        assert len(analyzer.violations) == 0

    def test_multiple_security_levels_violation(self):
        """Test detection of violations involving multiple security levels"""
        # Code that mixes different security levels inappropriately
        code = """
def main():
    financial_data = load_financial_data()
    medical_data = load_medical_data()
    mixed_data = financial_data + medical_data
    send_personal_email(content=mixed_data)
"""
        tree = ast.parse(code)
        memory = create_security_memory_model()
        analyzer = FlowAnalyzer(memory)
        parser = FlowParser()
        flow = parser.parse(tree)
        analyzer.analyze_flow(flow)

        # Should detect violation due to sensitive data going to personal channel
        assert not analyzer.valid
        assert len(analyzer.violations) > 0

    def test_conditional_flow_security(self):
        """Test security enforcement in conditional flows"""
        code = """
def main():
    if True:
        sensitive_data = load_financial_data()
        send_personal_email(content=sensitive_data)
    else:
        public_data = load_public_data()
        send_public_message(content=public_data)
"""
        tree = ast.parse(code)
        memory = create_security_memory_model()
        analyzer = FlowAnalyzer(memory)
        parser = FlowParser()
        flow = parser.parse(tree)
        analyzer.analyze_flow(flow)

        # Should detect violation in the if branch
        assert not analyzer.valid

    def test_loop_flow_security(self):
        """Test security enforcement in loop flows"""
        code = """
def main():
    for i in range(3):
        sensitive_data = load_financial_data()
        log_activity(content=sensitive_data)  # This should be a violation
"""
        tree = ast.parse(code)
        memory = create_security_memory_model()
        analyzer = FlowAnalyzer(memory)
        parser = FlowParser()
        flow = parser.parse(tree)
        analyzer.analyze_flow(flow)

        # Should detect violation: financial data logged to public channel
        assert not analyzer.valid


class TestPrivilegeBasedPlanSelection:
    """Test privilege-based plan selection and least privilege principle"""

    @patch("src.ace.plan.concrete.MemoryModel")
    @patch("src.ace.plan.concrete.FlowAnalyzer")
    def test_least_privilege_tool_selection(self, mock_analyzer_class, mock_memory_class):
        """Test that the system selects tools with least privilege"""
        # Setup mocks for security analysis
        mock_analyzer = MagicMock()
        mock_analyzer.valid = True
        mock_analyzer_class.return_value = mock_analyzer

        # Create tools with different privilege levels
        low_privilege_tool = MockConcreteTool("low_priv", {"public"}, {Permission.NETWORK})
        high_privilege_tool = MockConcreteTool(
            "high_priv", {"financial", "medical"}, {Permission.NETWORK, Permission.FILESYSTEM}
        )

        # Create a planner that will choose between these tools
        planner = InfoFlowPlanner.__new__(InfoFlowPlanner)
        planner.generate_matches_for_tool = lambda tool: [low_privilege_tool, high_privilege_tool]

        abstract_tool = AbstractTool(
            name="test_tool", description="Test tool", args_schema=MockInputSchema, output_schema=MockOutputSchema
        )
        plan = AbstractPlan(script="def main(): return test_tool('data')", abs_tools=[abstract_tool])

        result = planner.implement_plan(plan)

        # Should select the low privilege tool first (assuming both are valid)
        assert result is not None
        assert result["test_tool"].name == "low_priv"

    def test_tool_clearance_levels(self):
        """Test that tools are properly categorized by clearance levels"""
        public_tool = MockConcreteTool("public_tool", {"public"})
        personal_tool = MockConcreteTool("personal_tool", {"personal"})
        financial_tool = MockConcreteTool("financial_tool", {"financial"})
        medical_tool = MockConcreteTool("medical_tool", {"medical"})

        # Verify clearance assignment
        assert "public" in public_tool.clearances
        assert "personal" in personal_tool.clearances
        assert "financial" in financial_tool.clearances
        assert "medical" in medical_tool.clearances

        # Verify tools have appropriate permissions
        assert Permission.NETWORK in public_tool.permissions


class TestClearanceLevelEnforcement:
    """Test clearance level enforcement for tools and data"""

    def test_clearance_hierarchy_enforcement(self):
        """Test that clearance levels form a proper hierarchy"""
        # Test lattice operations for clearance levels
        public = SubsetLattice({"public"})
        personal = SubsetLattice({"personal"})
        financial = SubsetLattice({"financial"})
        mixed = SubsetLattice({"personal", "financial"})

        # Test join operations (union)
        assert (public + personal) == SubsetLattice({"public", "personal"})
        assert (personal + financial) == mixed

        # Test meet operations (intersection)
        assert (public * personal) == SubsetLattice(set())
        assert (mixed * financial) == financial

        # Test ordering
        assert public <= (public + personal)
        assert financial <= mixed
        assert not (mixed <= financial)

    def test_tool_clearance_validation(self):
        """Test that tool clearances are properly validated"""
        # Tools with different clearance levels
        tools = {
            "public": MockConcreteTool("public_tool", {"public"}),
            "personal": MockConcreteTool("personal_tool", {"personal"}),
            "financial": MockConcreteTool("financial_tool", {"financial"}),
            "multi": MockConcreteTool("multi_tool", {"personal", "financial"}),
        }

        # Verify each tool has the expected clearances
        assert tools["public"].clearances == {"public"}
        assert tools["personal"].clearances == {"personal"}
        assert tools["financial"].clearances == {"financial"}
        assert tools["multi"].clearances == {"personal", "financial"}

    def test_data_flow_clearance_enforcement(self):
        """Test that data flows respect clearance constraints"""
        memory = MemoryModel(
            lattice_type=SubsetLattice,
            dynamic_vars={},
            static_vars={
                "public_source": SubsetLattice({"public"}),
                "personal_source": SubsetLattice({"personal"}),
                "financial_sink": SubsetLattice({"financial"}),
                "public_sink": SubsetLattice({"public"}),
            },
        )

        analyzer = FlowAnalyzer(memory)

        # Test valid flow: personal to financial (upward flow allowed)
        valid_flow = ExplicitFlow(function="financial_sink", inputs={"personal_source"}, outputs={"result"})

        # This should be allowed (personal data can flow to financial level)
        analyzer.apply_explicit_flow(valid_flow)
        # We expect violations only when data flows to insufficient clearance levels

        # Test invalid flow: financial to public (downward flow - should be restricted)
        invalid_flow = ExplicitFlow(function="public_sink", inputs={"financial_source"}, outputs={"result"})

        # Add financial source to memory
        memory.static_vars["financial_source"] = SubsetLattice({"financial"})
        analyzer_invalid = FlowAnalyzer(memory)
        analyzer_invalid.apply_explicit_flow(invalid_flow)

        # Should detect violation when financial data flows to public sink
        assert len(analyzer_invalid.violations) > 0
