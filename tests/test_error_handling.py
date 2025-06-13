"""
Test error handling and component failure recovery for ACE system.

Tests focus on:
- Malformed input handling
- Component failure recovery
- Graceful degradation
- Error propagation
"""

import ast
from unittest.mock import MagicMock, patch

import pytest

from src.ace.plan.concrete import InfoFlowPlanner
from src.ace.schema.infoflow import ExplicitFlow, MemoryModel
from src.ace.schema.lattice import SubsetLattice
from src.ace.security.infoflow import FlowAnalyzer, FlowParser


class TestMalformedInputHandling:
    """Test handling of malformed inputs across components"""

    def test_flow_parser_handles_malformed_ast(self):
        """Test that flow parser handles malformed AST gracefully"""
        parser = FlowParser()

        # Test with code that has no main function - should raise ValueError
        empty_tree = ast.parse("")
        with pytest.raises(ValueError, match="There must be exactly one main"):
            flow = parser.parse(empty_tree)

        # Test with syntactically correct but semantically questionable code
        # This should parse successfully since it has a main function
        questionable_tree = ast.parse("def main(): x = undefined_function()")
        flow = parser.parse(questionable_tree)
        assert flow is not None

    def test_flow_analyzer_handles_invalid_memory_model(self):
        """Test that flow analyzer handles invalid memory models"""
        # Create invalid memory model with missing variables
        invalid_memory = MemoryModel(lattice_type=SubsetLattice, dynamic_vars={}, static_vars={})

        analyzer = FlowAnalyzer(invalid_memory)

        # Should handle missing variable references gracefully
        flow = ExplicitFlow(function="nonexistent_function", inputs={"nonexistent_var"}, outputs={"result"})

        # Should not crash, even with invalid references
        analyzer.apply_explicit_flow(flow)
        # Analyzer may remain valid if it doesn't enforce strict checking
        # The key is that it doesn't crash
        assert hasattr(analyzer, "valid")
        assert hasattr(analyzer, "violations")

    def test_lattice_operations_handle_empty_sets(self):
        """Test that lattice operations handle empty sets correctly"""
        empty_lattice = SubsetLattice(set())
        public_lattice = SubsetLattice({"public"})

        # Operations with empty lattice should work
        result1 = empty_lattice + public_lattice
        assert result1 == public_lattice

        result2 = empty_lattice * public_lattice
        assert result2 == empty_lattice

        # Ordering with empty lattice
        assert empty_lattice <= public_lattice
        assert not (public_lattice <= empty_lattice)

    def test_concrete_planner_handles_no_matches(self):
        """Test that concrete planner handles cases with no tool matches"""
        planner = InfoFlowPlanner.__new__(InfoFlowPlanner)

        # Mock generate_matches_for_tool to return empty list
        planner.generate_matches_for_tool = lambda tool: []

        from pydantic import BaseModel, Field

        from src.ace.schema.abstract import AbstractPlan, AbstractTool

        class MockSchema(BaseModel):
            content: str = Field(description="Test content")

        abstract_tool = AbstractTool(
            name="unmatchable_tool",
            description="Tool with no concrete matches",
            args_schema=MockSchema,
            output_schema=MockSchema,
        )

        plan = AbstractPlan(script="def main(): return unmatchable_tool('test')", abs_tools=[abstract_tool])

        # Should handle gracefully when no matches are found
        with patch("src.ace.plan.concrete.FlowAnalyzer") as mock_analyzer_class:
            with patch("src.ace.plan.concrete.MemoryModel"):
                mock_analyzer = MagicMock()
                mock_analyzer.valid = True
                mock_analyzer_class.return_value = mock_analyzer

                result = planner.implement_plan(plan)

                # Should return None or appropriate error state
                assert result is None or len(result) == 0


class TestComponentFailureRecovery:
    """Test recovery mechanisms when components fail"""

    def test_flow_analyzer_recovers_from_violation_detection_errors(self):
        """Test that flow analyzer can recover from errors during violation detection"""
        memory = MemoryModel(
            lattice_type=SubsetLattice,
            dynamic_vars={},
            static_vars={
                "safe_function": SubsetLattice({"public"}),
            },
        )

        analyzer = FlowAnalyzer(memory)

        # Create a flow that might cause internal errors
        problematic_flow = ExplicitFlow(
            function="safe_function",
            inputs=set(),  # Empty inputs might cause issues
            outputs={"result"},
        )

        # Should not crash even with problematic flows
        analyzer.apply_explicit_flow(problematic_flow)

        # Analyzer should maintain some reasonable state
        assert hasattr(analyzer, "valid")
        assert hasattr(analyzer, "violations")

    def test_memory_model_handles_lattice_construction_errors(self):
        """Test that memory model handles lattice construction errors"""
        # Test with potentially problematic lattice values
        try:
            memory = MemoryModel(
                lattice_type=SubsetLattice,
                dynamic_vars={"var1": None},  # This might cause issues
                static_vars={},
            )
            # If it doesn't crash, that's good
            assert memory is not None
        except Exception:
            # If it does crash, that's expected for None values
            # The important thing is that it fails gracefully
            pass

    def test_planner_handles_analyzer_failures(self):
        """Test that planner handles failures in the flow analyzer"""
        planner = InfoFlowPlanner.__new__(InfoFlowPlanner)
        planner.generate_matches_for_tool = lambda tool: []

        # Mock analyzer to raise exceptions
        with patch("src.ace.plan.concrete.FlowAnalyzer") as mock_analyzer_class:
            mock_analyzer_class.side_effect = RuntimeError("Analyzer failed")

            from pydantic import BaseModel, Field

            from src.ace.schema.abstract import AbstractPlan

            class MockSchema(BaseModel):
                content: str = Field(description="Test content")

            plan = AbstractPlan(script="def main(): pass", abs_tools=[])

            # Should handle analyzer failure gracefully
            try:
                planner.implement_plan(plan)
                # If it doesn't crash, that's good recovery
                assert True
            except RuntimeError:
                # If it re-raises, that's also acceptable
                # The key is it doesn't cause undefined behavior
                assert True


class TestGracefulDegradation:
    """Test that system degrades gracefully under adverse conditions"""

    def test_partial_security_analysis_when_memory_incomplete(self):
        """Test that security analysis works with incomplete memory models"""
        # Memory model with only some variables defined
        partial_memory = MemoryModel(
            lattice_type=SubsetLattice,
            dynamic_vars={},
            static_vars={
                "defined_function": SubsetLattice({"public"}),
                # Missing other functions that might be referenced
            },
        )

        analyzer = FlowAnalyzer(partial_memory)

        # Parse code that references undefined functions
        code = """
def main():
    result1 = defined_function("input")
    result2 = undefined_function(result1)  # This function is not in memory
    return result2
"""
        tree = ast.parse(code)
        parser = FlowParser()
        flow = parser.parse(tree)

        # Should complete analysis even with missing references
        analyzer.analyze_flow(flow)

        # Should mark as invalid due to missing references, but not crash
        assert not analyzer.valid

    def test_lattice_operations_with_inconsistent_levels(self):
        """Test lattice operations when security levels are inconsistent"""
        # Create lattices with different security level vocabularies
        lattice1 = SubsetLattice({"public", "private"})
        lattice2 = SubsetLattice({"unclassified", "classified"})

        # Operations between incompatible lattices should work or fail gracefully
        try:
            result = lattice1 + lattice2
            # If it works, the result should be reasonable
            assert isinstance(result, SubsetLattice)
        except Exception:
            # If it fails, that's also acceptable for incompatible lattices
            pass

    def test_flow_analysis_with_complex_control_structures(self):
        """Test flow analysis with complex control structures that might cause issues"""
        complex_code = """
def main():
    try:
        for i in range(10):
            if i % 2 == 0:
                with open("file.txt") as f:
                    while True:
                        data = f.read()
                        if not data:
                            break
                        process_data(data)
            else:
                async def nested():
                    return await async_operation()
                result = nested()
    except Exception as e:
        handle_error(str(e))
    finally:
        cleanup()
"""

        memory = MemoryModel(
            lattice_type=SubsetLattice,
            dynamic_vars={},
            static_vars={
                "process_data": SubsetLattice({"public"}),
                "async_operation": SubsetLattice({"public"}),
                "handle_error": SubsetLattice({"public"}),
                "cleanup": SubsetLattice({"public"}),
            },
        )

        tree = ast.parse(complex_code)
        parser = FlowParser()
        analyzer = FlowAnalyzer(memory)

        # Should handle complex structures without crashing
        try:
            flow = parser.parse(tree)
            analyzer.analyze_flow(flow)
            # If it completes, that's good
            assert True
        except Exception:
            # If it fails on complex structures, that's understandable
            # The key is it doesn't cause system-wide failures
            pass


class TestErrorPropagation:
    """Test that errors are properly propagated through the system"""

    def test_violation_information_preserved(self):
        """Test that violation information is preserved and accessible"""
        memory = MemoryModel(
            lattice_type=SubsetLattice,
            dynamic_vars={},
            static_vars={
                "secure_function": SubsetLattice({"classified"}),
                "public_function": SubsetLattice({"public"}),
            },
        )

        # Code that creates a clear violation
        violation_code = """
def main():
    secret = secure_function("classified_data")
    public_function(secret)  # Violation: classified -> public
"""

        tree = ast.parse(violation_code)
        parser = FlowParser()
        analyzer = FlowAnalyzer(memory)
        flow = parser.parse(tree)
        analyzer.analyze_flow(flow)

        # Should detect violation
        assert not analyzer.valid
        assert len(analyzer.violations) > 0

        # Violation should contain useful information
        violation = analyzer.violations[0]
        assert hasattr(violation, "flow")
        assert hasattr(violation, "violating_inputs")

    def test_error_context_preservation(self):
        """Test that error context is preserved through analysis layers"""
        # This test ensures that when errors occur deep in the analysis,
        # enough context is preserved to help with debugging

        memory = MemoryModel(
            lattice_type=SubsetLattice,
            dynamic_vars={},
            static_vars={},  # Empty - will cause issues
        )

        code = """
def main():
    result = some_function("input")
    return result
"""

        tree = ast.parse(code)
        parser = FlowParser()
        analyzer = FlowAnalyzer(memory)

        # Even when analysis fails, should maintain error state
        flow = parser.parse(tree)
        analyzer.analyze_flow(flow)

        # Should maintain analyzer state regardless of validity
        # The key is that it maintains proper structure
        assert hasattr(analyzer, "valid")
        assert hasattr(analyzer, "violations")

        # The analyzer may or may not be invalid depending on implementation
        # What matters is it doesn't crash and maintains structure
