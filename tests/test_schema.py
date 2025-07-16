# Comprehensive schema import test
import pytest
from pydantic import ValidationError

try:
    from pydantic import Field, create_model

    from src.ace.schema import abstract, concrete, infoflow, lattice, permissions
    from src.ace.schema.abstract import AbstractPlan, AbstractTool
except ImportError:
    pytest.skip("src.ace.schema modules not available", allow_module_level=True)


def test_schema_imports():
    assert abstract is not None
    assert concrete is not None
    assert infoflow is not None
    assert lattice is not None
    assert permissions is not None


class TestAbstractPlanValidation:
    """Test the validation logic for AbstractPlan."""

    def create_dummy_tool(self) -> AbstractTool:
        """Create a dummy AbstractTool for testing."""
        args_schema = create_model("TestArgs", text=(str, Field(..., description="Test input")))
        output_schema = create_model("TestOutput", result=(str, Field(..., description="Test output")))

        return AbstractTool(
            name="TestTool", description="A test tool", args_schema=args_schema, output_schema=output_schema
        )

    def test_valid_plan(self):
        """Test that a valid plan passes validation."""
        valid_script = """def main():
    result: str = TestTool("test input")
    display(result)
    return result
final_output = main()"""

        plan = AbstractPlan(script=valid_script, abs_tools=[self.create_dummy_tool()])
        assert plan.script == valid_script

    def test_missing_type_annotation(self):
        """Test that variables without type annotations are rejected."""
        invalid_script = """def main():
    result = "test"
    return result
final_output = main()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "must be declared with a type annotation" in str(exc_info.value)

    def test_duplicate_variable_declaration(self):
        """Test that variables can only be declared once."""
        invalid_script = """def main():
    result: str = "first"
    result: str = "second"
    return result
final_output = main()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "already declared" in str(exc_info.value)

    def test_declaration_in_if_block(self):
        """Test that variable declarations in if blocks are rejected."""
        invalid_script = """def main():
    condition: bool = True
    if condition:
        result: str = "test"
    return result
final_output = main()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "cannot be declared inside if/else/loop blocks" in str(exc_info.value)

    def test_invalid_type(self):
        """Test that invalid types are rejected."""
        invalid_script = """def main():
    result: list = []
    return result
final_output = main()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "not allowed" in str(exc_info.value)

    def test_tool_type_allowed(self):
        """Test that abstract tool names are allowed as types."""
        valid_script = """def main():
    tool_result: TestTool = TestTool("input")
    return tool_result
final_output = main()"""

        plan = AbstractPlan(script=valid_script, abs_tools=[self.create_dummy_tool()])
        assert plan.script == valid_script

    def test_invalid_syntax(self):
        """Test that syntactically invalid Python is rejected."""
        invalid_script = """def main(:
    return "test"
final_output = main()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "invalid Python syntax" in str(exc_info.value)

    def test_multiple_functions(self):
        """Test that multiple function definitions are rejected."""
        invalid_script = """def main():
    return "test"

def helper():
    return "helper"

final_output = main()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "single main function definition" in str(exc_info.value)

    def test_wrong_function_name(self):
        """Test that function must be named 'main'."""
        invalid_script = """def execute():
    return "test"
final_output = execute()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "must be named 'main'" in str(exc_info.value)

    def test_function_with_arguments(self):
        """Test that main() function cannot take arguments."""
        invalid_script = """def main(arg1, arg2):
    return "test"
final_output = main()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "must take no arguments" in str(exc_info.value)

    def test_missing_final_output(self):
        """Test that missing final_output assignment is rejected."""
        invalid_script = '''def main():
    return "test"'''

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "final_output assignment" in str(exc_info.value)

    def test_wrong_final_output_variable(self):
        """Test that assignment must be to 'final_output'."""
        invalid_script = """def main():
    return "test"
result = main()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "assignment to 'final_output'" in str(exc_info.value)

    def test_final_output_not_main_call(self):
        """Test that final_output must be assigned from main() call."""
        invalid_script = '''def main():
    return "test"
final_output = "test"'''

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "calling main()" in str(exc_info.value)

    def test_main_call_with_arguments(self):
        """Test that main() call cannot have arguments."""
        invalid_script = """def main():
    return "test"
final_output = main("arg")"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "main() call must have no arguments" in str(exc_info.value)

    def test_complex_valid_plan(self):
        """Test a more complex but valid plan."""
        valid_script = """def main():
    input_text: str = "Hello world"
    result: str = TestTool(input_text)
    result_upper: str = result.upper()
    final_result: str = "Result: " + result_upper
    display(final_result)
    return final_result
final_output = main()"""

        plan = AbstractPlan(script=valid_script, abs_tools=[self.create_dummy_tool()])
        assert plan.script == valid_script

    def test_blacklisted_builtin(self):
        """Test that blacklisted builtins are rejected."""
        invalid_script = """def main():
    result: str = open("file.txt").read()
    return result
final_output = main()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "not allowed" in str(exc_info.value)

    def test_augmented_assignment(self):
        """Test that augmented assignments are rejected."""
        invalid_script = """def main():
    result: str = "test"
    result += " more"
    return result
final_output = main()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "Augmented assignments not allowed" in str(exc_info.value)

    def test_nested_function(self):
        """Test that nested functions are rejected."""
        invalid_script = """def main():
    def helper():
        return "help"
    return helper()
final_output = main()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "Nested function definitions not allowed" in str(exc_info.value)

    def test_invalid_import(self):
        """Test that non-math imports are rejected."""
        invalid_script = """import os

def main():
    return "test"
final_output = main()"""

        with pytest.raises(ValidationError) as exc_info:
            AbstractPlan(script=invalid_script, abs_tools=[])

        assert "not allowed" in str(exc_info.value)

    def test_valid_math_import(self):
        """Test that math imports are allowed."""
        valid_script = """import math

def main():
    result: float = math.sqrt(16)
    result_str: str = str(result)
    return result_str
final_output = main()"""

        plan = AbstractPlan(script=valid_script, abs_tools=[])
        assert plan.script == valid_script

    def test_allowed_builtins(self):
        """Test that allowed builtins work."""
        valid_script = """def main():
    numbers: frozenset = frozenset([1, 2, 3])
    total: int = sum(numbers)
    result: str = str(total)
    return result
final_output = main()"""

        plan = AbstractPlan(script=valid_script, abs_tools=[])
        assert plan.script == valid_script
