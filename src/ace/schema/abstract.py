import ast
import json

import astor
from pydantic import BaseModel, model_validator


class AbstractTool(BaseModel):
    name: str
    description: str
    args_schema: type[BaseModel]
    output_schema: type[BaseModel]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema.model_json_schema(),
            "output_schema": self.output_schema.model_json_schema(),
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=None)


class ToolCallTransformer(ast.NodeTransformer):
    def __init__(self, tool_functions: set[str]):
        self.tool_functions = tool_functions

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.tool_functions:
            # Replace call: foo(arg1, arg2, ...) with invoke("foo", arg1, arg2, ...)
            new_args: list[ast.expr] = [ast.Constant(value=node.func.id)]
            new_args.extend(node.args)
            new_node = ast.Call(func=ast.Name(id="invoke", ctx=ast.Load()), args=new_args, keywords=node.keywords)
            return ast.copy_location(new_node, node)
        return node


class AnnotationTransformer(ast.NodeTransformer):
    """Converts attribute access (x.y) to string indexing (x["y"]) in the AST."""

    def visit_Attribute(self, node):
        self.generic_visit(node)
        return ast.Subscript(
            value=node.value,
            slice=ast.Constant(value=node.attr),
            ctx=node.ctx,
        )


class AbstractPlan(BaseModel):
    script: str
    abs_tools: list[AbstractTool]

    @model_validator(mode="after")
    def validate_script_syntax(self) -> "AbstractPlan":
        """Validate that the script follows our restricted Python grammar and semantics."""
        try:
            # First, ensure the script can be parsed as valid Python
            prog = ast.parse(self.script)
        except SyntaxError as e:
            raise ValueError(f"Script contains invalid Python syntax: {e}")

        # Basic structure validation
        main = self._validate_program_structure(prog)

        # Validate import rules
        module_validator = ModuleLevelValidator()
        module_validator.validate(prog)

        # Advanced grammar rules and semantics validation
        grammar_validator = PlanGrammarValidator(self.abs_tools)
        grammar_validator.validate(main)

        return self

    @classmethod
    def _validate_program_structure(cls, prog: ast.Module) -> ast.FunctionDef:
        """Validate the basic structure of the program and return imports and main function."""
        # Check that all nodes are either Import, ImportFrom, FunctionDef, or Assign
        allowed_node_types = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign)
        for node in prog.body:
            if not isinstance(node, allowed_node_types):
                raise ValueError(
                    f"Invalid node type '{type(node).__name__}' in program. "
                    "Only imports, function definitions, and assignments are allowed."
                )

        # Collect imports at the top
        imports = []
        import_count = 0
        for node in prog.body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                imports.append(node)
                import_count += 1
            else:
                break  # Stop at first non-import

        # After imports, expect a single main function definition
        if len(prog.body) < import_count + 2:
            raise ValueError("Program must contain a main function and a final_output assignment after imports")

        # The next node must be a FunctionDef
        if not isinstance(prog.body[import_count], ast.FunctionDef):
            raise ValueError("Main function definition must follow imports")

        # The last node must be an assignment to final_output
        if not isinstance(prog.body[-1], ast.Assign):
            raise ValueError("Program must end with 'final_output = main()' assignment")

        # All nodes after imports except the last must be the main function
        if len(prog.body) != import_count + 2:
            raise ValueError(
                "Program must contain only imports, a single main function definition, and a final_output assignment"
            )

        # Must have exactly one function definition at module level
        function_defs = [node for node in prog.body if isinstance(node, ast.FunctionDef)]

        if len(function_defs) != 1:
            raise ValueError(f"Program must contain exactly one function definition, found {len(function_defs)}")

        main_func = function_defs[0]

        # Function must be named 'main'
        if main_func.name != "main":
            raise ValueError(f"Function must be named 'main', found '{main_func.name}'")

        # Function must take no arguments
        if main_func.args.args or main_func.args.posonlyargs or main_func.args.kwonlyargs:
            raise ValueError("main() function must take no arguments")

        # Must have a final_output assignment at the end
        assignments = [node for node in prog.body if isinstance(node, ast.Assign)]

        if not assignments:
            raise ValueError("Program must end with 'final_output = main()' assignment")

        last_assignment = assignments[-1]

        # Check if it's assigning to 'final_output'
        if len(last_assignment.targets) != 1:
            raise ValueError("final_output assignment must have exactly one target")

        target = last_assignment.targets[0]
        if not isinstance(target, ast.Name) or target.id != "final_output":
            raise ValueError("Program must end with assignment to 'final_output'")

        # Check if the value is a call to main()
        value = last_assignment.value
        if not isinstance(value, ast.Call):
            raise ValueError("final_output must be assigned the result of calling main()")

        if not isinstance(value.func, ast.Name) or value.func.id != "main":
            raise ValueError("final_output must be assigned the result of calling main()")

        # main() call should have no arguments
        if value.args or value.keywords:
            raise ValueError("main() call must have no arguments")

        return main_func

    def compile_for_analysis(self) -> ast.Module:
        # Parse the script into an AST
        prog = ast.parse(self.script)
        return prog

    def compile_for_protocol(self) -> str:
        prog = ast.parse(self.script)

        # Replace function calls with invoke statements
        tool_functions = {tool.name for tool in self.abs_tools}
        transformer = ToolCallTransformer(tool_functions)
        prog = transformer.visit(prog)

        # Convert attribute access to string indexing
        annotation_transformer = AnnotationTransformer()
        prog = annotation_transformer.visit(prog)
        ast.fix_missing_locations(prog)

        return astor.to_source(prog)


class PlanGrammarValidator(ast.NodeVisitor):
    """AST visitor that validates our restricted Python grammar rules."""

    # Allowed builtin functions
    ALLOWED_BUILTINS = frozenset(
        {"abs", "bool", "float", "int", "str", "frozenset", "all", "any", "len", "pow", "round", "sum"}
    )

    # Primitive types that are always allowed
    PRIMITIVE_TYPES = frozenset({"bool", "float", "int", "str", "frozenset", "tuple"})

    # Blacklisted builtin functions
    BLACKLISTED_BUILTINS = frozenset(
        {
            "open",
            "exec",
            "eval",
            "compile",
            "__import__",
            "input",
            "globals",
            "locals",
            "vars",
            "dir",
            "help",
            "exit",
            "quit",
            "getattr",
            "setattr",
            "delattr",
            "super",
            "memoryview",
        }
    )

    # Built-in functions that are allowed in our language
    LANGUAGE_BUILTINS = frozenset({"display"})

    def __init__(self, abstract_tools: list[AbstractTool] | None = None):
        self.variables: dict[str, str] = {}  # variable_name -> type_name
        self.errors: list[str] = []
        self.in_control_flow = False  # Track if we're inside if/else/loop

        # Build allowed types from abstract tools
        self.allowed_types = set(self.PRIMITIVE_TYPES)
        if abstract_tools:
            for tool in abstract_tools:
                # Add tool names as valid callable types
                self.allowed_types.add(tool.name)

    def validate(self, main_func: ast.FunctionDef) -> None:
        """Validate the main function and raise ValueError if any rules are violated."""
        self.visit(main_func)

        if self.errors:
            raise ValueError(f"Grammar validation failed: {'; '.join(self.errors)}")

    def visit_If(self, node: ast.If) -> None:
        """Mark that we're entering control flow."""
        old_in_control_flow = self.in_control_flow
        self.in_control_flow = True
        self.generic_visit(node)
        self.in_control_flow = old_in_control_flow

    def visit_For(self, node: ast.For) -> None:
        """Mark that we're entering control flow."""
        old_in_control_flow = self.in_control_flow
        self.in_control_flow = True
        self.generic_visit(node)
        self.in_control_flow = old_in_control_flow

    def visit_While(self, node: ast.While) -> None:
        """Mark that we're entering control flow."""
        old_in_control_flow = self.in_control_flow
        self.in_control_flow = True
        self.generic_visit(node)
        self.in_control_flow = old_in_control_flow

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Validate annotated assignments (var: type = value)."""
        if not isinstance(node.target, ast.Name):
            self.errors.append("Only simple variable assignments allowed")
            return

        var_name = node.target.id

        # Check if we're in control flow
        if self.in_control_flow:
            self.errors.append(f"Variable '{var_name}' cannot be declared inside if/else/loop blocks")
            return

        # Check if variable is already declared
        if var_name in self.variables:
            self.errors.append(f"Variable '{var_name}' is already declared. Variables can only be declared once")
            return

        # Extract and validate type annotation
        if isinstance(node.annotation, ast.Name):
            type_name = node.annotation.id
            if type_name not in self.allowed_types:
                self.errors.append(f"Type '{type_name}' is not allowed. Allowed types: {sorted(self.allowed_types)}")
                return

            self.variables[var_name] = type_name
        else:
            self.errors.append(f"Complex type annotations not supported for variable '{var_name}'")

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Validate regular assignments (should only be reassignments)."""
        if len(node.targets) != 1:
            self.errors.append("Multiple assignment targets not allowed")
            return

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            self.errors.append("Only simple variable assignments allowed")
            return

        var_name = target.id

        # Variables should be declared with type annotations first
        if var_name not in self.variables:
            self.errors.append(f"Variable '{var_name}' must be declared with a type annotation before use")

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Validate function calls."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Check for blacklisted builtins
            if func_name in self.BLACKLISTED_BUILTINS:
                self.errors.append(f"Builtin function '{func_name}' is not allowed")

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Validate augmented assignments (+=, -=, etc.)."""
        self.errors.append("Augmented assignments not allowed - use reconstruction instead")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Only visit the main function, no nested functions allowed."""
        if node.name != "main":
            self.errors.append("Nested function definitions not allowed")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Class definitions not allowed."""
        self.errors.append("Class definitions not allowed")

    def visit_Global(self, node: ast.Global) -> None:
        """Global statements not allowed."""
        self.errors.append("Global statements not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Nonlocal statements not allowed."""
        self.errors.append("Nonlocal statements not allowed")


class ModuleLevelValidator(ast.NodeVisitor):
    """Validator for module-level constructs like imports."""

    ALLOWED_IMPORTS = frozenset({"math"})

    def __init__(self):
        self.errors: list[str] = []

    def validate(self, module: ast.Module) -> None:
        """Validate the module and raise ValueError if any rules are violated."""
        self.visit(module)

        if self.errors:
            raise ValueError(f"Module validation failed: {'; '.join(self.errors)}")

    def visit_Import(self, node: ast.Import) -> None:
        """Validate import statements."""
        for alias in node.names:
            if alias.name not in self.ALLOWED_IMPORTS:
                self.errors.append(
                    f"Import of '{alias.name}' not allowed. Allowed modules: {', '.join(self.ALLOWED_IMPORTS)}"
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Validate from imports."""
        if node.module not in self.ALLOWED_IMPORTS:
            self.errors.append(
                f"Import from '{node.module}' not allowed. Allowed modules: {', '.join(self.ALLOWED_IMPORTS)}"
            )


TOOL_GENERATION_SCHEMA = {
    "name": "function_tool_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "apps": {
                "type": "array",
                "description": "A list of function tool schemas designed to help accomplish a task.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The unique, brief name of the tool or function."},
                        "description": {
                            "type": "string",
                            "description": "A natural language description explaining the tool's purpose and what task it helps accomplish.",
                        },
                        "inputs": {
                            "type": "array",
                            "description": "A list of input parameters each with a name, description, and type.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "The name of the input parameter."},
                                    "description": {
                                        "type": "string",
                                        "description": "A natural language description of the input parameter.",
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "The type of the input parameter (must be one of the allowed primitive types).",
                                        "enum": ["str", "int", "float", "bool"],
                                    },
                                },
                                "required": ["name", "description", "type"],
                                "additionalProperties": False,
                            },
                        },
                        "outputs": {
                            "type": "array",
                            "description": "A list of output fields each with a name, description, and type.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "The name of the output field."},
                                    "description": {
                                        "type": "string",
                                        "description": "A natural language description of the output field.",
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "The type of the output field (must be one of the allowed primitive types).",
                                        "enum": ["str", "int", "float", "bool"],
                                    },
                                },
                                "required": ["name", "description", "type"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["name", "description", "inputs", "outputs"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["apps"],
        "additionalProperties": False,
    },
}
