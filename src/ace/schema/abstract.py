import ast
import json

import astor
from pydantic import BaseModel, Field, model_validator


class ToolParameter(BaseModel):
    """Schema for tool input/output parameters."""

    name: str = Field(..., description="The name of the parameter.")
    description: str = Field(..., description="A natural language description of the parameter.")
    type: str = Field(
        ...,
        description="The type of the parameter (must be one of the allowed primitive types).",
        pattern="^(str|int|float|bool|tuple\\[str\\]|tuple\\[int\\]|tuple\\[float\\]|tuple\\[bool\\])$",
    )


class ToolSchema(BaseModel):
    """Schema for individual tool specification."""

    name: str = Field(..., description="The unique, brief name of the tool or function.")
    description: str = Field(
        ...,
        description="A natural language description explaining the tool's purpose and what task it helps accomplish.",
    )
    inputs: list[ToolParameter] = Field(
        ..., description="A list of input parameters each with a name, description, and type."
    )
    outputs: list[ToolParameter] = Field(
        ..., description="A list of output fields each with a name, description, and type."
    )


class ToolGenerationResponse(BaseModel):
    """Response model for tool generation."""

    apps: list[ToolSchema] = Field(
        ..., description="A list of function tool schemas designed to help accomplish a task."
    )


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
    """Converts attribute access (x.y) to string indexing (x["y"])."""

    # TODO: this is a hotfix. The correct implementation will use the type system more directly.
    def __init__(self, abstract_tools: list[AbstractTool]):
        self.abstract_tools = abstract_tools
        # Create mapping from tool name to output schema fields
        self.tool_output_fields = {}
        for tool in abstract_tools:
            self.tool_output_fields[tool.name] = set(tool.output_schema.model_fields.keys())

        # Track variables annotated with tool output types
        self.tool_output_vars = {}  # var_name -> tool_name

    def visit_AnnAssign(self, node):
        """Handle annotated assignments like: var: ToolType = ..."""
        self.generic_visit(node)

        # Check if the annotation is a tool output type
        if (
            isinstance(node.annotation, ast.Name)
            and node.annotation.id in self.tool_output_fields
            and isinstance(node.target, ast.Name)
        ):
            # Track this variable as having a tool output type
            self.tool_output_vars[node.target.id] = node.annotation.id

        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)

        # Only transform if:
        # 1. The base is a variable we're tracking as a tool output
        # 2. The attribute is actually a field in that tool's output schema
        if isinstance(node.value, ast.Name) and node.value.id in self.tool_output_vars:
            tool_name = self.tool_output_vars[node.value.id]
            if node.attr in self.tool_output_fields[tool_name]:
                return ast.Subscript(
                    value=node.value,
                    slice=ast.Constant(value=node.attr),
                    ctx=node.ctx,
                )

        return node


class AbstractPlan(BaseModel):
    script: str
    abs_tools: list[AbstractTool]

    @model_validator(mode="after")
    def validate_script_syntax(self) -> "AbstractPlan":
        """Validate that the script follows our restricted Python grammar and semantics."""
        try:
            # Ensure the script can be parsed as valid Python
            ast.parse(self.script)
        except SyntaxError as e:
            raise ValueError(f"Script contains invalid Python syntax: {e}")

        return self

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

        # Convert attribute access to string indexing (only for tool outputs)
        annotation_transformer = AnnotationTransformer(self.abs_tools)
        prog = annotation_transformer.visit(prog)
        ast.fix_missing_locations(prog)

        return astor.to_source(prog)
