import ast
import json

import astor
from pydantic import BaseModel


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
        return json.dumps(self.as_dict(), indent=4)


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


class AbstractPlan(BaseModel):
    script: str
    abs_tools: list[AbstractTool]

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
        ast.fix_missing_locations(prog)

        return astor.to_source(prog)


# TODO: can type check the program here in validator?


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
