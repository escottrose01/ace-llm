from abc import ABC, abstractmethod
from typing import Type, Optional, Any

import ast
from pydantic import BaseModel, model_validator, create_model
import json

from .permissions import Permission

ALLOWED_TYPES_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool
}


class ConcreteToolBase(BaseModel, ABC):
    name: str
    provider: Optional[str]
    description: Optional[str]
    clearances: set[str]
    permissions: set[Permission]
    args_schema: Optional[Type[BaseModel]]
    output_schema: Optional[Type[BaseModel]]

    @abstractmethod
    def generate_source(self) -> str:
        pass

    def compile(self) -> ast.Module:
        return ast.parse(self.generate_source())

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema.model_json_schema(),
            "output_schema": self.output_schema.model_json_schema()
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=4)


class CustomTool(ConcreteToolBase):
    source_code: str

    @model_validator(mode="before")
    def validate_and_infer_types(
        cls,
        data: dict[str, Any]
    ) -> dict[str, Any]:
        # Validate code syntax
        code = data.get("source_code")
        if not code:
            raise ValueError("source_code is required.")
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in source_code: {e}")

        # Tool grammar rule: must contain a single function named main
        main_func_check = (
            len(tree.body) == 1
            and isinstance(tree.body[0], ast.FunctionDef)
            and tree.body[0].name == "main"
        )

        if not main_func_check:
            raise ValueError(
                "source_code must contain a single function named main.")

        return data

    def generate_source(self) -> str:
        return self.source_code


class LangChainTool(ConcreteToolBase):
    function_name: str

    @model_validator(mode="before")
    def validate_and_infer_types(
        cls,
        data: dict[str, Any]
    ) -> dict[str, Any]:
        function_name = data.get("function_name")
        if not function_name:
            raise ValueError("function_name is required.")

        try:
            import langchain_community.tools as lc_tools
        except ImportError as e:
            raise ValueError(
                "Could not import langchain_community.tools: "
                "make sure it is installed."
            )

        if not hasattr(lc_tools, function_name):
            raise ValueError(
                f"Function {function_name} not found in langchain_community.tools."
            )

        # Infer args_schema and output_schema
        args_fields = dict()
        func = getattr(lc_tools, function_name)
        if func.__annotations__:
            for arg, arg_type in func.__annotations__.items():
                if arg == "return":
                    output_schema = arg_type
                else:
                    args_fields[arg] = (arg_type, ...)
            args_schema = create_model(f"{function_name}Args", **args_fields)
        else:
            args_schema = create_model(f"{function_name}Args")

        if func.__annotations__.get("return"):
            output_schema = func.__annotations__["return"]
            output_schema = create_model(
                f"{function_name}Output", product=(output_schema, ...))
        else:
            output_schema = create_model(f"{function_name}Output")

        # data["description"] = "TODO: get description from langchain_community.tools"
        data["description"] = func.__doc__
        data["args_schema"] = args_schema
        data["output_schema"] = output_schema

        return data

    def generate_source(self) -> str:
        return (
            f"from langchain_community.tools import {self.function_name}\n\n"
            f"def main(*args, **kwargs):\n"
            f"    tool_instance = {self.function_name}()\n"
            f"    return tool_instance.run(*args, **kwargs)\n"

        )


# TODO:
# check out https://github.com/letta-ai/letta/blob/main/letta/schemas/tool.py
