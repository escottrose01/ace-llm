import json
import logging
from pathlib import Path
from typing import Any

from ..schema import ConcreteToolBase, CustomTool, LangChainTool, Permission
from .helper import parse_schemas_from_json

logger = logging.getLogger(__name__)


class ToolManager:
    tools: list[ConcreteToolBase]

    def __init__(self, tools: list[ConcreteToolBase]):
        self.tools = tools

    def get_by_name(self, name: str) -> ConcreteToolBase:
        # yeah yeah, I know.
        for tool in self.tools:
            if tool.name == name:
                return tool

        raise ValueError(f"Tool {name} not found.")

    @classmethod
    def from_manifest(cls, manifest_path: str | Path):
        tools = []

        if isinstance(manifest_path, str):
            manifest_path = Path(manifest_path)

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file {manifest_path} not found.")

        with open(manifest_path) as f:
            manifest = json.load(f)

        for item in manifest:
            try:
                tool = instantiate_tool_from_manifest(item, manifest_path)
                tools.append(tool)
            except Exception as e:
                logger.warning(f"Error instantiating tool {item['name']}: {e}")

        return cls(tools)


def instantiate_tool_from_manifest(item: dict[str, Any], manifest_path: Path) -> ConcreteToolBase:
    # TODO: should maybe make schema or ORM for manifest file . . .
    tool_type = item["tool_type"]

    name = item["name"]
    clearances = item["clearances"]
    permissions = set(map(Permission, item["permissions"]))

    if tool_type == "custom":
        provider = item["provider"]
        description = item["description"]
        schema_path = manifest_path.parent / "schema" / f"{item['path']}.json"
        tool_path = manifest_path.parent / "source" / f"{item['path']}.py"

        with open(tool_path) as f:
            source_code = f.read()
        with open(schema_path) as f:
            schema_json = json.load(f)

        args_schema, output_schema = parse_schemas_from_json(schema_json)

        return CustomTool(
            name=name,
            provider=provider,
            description=description,
            clearances=clearances,
            permissions=permissions,
            source_code=source_code,
            args_schema=args_schema,
            output_schema=output_schema,
        )
    elif tool_type == "langchain":
        return LangChainTool(
            name=name,
            provider="LangChain Community",
            description=None,  # Will be inferred by model_validator
            clearances=clearances,
            permissions=permissions,
            args_schema=None,  # Will be inferred by model_validator
            output_schema=None,  # Will be inferred by model_validator
            function_name=item["function_name"],
        )
    else:
        raise ValueError(f"Unknown tool type: {tool_type}")
