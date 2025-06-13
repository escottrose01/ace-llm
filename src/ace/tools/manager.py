import json
from pathlib import Path
from typing import Any

from ..logging_config import get_logger
from ..schema import ConcreteToolBase, CustomTool, LangChainTool, Permission
from .helper import parse_schemas_from_json

logger = get_logger(__name__)


class ToolManager:
    tools: list[ConcreteToolBase]

    def __init__(self, tools: list[ConcreteToolBase]):
        logger.info(f"Initializing ToolManager with {len(tools)} tools")
        self.tools = tools
        logger.debug(f"Available tools: {[tool.name for tool in tools]}")

    def get_by_name(self, name: str) -> ConcreteToolBase:
        logger.debug(f"Looking up tool by name: {name}")
        # yeah yeah, I know.
        for tool in self.tools:
            if tool.name == name:
                logger.debug(f"Found tool: {name}")
                return tool

        logger.error(f"Tool {name} not found in available tools: {[tool.name for tool in self.tools]}")
        raise ValueError(f"Tool {name} not found.")

    @classmethod
    def from_manifest(cls, manifest_path: str | Path):
        tools = []

        if isinstance(manifest_path, str):
            manifest_path = Path(manifest_path)

        if not manifest_path.exists():
            logger.error(f"Manifest file {manifest_path} not found")
            raise FileNotFoundError(f"Manifest file {manifest_path} not found.")

        logger.info(f"Loading tools from manifest: {manifest_path}")

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            logger.debug(f"Manifest loaded with {len(manifest)} tool definitions")
        except Exception as e:
            logger.error(f"Failed to read manifest file: {e}", exc_info=True)
            raise

        for i, item in enumerate(manifest):
            try:
                tool_name = item.get("name", f"tool_{i}")
                logger.debug(f"Instantiating tool {i + 1}/{len(manifest)}: {tool_name}")
                tool = instantiate_tool_from_manifest(item, manifest_path)
                tools.append(tool)
                logger.debug(f"Successfully loaded tool: {tool_name}")
            except Exception as e:
                logger.warning(f"Error instantiating tool {item.get('name', 'unknown')}: {e}")

        logger.info(f"Successfully loaded {len(tools)} tools from manifest")
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
