from typing import Any, Type, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field, create_model
from langchain.tools import Tool as LangchainTool


def generate_langchain_tool_source(
    tool: LangchainTool
) -> str:
    tool_name = tool.__class__.__name__
    import_stmt = f"from langchain.tools import {tool_name}\n"

    # TODO: borrow from https://github.com/letta-ai/letta/blob/a1a2dd44f57ff868d46e7e4bc517e4d299185771/letta/functions/helpers.py#L104

    source_code = None
    return source_code


def json_schema_to_base_model(schema: dict[str, Any]) -> Type[BaseModel]:
    """
    Recursively converts a JSON schema into a Pydantic model.
    """
    type_mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    properties = schema.get("properties")
    required_fields = schema.get("required", [])
    model_fields = {}

    def process_field(field_name: str, field_props: dict[str, Any]) -> tuple[type, Field]:
        """
        Recursively processes a field and returns its type and a Pydantic Field instance.
        """
        json_type = field_props.get("type")
        enum_values = field_props.get("enum")

        if not json_type:
            raise ValueError(f"Field '{field_name}' is missing a 'type' key.")

        # Handle Enums
        if enum_values:
            enum_name: str = f"{field_name.capitalize()}Enum"
            # Note: This may fail if enum values are not valid Python identifiers.
            field_type = Enum(enum_name, {str(v): v for v in enum_values})
        # Handle Nested Objects
        elif json_type == "object" and "properties" in field_props:
            field_type = json_schema_to_base_model(
                field_props)  # Recursively create submodel
        # Handle Arrays with Nested Objects
        elif json_type == "array" and "items" in field_props:
            item_props = field_props["items"]
            if item_props.get("type") == "object":
                item_type: type[BaseModel] = json_schema_to_base_model(
                    item_props)
            else:
                item_type: type = type_mapping.get(item_props.get("type"), Any)
            field_type = list[item_type]
        else:
            field_type = type_mapping.get(json_type, Any)

        # Handle default values, optionality, and descriptions
        default_value = field_props.get("default", ...)
        nullable = field_props.get("nullable", False)
        # Use "description" if available, fallback to "title"
        description = field_props.get(
            "description", field_props.get("title", ""))

        if nullable:
            field_type = Optional[field_type]

        if field_name not in required_fields:
            default_value = field_props.get("default", None)

        return field_type, Field(default_value, description=description)

    # Process each field in the schema's properties
    if properties is None:
        json_type = schema.get("type", "string")
        result_type = type_mapping.get(json_type, Any)
        description = schema.get("description", schema.get("title", ""))
        return create_model(
            schema.get("title", "DynamicModel"),
            result=(result_type, Field(..., description=description))
        )
    else:
        for field_name, field_props in properties.items():
            model_fields[field_name] = process_field(field_name, field_props)

    # Use the schema's title as the model name, defaulting to "DynamicModel"
    return create_model(schema.get("title", "DynamicModel"), **model_fields)


def parse_schemas_from_json(schema: dict[str, Any]) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    # Assume there is one top-level key representing the tool name.
    top_level_props = schema.get("properties", {})
    if not top_level_props:
        raise ValueError("The schema must contain a 'properties' key.")

    tool_key = list(top_level_props.keys())[0]
    tool_schema = top_level_props[tool_key]
    tool_properties = tool_schema.get("properties", {})

    if "request" not in tool_properties or "response" not in tool_properties:
        raise ValueError(
            "The tool schema must contain both 'request' and 'response' keys.")
    if len(tool_properties["request"]["required"]) == 0:
        pass

    args_schema = json_schema_to_base_model(tool_properties["request"])
    output_schema = json_schema_to_base_model(tool_properties["response"])
    return args_schema, output_schema
