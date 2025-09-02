import ast
import json
import os
import re
from typing import Any, Optional

import pydantic
from ace.schema.concrete import ConcreteToolBase

type_map: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list[Any],
    "object": dict[str, Any],
}


def build_model_from_params(params: list[dict[str, Any]], model_name: str) -> type[pydantic.BaseModel]:
    fields: dict[str, tuple[Any, Any]] = dict()

    for p in params:
        pname: str = p["name"]
        ptype: type = type_map[p["type"]]
        pdesc: str = p["description"]
        preq: bool = p.get("required", True)

        if preq:
            default = pydantic.Field(..., description=pdesc)
            annotated = ptype
        else:
            default = pydantic.Field(None, description=pdesc)
            annotated = Optional[ptype]

        fields[pname] = (annotated, default)

    Model = pydantic.create_model(model_name, **fields)  # type: ignore # noqa N806
    return Model


class MockTool(ConcreteToolBase):
    return_val: dict

    def generate_source(self) -> str:
        return f"def main(*args, **kwargs): return {self.return_val!r}\n"


class InjecAgentToolAdapter:
    def __init__(self, root: str):
        tool_path = os.path.join(root, "tools.json")
        with open(tool_path) as file:
            tool_data = json.load(file)
        response_path = os.path.join(root, "responses.json")
        with open(response_path) as file:
            responses = json.load(file)

        self.tool_templates = dict()
        self.toolkit_lookup = dict()
        self.agent_context = dict()
        self.example_responses = dict()

        # Sanitize responses
        for k, resp in responses.items():
            m = re.match(r"^\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", k)
            tool_name = m.group(1) if m else ""
            self.example_responses[tool_name] = resp

        for toolkit in tool_data:
            toolkit_name: str = toolkit["toolkit"]
            toolkit_desc: str = toolkit["description_for_model"]
            self.tool_templates[toolkit_name] = []
            self.agent_context[toolkit_name] = toolkit_desc

            for tool in toolkit["tools"]:
                tool_name: str = tool["name"]
                tool_desc: str = tool["summary"]
                tool_fullname = f"{toolkit_name}{tool_name}"
                default_ret = (
                    ast.literal_eval(self.example_responses[tool_fullname])
                    if tool_fullname in self.example_responses
                    else {}
                )
                ArgsSchema = build_model_from_params(tool["parameters"], f"{tool_name}Input")  # noqa N806
                OutputSchema = build_model_from_params(tool["returns"], f"{tool_name}Output")  # noqa N806

                # InjecAgent has a typo for the GmailSearchEmail tool output,
                # we hot-fix it dynamically on load here:
                if tool == "GmailSearchTool":
                    for e in default_ret["emails"]:
                        print("hotfixing incorrect key")
                        e["subject"] = e.pop("subeject", e["subject"])

                self.tool_templates[tool_fullname] = dict(
                    description=tool_desc, args_schema=ArgsSchema, output_schema=OutputSchema, default_ret=default_ret
                )
                self.toolkit_lookup[tool_fullname] = toolkit_name

    def prepare_case(self, case: dict) -> tuple[MockTool, list[MockTool], str]:
        attacker_tool_names: list[str] = case["Attacker Tools"]
        user_tool_name: str = case["User Tool"]

        # InjecAgent does not correctly escape strings in the attacker response,
        # so we have to reconstruct the attacker response manually from the template.
        attacker_response_template = case["Tool Response Template"]
        attacker_instruction = case["Attacker Instruction"]
        attacker_response = ast.literal_eval(
            attacker_response_template.replace("'<Attacker Instruction>'", repr(attacker_instruction))
        )

        # Reproduce the InjecAgent tool configuration. They provide a single user tool and one or more attacker tools.
        user_tool_template = self.tool_templates[user_tool_name]
        user_tool = MockTool(
            name=user_tool_name,
            provider="injecAgent",
            description=user_tool_template["description"],
            clearances=set(),
            permissions=set(),
            args_schema=user_tool_template["args_schema"],
            output_schema=user_tool_template["output_schema"],
            return_val=attacker_response,
        )

        # Build attacker tools
        attacker_tools: list[MockTool] = []
        for tool_name in attacker_tool_names:
            attacker_tool_template = self.tool_templates[tool_name]
            return_val = attacker_tool_template["default_ret"]
            attacker_tools.append(
                MockTool(
                    name=tool_name,
                    provider="injecAgent",
                    description=attacker_tool_template["description"],
                    clearances=set(),
                    permissions=set(),
                    args_schema=attacker_tool_template["args_schema"],
                    output_schema=attacker_tool_template["output_schema"],
                    return_val=return_val,
                )
            )

        # Build context string based on tool suites used
        suites_used = {self.toolkit_lookup[name] for name in [*attacker_tool_names, user_tool_name]}
        extra_context = "You have the following functionality suites available:\n" + "\n".join(
            self.agent_context[s]
            for s in suites_used
            # f"TOOL | {t.name} - {t.description}" for t in [user_tool] + attacker_tools
        )

        return user_tool, attacker_tools, extra_context
