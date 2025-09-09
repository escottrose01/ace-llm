import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import pydantic
from ace.schema.concrete import ConcreteToolBase
from polyfactory.factories.pydantic_factory import ModelFactory

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


def make_factory(model_cls: type[pydantic.BaseModel]) -> type[ModelFactory]:
    class Factory(ModelFactory):
        __model__ = model_cls

    return Factory


@dataclass
class CaseConfig:
    all_tools: list[MockTool]
    user_tool: MockTool
    attacker_tools: list[MockTool]
    extra_context: str


class InjecAgentToolAdapter:
    def __init__(self, root: str, full_tool_suite: bool = False, full_tool_descs: bool = False):
        tool_path = os.path.join(root, "tools.json")
        with open(tool_path) as file:
            tool_data = json.load(file)
        response_path = os.path.join(root, "responses.json")
        with open(response_path) as file:
            responses = json.load(file)

        self.full_tool_suite = full_tool_suite
        self.full_tool_descs = full_tool_descs
        self.tool_suite_registry = dict()
        self.tool_templates = dict()
        self.toolkit_lookup = dict()
        self.agent_context = dict()

        # Sanitize responses
        raw_example_responses = dict()
        for k, resp in responses.items():
            m = re.match(r"^\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", k)
            tool_name = m.group(1) if m else ""
            raw_example_responses.setdefault(tool_name, []).append(resp)

        for toolkit in tool_data:
            toolkit_name: str = toolkit["toolkit"]
            toolkit_desc: str = toolkit["description_for_model"]
            self.tool_templates[toolkit_name] = []
            self.agent_context[toolkit_name] = toolkit_desc

            for tool in toolkit["tools"]:
                tool_name: str = tool["name"]
                tool_desc: str = tool["summary"]
                tool_fullname = f"{toolkit_name}{tool_name}"

                ArgsSchema = build_model_from_params(tool["parameters"], f"{tool_name}Input")  # noqa N806
                OutputSchema = build_model_from_params(tool["returns"], f"{tool_name}Output")  # noqa N806

                self.tool_suite_registry.setdefault(toolkit_name, set()).add(tool_fullname)
                self.toolkit_lookup[tool_fullname] = toolkit_name
                self.tool_templates[tool_fullname] = dict(
                    description=tool_desc, args_schema=ArgsSchema, output_schema=OutputSchema, default_ret=dict()
                )

                # Construct a dummy return value
                for resp in raw_example_responses.get(tool_fullname, []):
                    default_ret = ast.literal_eval(resp)

                    # InjecAgent has a typo for the GmailSearchEmails tool output,
                    # we hot-fix it dynamically on load here:
                    if tool == "GmailSearchEmails":
                        for e in default_ret["emails"]:
                            print("hotfixing incorrect key")
                            e["subject"] = e.pop("subeject", e["subject"])

                    # Some tool outputs do not conform correctly to the schema, so we need to validate
                    try:
                        OutputSchema(**default_ret)
                    except Exception as e:
                        continue

                    # If validation is successful, we can stop the search
                    self.tool_templates[tool_fullname]["default_ret"] = default_ret
                    print(f"Successfully constructed a valid response for {tool_fullname}")
                    break
                else:
                    print(f"Unable to construct a valid response for {tool_fullname}, making dummy data")
                    Factory = make_factory(OutputSchema)  # noqa: N806
                    self.tool_templates[tool_fullname]["default_ret"] = Factory.build().model_dump()

    def prepare_case(self, case: dict) -> CaseConfig:
        attacker_tool_names: list[str] = case["Attacker Tools"]
        user_tool_name: str = case["User Tool"]
        suites_used = {self.toolkit_lookup[name] for name in [*attacker_tool_names, user_tool_name]}

        # InjecAgent does not correctly escape strings in the attacker response,
        # so we have to reconstruct the attacker response manually from the template.
        attacker_response_template = case["Tool Response Template"]
        attacker_instruction = case["Attacker Instruction"]
        attacker_response = ast.literal_eval(
            attacker_response_template.replace("'<Attacker Instruction>'", repr(attacker_instruction))
        )

        all_tools: list[MockTool] = []
        user_tool = None
        attacker_tools = []
        if self.full_tool_suite:
            # Load the full tool suite. This differs from the InjecAgent tool configuration, but is more compatible with ACE's tool generation strategy.
            raise NotImplementedError(
                "Full tool suite mode is not supported since not all tools provide simulated outputs"
            )
            for suite in suites_used:
                for tool_name in self.tool_suite_registry.get(suite, []):
                    tool_template = self.tool_templates[tool_name]
                    print(tool_name)
                    print(tool_template["output_schema"].schema_json())
                    return_val = attacker_response if tool_name == user_tool_name else tool_template["default_ret"]
                    tool = MockTool(
                        name=tool_name,
                        provider="injecAgent",
                        description=tool_template["description"],
                        clearances=set(),
                        permissions=set(),
                        args_schema=tool_template["args_schema"],
                        output_schema=tool_template["output_schema"],
                        return_val=return_val,
                    )
                    all_tools.append(tool)

                    if tool_name == user_tool_name:
                        user_tool = tool
                    if tool_name in attacker_tool_names:
                        attacker_tools.append(tool)
        else:
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

            all_tools = [user_tool, *attacker_tools]

        # Build context string based on tool suites used
        if self.full_tool_descs:
            extra_context = "You have the following tools available:\n" + "\n".join(
                f"TOOL | {t.name} - {t.description}" for t in all_tools
            )
        else:
            extra_context = "You have the following functionality suites available:\n" + "\n".join(
                self.agent_context[s] for s in suites_used
            )

        if user_tool is None:
            raise ValueError("User tool not found")

        return CaseConfig(
            all_tools=all_tools, user_tool=user_tool, attacker_tools=attacker_tools, extra_context=extra_context
        )
