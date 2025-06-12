import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, create_model

from ..prompts.abstract_templates import generate_abstract_plan_template, generate_abstract_tool_template
from ..schema import AbstractPlan, AbstractTool


def parse_text_to_python(text: str | AIMessage) -> str:
    if isinstance(text, AIMessage):
        text = text.content
    pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1)
    else:
        # Fallback: assume the entire text is code.
        code = text
    # Remove any leading or trailing whitespace.
    code = code.strip()
    return code


def create_pydantic_schema(fields: dict[str, dict[str, str]]) -> type[BaseModel]:
    schema_fields = {}
    for field_name, field_props in fields.items():
        # Convert string type to actual Python type
        field_type = eval(field_props["type"])
        field_description = field_props.get("description", "")
        schema_fields[field_name] = (field_type, Field(..., description=field_description))

    return create_model("ArgsSchema", **schema_fields)


class AbstractPlanner:
    base_llm: BaseChatModel
    toolgen_chain: Runnable
    plangen_chain: Runnable

    def __init__(self, base_llm):
        self.base_llm = base_llm

        toolgen_prompt = generate_abstract_tool_template()
        plangen_prompt = generate_abstract_plan_template()
        json_parser = JsonOutputParser()

        # These chains will need to be updated to use the new LLM interface if they use .invoke
        self.toolgen_chain = toolgen_prompt | self.base_llm | json_parser
        self.plangen_chain = plangen_prompt | self.base_llm

        # Last generated tools. Used for test trials
        self.tools = {}

    def generate_abstract_tools(self, query) -> dict:
        output = self.toolgen_chain.invoke({"input": query})
        return output

    def generate_abstract_plan(self, query):
        abstract_tools = self.generate_abstract_tools(query)
        self.tools = abstract_tools
        abstract_plan = self.plangen_chain.invoke({"input": query, "tools": abstract_tools})

        abstract_tool_list = []
        for tool in abstract_tools["apps"]:
            input_ = tool.get("input", tool.get("inputs"))
            output_ = tool.get("output", tool.get("outputs"))
            tool_input_schema = create_pydantic_schema(input_)
            tool_output_schema = create_pydantic_schema({"output": output_})

            abstract_tool = AbstractTool(
                name=tool["name"],
                description=tool["description"],
                args_schema=tool_input_schema,
                output_schema=tool_output_schema,
            )

            abstract_tool_list.append(abstract_tool)

        return AbstractPlan(script=parse_text_to_python(abstract_plan), abs_tools=abstract_tool_list)
