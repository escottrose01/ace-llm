import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, create_model

from ..logging_config import get_logger
from ..prompts.abstract_templates import generate_abstract_plan_template, generate_abstract_tool_template
from ..schema import TOOL_GENERATION_SCHEMA, AbstractPlan, AbstractTool

logger = get_logger(__name__)


def parse_text_to_python(text: str | AIMessage) -> str:
    if isinstance(text, AIMessage):
        content = text.content
        if isinstance(content, list):
            # Extract text from list of content pieces
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item["text"]))
            text = " ".join(text_parts)
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

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


def create_pydantic_schema(fields: list[dict[str, str]]) -> type[BaseModel]:
    schema_fields = {}
    for item in fields:
        print(item)
        name = item["name"]
        type = item["type"]
        desc = item["description"]

        # Extract type
        match type:
            case "int":
                field_type = int
            case "float":
                field_type = float
            case "str":
                field_type = str
            case "bool":
                field_type = bool
            case _:
                raise ValueError(f"Unsupported type: {type}")
        schema_fields[name] = (field_type, Field(..., description=desc))

    return create_model("ArgsSchema", **schema_fields)


class AbstractPlanner:
    base_llm: BaseChatModel
    toolgen_chain: Runnable
    plangen_chain: Runnable

    def __init__(self, base_llm: BaseChatModel):
        logger.info("Initializing AbstractPlanner")
        self.base_llm = base_llm

        # Create prompt templates
        toolgen_prompt = generate_abstract_tool_template()
        plangen_prompt = generate_abstract_plan_template()

        # Generation chains
        self.toolgen_chain = toolgen_prompt | self.base_llm.with_structured_output(TOOL_GENERATION_SCHEMA)
        self.plangen_chain = plangen_prompt | self.base_llm

    def generate_abstract_tools(self, query) -> dict:
        logger.debug(f"Generating abstract tools for query: {query[:100]}...")
        logger.log_query(query)

        try:
            output = self.toolgen_chain.invoke({"input": query})
            logger.info(f"Generated {len(output.get('apps', []))} abstract tools")
            logger.debug(f"Abstract tools: {[app.get('name', 'unnamed') for app in output.get('apps', [])]}")

            # Log full tool definitions
            for i, app in enumerate(output.get("apps", [])):
                tool_name = app.get("name", "unnamed")
                logger.log_structured("info", f"ABSTRACT_TOOL_{i + 1}", f"name={tool_name}")
                logger.log_structured("debug", f"ABSTRACT_TOOL_{i + 1}_FULL", str(app))

            return output
        except Exception as e:
            logger.error(f"Failed to generate abstract tools: {e}", exc_info=True)
            raise

    def generate_abstract_plan(self, query):
        logger.info("Starting abstract plan generation")

        # Generate abstract tools first
        abstract_tools = self.generate_abstract_tools(query)

        # Generate the plan using the tools
        logger.debug("Generating plan script using abstract tools")
        try:
            abstract_plan = self.plangen_chain.invoke({"input": query, "tools": abstract_tools})
            logger.debug("Plan script generation completed")

            # Log the raw plan output using structured logging
            logger.log_raw_llm_output("abstract_plan", str(abstract_plan))

        except Exception as e:
            logger.error(f"Failed to generate abstract plan script: {e}", exc_info=True)
            raise

        # Convert tools to AbstractTool objects
        abstract_tool_list = []
        for i, tool in enumerate(abstract_tools["apps"]):
            logger.debug(
                f"Processing abstract tool {i + 1}/{len(abstract_tools['apps'])}: {tool.get('name', 'unnamed')}"
            )
            try:
                tool_input_schema = create_pydantic_schema(tool["inputs"])
                tool_output_schema = create_pydantic_schema(tool["outputs"])

                abstract_tool = AbstractTool(
                    name=tool["name"],
                    description=tool["description"],
                    args_schema=tool_input_schema,
                    output_schema=tool_output_schema,
                )

                abstract_tool_list.append(abstract_tool)
                logger.debug(f"Successfully created AbstractTool: {tool['name']}")
            except Exception as e:
                logger.error(f"Failed to create AbstractTool for {tool.get('name', 'unnamed')}: {e}")
                raise

        parsed_script = parse_text_to_python(abstract_plan)
        logger.info(f"Abstract plan generation completed with {len(abstract_tool_list)} tools")
        logger.debug(f"Script length: {len(parsed_script)} characters")

        # Log the final abstract plan using enhanced logger
        logger.log_plan(parsed_script, abstract_tool_list)

        return AbstractPlan(script=parsed_script, abs_tools=abstract_tool_list)
