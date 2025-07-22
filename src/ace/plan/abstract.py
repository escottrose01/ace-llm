import re
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field, create_model

from ..analysis import Analyzer
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
            case "tuple[int]":
                field_type = tuple[int, ...]
            case "tuple[float]":
                field_type = tuple[float, ...]
            case "tuple[str]":
                field_type = tuple[str, ...]
            case "tuple[bool]":
                field_type = tuple[bool, ...]
            case _:
                raise ValueError(f"Unsupported type: {type}")
        schema_fields[name] = (field_type, Field(..., description=desc))

    return create_model("ArgsSchema", **schema_fields)


class AbstractPlanner:
    base_llm: BaseChatModel
    toolgen_chain: Runnable
    plangen_chain: Runnable
    max_retries: int = 3
    analyzer: Analyzer

    def __init__(
        self,
        base_llm: BaseChatModel,
        context: str = "",
        analyzer: Optional[Analyzer] = None,
        max_retries: int = 3,
    ):
        logger.info("Initializing AbstractPlanner")
        self.base_llm = base_llm
        self.context = context
        self.max_retries = max_retries

        # Create prompt templates
        toolgen_prompt = generate_abstract_tool_template()
        plangen_prompt = generate_abstract_plan_template()

        # Runnable for parsing and validating tools
        parse_tools_runnable = RunnableLambda(self._parse_tools_fn)

        # Generation chains
        self.toolgen_chain = toolgen_prompt | (
            self.base_llm.with_structured_output(TOOL_GENERATION_SCHEMA) | parse_tools_runnable
        ).with_retry(stop_after_attempt=self.max_retries)

        format_tools_runnable = RunnableLambda(lambda inputs: "\n".join([tool.as_json() for tool in inputs["tools"]]))  # type: ignore
        parse_plan_runnable = RunnableLambda(self._parse_plan_with_tools)

        self.plangen_chain = (
            RunnablePassthrough.assign(
                script=(RunnablePassthrough.assign(tools=format_tools_runnable) | plangen_prompt | self.base_llm)
            )
            | parse_plan_runnable
        ).with_retry(stop_after_attempt=self.max_retries)

        # Initialize analyzer
        self.analyzer = analyzer or Analyzer()

    def _parse_plan_with_tools(self, inputs: dict) -> AbstractPlan:
        script_output = inputs["script"]
        tools = inputs["tools"]
        parsed_script = parse_text_to_python(script_output)
        plan = AbstractPlan(script=parsed_script, abs_tools=tools)

        self.analyzer.analyze_post_abstract_plan(plan)

        return plan

    def _parse_tools_fn(self, output: dict) -> list[AbstractTool]:
        if not output or "apps" not in output:
            raise ValueError("Malformed output; missing 'apps' key.")
        abstract_tool_list = []
        for i, tool in enumerate(output["apps"]):
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
                logger.log_structured("info", f"ABSTRACT_TOOL_{i + 1}", f"name={tool['name']}")
                logger.log_structured("debug", f"ABSTRACT_TOOL_{i + 1}_FULL", str(tool))
            except Exception as e:
                raise ValueError(f"Failed to instantiate AbstractTool for {tool.get('name', 'unnamed')}: {e}")

        self.analyzer.analyze_post_abstract_tools(abstract_tool_list)

        return abstract_tool_list

    def generate_abstract_tools(self, query) -> list[AbstractTool]:
        logger.debug(f"Generating abstract tools for query: {query[:100]}...")
        logger.log_query(query)
        try:
            output: list[AbstractTool] = self.toolgen_chain.invoke({"query": query, "context": self.context})
            logger.info(f"Generated {len(output)} abstract tools")
            logger.debug(f"Abstract tools: {[app.name for app in output]}")
            return output
        except Exception as e:
            logger.error(f"Failed to generate abstract tools: {e}", exc_info=True)
            raise

    def generate_abstract_plan(self, query):
        logger.info("Starting abstract plan generation")

        # Generate abstract tools first
        abs_tools = self.generate_abstract_tools(query)

        logger.debug("Generating plan script using abstract tools")
        try:
            plan_obj: AbstractPlan = self.plangen_chain.invoke(
                {"query": query, "context": self.context, "tools": abs_tools}
            )
            logger.debug("Plan script generation completed")

            # Log the raw plan output using structured logging
            logger.log_raw_llm_output("abstract_plan", str(plan_obj.script))

        except Exception as e:
            logger.error(f"Failed to generate abstract plan script: {e}", exc_info=True)
            raise

        logger.info(f"Abstract plan generation completed with {len(abs_tools)} tools")
        logger.debug(f"Script length: {len(plan_obj.script)} characters")

        # Log the final abstract plan using enhanced logger
        logger.log_plan(plan_obj.script, abs_tools)

        return plan_obj
