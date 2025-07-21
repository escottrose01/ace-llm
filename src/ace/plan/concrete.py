import ast
import itertools
from abc import ABC, abstractmethod
from typing import Optional

import astor
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from pydantic import model_validator

from ..analysis import Analyzer
from ..logging_config import get_logger
from ..prompts.concrete_templates import generate_tool_compatibility_json_template
from ..schema.abstract import AbstractPlan, AbstractTool, AnnotationTransformer
from ..schema.concrete import ConcreteToolBase
from ..schema.infoflow import MemoryModel
from ..schema.lattice import SubsetLattice
from ..security.infoflow import FlowAnalyzer
from ..tools.manager import ToolManager

logger = get_logger(__name__)


class SchemaAdaptedTool(ConcreteToolBase):
    wrapped_tool: ConcreteToolBase
    input_mapping_source: str
    output_mapping_source: str

    @model_validator(mode="after")
    def validate_mappings(self) -> "SchemaAdaptedTool":
        if not self.input_mapping_source:
            raise ValueError("input_mapping_source must be defined")
        if not self.output_mapping_source:
            raise ValueError("output_mapping_source must be defined")

        # Validate that the mappings are valid Python code
        try:
            ast.parse(self.input_mapping_source)
            ast.parse(self.output_mapping_source)
        except SyntaxError as e:
            raise ValueError(f"Invalid mapping source code: {e}")

        # TODO: should verify other things, like that annotations correspond to tool schemas

        return self

    def generate_source(self) -> str:
        underlying_ast = self.wrapped_tool.compile()

        # Extract and rename underlying main function
        for node in underlying_ast.body:
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                underlying_main = node
                underlying_main.name = "_tool"
                break
        else:
            raise ValueError("Tool source must contain a main() function.")

        if not self.args_schema:
            raise ValueError("Tool must have args_schema defined")

        field_names = list(self.args_schema.model_fields.keys())

        # Parse and transform the input and output mapping sources
        input_mapping = ast.parse(self.input_mapping_source).body[0]
        input_mapping = AnnotationTransformer().visit(input_mapping)
        input_mapping = ast.fix_missing_locations(input_mapping)

        output_mapping = ast.parse(self.output_mapping_source).body[0]
        output_mapping = AnnotationTransformer().visit(output_mapping)
        output_mapping = ast.fix_missing_locations(output_mapping)

        # Begin building the main function
        main_body = []
        main_body.append(underlying_main)
        main_body.append(input_mapping)
        main_body.append(output_mapping)

        # Map input fields to inner input
        input_args: list[ast.expr] = [ast.Name(id=field, ctx=ast.Load()) for field in field_names]
        assign_inner_input = ast.Assign(
            targets=[ast.Name(id="inner_input", ctx=ast.Store())],
            value=ast.Call(func=ast.Name(id="input_mapping", ctx=ast.Load()), args=input_args, keywords=[]),
        )
        main_body.append(assign_inner_input)

        # Call inner tool
        assign_inner_output = ast.Assign(
            targets=[ast.Name(id="inner_output", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="_tool", ctx=ast.Load()),
                args=[],
                keywords=[ast.keyword(arg=None, value=ast.Name(id="inner_input", ctx=ast.Load()))],
            ),
        )
        main_body.append(assign_inner_output)

        # Map inner output to output fields
        assign_outer_output = ast.Assign(
            targets=[ast.Name(id="outer_output", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="output_mapping", ctx=ast.Load()),
                args=[ast.Name(id="inner_output", ctx=ast.Load())],
                keywords=[],
            ),
        )
        main_body.append(assign_outer_output)

        # Return output
        return_stmt = ast.Return(value=ast.Name(id="outer_output", ctx=ast.Load()))
        main_body.append(return_stmt)

        # Construct main function from body
        params = [ast.arg(arg=field, annotation=None) for field in field_names]
        main_func = ast.FunctionDef(
            name="main",
            args=ast.arguments(
                posonlyargs=[], args=params, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
            ),
            body=main_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
            type_params=[],
        )

        # Create a new module containing the adapted tool
        module = ast.Module(body=[main_func], type_ignores=[])
        final_source = astor.to_source(module)
        return final_source


class ConcretePlannerBase(ABC):
    def generate_all_feasible_matches(self, abstract_tools: list[AbstractTool]) -> list[list[ConcreteToolBase]]:
        feasible_matches = [self.generate_matches_for_tool(abstract_tool) for abstract_tool in abstract_tools]
        return feasible_matches

    @abstractmethod
    def generate_matches_for_tool(self, abstract_tool: AbstractTool) -> list[ConcreteToolBase]:
        pass

    @abstractmethod
    def implement_plan(self, plan: AbstractPlan) -> Optional[dict[str, ConcreteToolBase]]:
        pass


class SimpleConcretePlanner(ConcretePlannerBase):
    tool_manager: ToolManager
    base_llm: BaseChatModel
    embeddings: OpenAIEmbeddings
    faiss_store: FAISS
    filter_threshold: float
    compat_chain: Runnable
    max_retries: int = 3
    analyzer: Analyzer

    def __init__(
        self,
        tool_manager: ToolManager,
        base_llm: BaseChatModel,
        embedding_model: OpenAIEmbeddings,
        context: str = "",
        filter_threshold: float = 0.8,
        max_retries: int = 3,
    ):
        logger.info("Initializing SimpleConcretePlanner")
        self.tool_manager = tool_manager
        self.base_llm = base_llm
        self.embeddings = embedding_model
        self.filter_threshold = filter_threshold
        self.context = context

        logger.debug(f"Filter threshold: {filter_threshold}")
        logger.debug(f"Available tools: {len(tool_manager.tools)}")

        # Create compatibility mapping chain
        compat_prompt = generate_tool_compatibility_json_template()
        json_parser = JsonOutputParser()
        parse_mapping_runnable = RunnableLambda(self._parse_mapping_fn)
        format_abs_runnable = RunnableLambda(lambda inputs: inputs["abstract_tool"].as_json())
        format_conc_runnable = RunnableLambda(lambda inputs: inputs["concrete_tool"].as_json())

        self.compat_chain = (
            RunnablePassthrough.assign(
                mapping=(
                    RunnablePassthrough.assign(abstract_tool=format_abs_runnable)
                    | RunnablePassthrough.assign(concrete_tool=format_conc_runnable)
                    | compat_prompt
                    | self.base_llm
                    | json_parser
                )
            )
            | RunnablePassthrough.assign(result=parse_mapping_runnable)
        ).with_retry(stop_after_attempt=max_retries)

        self._generate_tool_embeddings()
        logger.debug("SimpleConcretePlanner initialization completed")

    def _parse_mapping_fn(self, inputs: dict) -> Optional[SchemaAdaptedTool]:
        if not inputs or ("abstract_tool" not in inputs or "concrete_tool" not in inputs or "mapping" not in inputs):
            raise ValueError("Invalid input format for parse_mapping_fn")

        abstract_tool: AbstractTool = inputs["abstract_tool"]
        concrete_tool: ConcreteToolBase = inputs["concrete_tool"]
        mapping_result: dict = inputs["mapping"]

        if not mapping_result:
            raise ValueError("Mapping result must not be empty")
        if "status" not in mapping_result:
            raise ValueError("Mapping result must contain 'status' key")

        result = None
        if mapping_result["status"] == "failure":
            pass
        elif mapping_result["status"] == "success":
            input_mapping: str = inputs["mapping"].get("input_mapping")
            output_mapping: str = inputs["mapping"].get("output_mapping")

            result = SchemaAdaptedTool(
                name=concrete_tool.name,
                provider=concrete_tool.provider,
                description=concrete_tool.description,
                clearances=concrete_tool.clearances,
                permissions=concrete_tool.permissions,
                args_schema=abstract_tool.args_schema,
                output_schema=abstract_tool.output_schema,
                wrapped_tool=concrete_tool,
                input_mapping_source=input_mapping,
                output_mapping_source=output_mapping,
            )
        else:
            raise ValueError(f"Invalid mapping status: {mapping_result['status']}")

        return result

    def generate_all_feasible_matches(self, abstract_tools: list[AbstractTool]) -> list[list[ConcreteToolBase]]:
        batch = []
        for abstract_tool in abstract_tools:
            query = f"{abstract_tool.name}: {abstract_tool.description}"
            try:
                hits = self.faiss_store.search(
                    query,
                    search_type="similarity_score_threshold",
                    threshold=self.filter_threshold,
                    k=100,  # should be no limit, but seems API forces one
                )
                hit_names = [hit.metadata["name"] for hit in hits]
                logger.info(f"Found {len(hit_names)} potential matches: {hit_names}")
            except Exception as e:
                logger.error(f"Embedding search failed: {e}", exc_info=True)
                raise

            # Conform concrete tools to abstract tool schema
            for i, tool_name in enumerate(hit_names):
                logger.debug(f"Evaluating compatibility {i + 1}/{len(hit_names)}: {tool_name}")

                concrete_tool = self.tool_manager.get_by_name(tool_name)
                if concrete_tool is None:
                    logger.warning(f"Tool {tool_name} not found in tool manager")
                    continue

                batch.append(
                    {
                        "context": self.context,
                        "abstract_tool": abstract_tool,
                        "concrete_tool": concrete_tool,
                    }
                )

        logger.info(f"Generated {len(batch)} compatibility checks for {len(abstract_tools)} abstract tools")

        # Run batch compatibility checks
        matches = self.compat_chain.batch(batch)
        mappings: list[list[ConcreteToolBase]] = [[] for _ in range(len(abstract_tools))]
        abstract_index: dict[str, int] = {tool.name: i for i, tool in enumerate(abstract_tools)}

        for m in matches:
            abstract_tool = m["abstract_tool"]
            if m["result"] is not None:
                adapted_tool = m["result"]
                mappings[abstract_index[abstract_tool.name]].append(adapted_tool)

        return mappings

    def generate_matches_for_tool(self, abstract_tool: AbstractTool) -> list[ConcreteToolBase]:
        logger.debug(f"Finding concrete tool matches for abstract tool: {abstract_tool.name}")
        logger.debug(f"Abstract tool description: {abstract_tool.description}")

        # TODO: Improve embedding-based filtering
        query = f"{abstract_tool.name}: {abstract_tool.description}"
        logger.debug(f"Embedding search query: {query}")

        try:
            hits = self.faiss_store.search(
                query,
                search_type="similarity_score_threshold",
                threshold=self.filter_threshold,
                k=100,  # should be no limit, but seems API forces one
            )
            hit_names = [hit.metadata["name"] for hit in hits]
            logger.info(f"Found {len(hit_names)} potential matches: {hit_names}")
        except Exception as e:
            logger.error(f"Embedding search failed: {e}", exc_info=True)
            raise

        # Conform concrete tools to abstract tool schema
        batch = []
        for tool_name in hit_names:
            concrete_tool = self.tool_manager.get_by_name(tool_name)
            if concrete_tool is None:
                logger.warning(f"Tool {tool_name} not found in tool manager")
                continue

            batch.append(
                {
                    "context": self.context,
                    "abstract_tool": abstract_tool,
                    "concrete_tool": concrete_tool,
                }
            )

        logger.info(f"Generated {len(batch)} compatibility checks for abstract tool: {abstract_tool.name}")

        # Run batch compatibility checks
        matches = self.compat_chain.batch(batch)
        tools: list[ConcreteToolBase] = []
        for m in matches:
            if m["result"] is not None:
                adapted_tool: SchemaAdaptedTool = m["result"]
                tools.append(adapted_tool)

        # Log summary using structured logging
        tool_list = [f"{tool.name} ({tool.provider})" for tool in tools]
        logger.log_structured(
            "info",
            f"TOOL_MATCH_SUMMARY_{abstract_tool.name}",
            f"found={len(tools)} candidates={len(hit_names)} tools={tool_list}",
        )

        return tools

    def implement_plan(self, plan: AbstractPlan) -> Optional[dict[str, ConcreteToolBase]]:
        concrete_plan = self._implement_plan(plan)

        # Validate concrete plan
        if concrete_plan is not None:
            self.analyzer.analyze_post_concrete_plan(plan, concrete_plan)

        return concrete_plan

    def _implement_plan(self, plan: AbstractPlan) -> dict[str, ConcreteToolBase] | None:
        logger.info(f"Implementing plan with {len(plan.abs_tools)} abstract tools")
        logger.debug(f"Abstract tools: {[tool.name for tool in plan.abs_tools]}")

        all_matches = self.generate_all_feasible_matches(plan.abs_tools)

        # Naive: just pick the first match for each tool
        tool_mapping = {}
        for i, abstract_tool in enumerate(plan.abs_tools):
            logger.debug(f"Finding implementation for tool {i + 1}/{len(plan.abs_tools)}: {abstract_tool.name}")
            matches = all_matches[i]
            if not matches:
                logger.info(f"No matches found for abstract tool: {abstract_tool.name}")
                return None
            selected_tool = matches[0]
            tool_mapping[abstract_tool.name] = selected_tool
            logger.info(f"Selected concrete tool for {abstract_tool.name}: {selected_tool.name}")

        logger.info("Plan implementation completed successfully")
        logger.debug(f"Tool mapping: {list(tool_mapping.keys())} -> {[tool.name for tool in tool_mapping.values()]}")
        return tool_mapping

    def _generate_tool_embeddings(self):
        logger.debug("Generating tool embeddings for similarity search")
        try:
            tool_documents = [
                Document(page_content=tool.description or "", metadata={"name": tool.name})
                for tool in self.tool_manager.tools
            ]
            logger.debug(f"Created {len(tool_documents)} tool documents")

            self.faiss_store = FAISS.from_documents(tool_documents, self.embeddings)
            logger.info(f"Generated embeddings for {len(tool_documents)} tools")
        except Exception as e:
            logger.error(f"Failed to generate tool embeddings: {e}", exc_info=True)
            raise


class InfoFlowPlanner(SimpleConcretePlanner):
    def _implement_plan(self, plan: AbstractPlan) -> dict[str, ConcreteToolBase] | None:
        logger.info(f"Starting information flow analysis for plan with {len(plan.abs_tools)} tools")

        # Generate all possible matches for each abstract tool
        matches = self.generate_all_feasible_matches(plan.abs_tools)

        total_combinations = 1
        for match_list in matches:
            total_combinations *= len(match_list)
        logger.info(f"Evaluating {total_combinations} possible tool combinations")

        if total_combinations == 0:
            logger.info("No possible combinations - at least one abstract tool has no matches")
            return None

        # Try each combination until we find one that satisfies information flow constraints
        for i, proposal in enumerate(itertools.product(*matches)):
            logger.debug(f"Evaluating combination {i + 1}/{total_combinations}")
            logger.debug(f"Proposal: {[tool.name for tool in proposal]}")

            memory = MemoryModel(
                lattice_type=SubsetLattice,
                dynamic_vars={},
                static_vars={
                    abstract_tool.name: SubsetLattice(concrete_tool.clearances)
                    for abstract_tool, concrete_tool in zip(plan.abs_tools, proposal)
                },
            )

            code = plan.compile_for_analysis()
            logger.debug("Running information flow analysis on proposal")

            # Run information flow analysis
            analyzer = FlowAnalyzer(memory)
            analyzer.analyze_ast(code)

            if analyzer.valid:
                logger.info(f"Found valid combination after {i + 1} attempts")
                logger.debug(f"Valid tools: {[tool.name for tool in proposal]}")
                return {
                    abstract_tool.name: concrete_tool for abstract_tool, concrete_tool in zip(plan.abs_tools, proposal)
                }
            else:
                logger.debug(f"Combination {i + 1} failed information flow analysis")

        # If no valid proposal found, return None
        logger.error(f"No valid tool combination found after evaluating {total_combinations} possibilities")
        return None
