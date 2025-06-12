import ast
import itertools
from abc import ABC, abstractmethod

import astor
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import OpenAIEmbeddings

from ..prompts.concrete_templates import generate_tool_compatibility_json_template
from ..schema.abstract import AbstractPlan, AbstractTool
from ..schema.concrete import ConcreteToolBase
from ..schema.infoflow import MemoryModel
from ..schema.lattice import SubsetLattice
from ..security.infoflow import FlowAnalyzer
from ..tools.manager import ToolManager


class SchemaAdaptedTool(ConcreteToolBase):
    wrapped_tool: ConcreteToolBase
    input_mapping_source: str
    output_mapping_source: str

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

        field_names = list(self.args_schema.model_fields.keys())

        input_mapping = ast.parse(self.input_mapping_source).body[0]
        output_mapping = ast.parse(self.output_mapping_source).body[0]

        # Begin building the main function
        main_body = []
        main_body.append(underlying_main)
        main_body.append(input_mapping)
        main_body.append(output_mapping)

        # Map input fields to inner input
        input_args = [ast.Name(id=field, ctx=ast.Load()) for field in field_names]
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
    def implement_plan(self, plan: AbstractPlan) -> dict[str, ConcreteToolBase]:
        pass

    @property
    @abstractmethod
    def results():
        pass


class SimpleConcretePlanner(ConcretePlannerBase):
    tool_manager: ToolManager
    base_llm: BaseChatModel
    embeddings: OpenAIEmbeddings
    faiss_store: FAISS
    filter_threshold: float
    compat_chain: Runnable
    _results: list[dict]

    def __init__(
        self,
        tool_manager: ToolManager,
        base_llm: BaseChatModel,
        embedding_model: OpenAIEmbeddings,
        filter_threshold: float = 0.8,
    ):
        self.tool_manager = tool_manager
        self.base_llm = base_llm
        self.embeddings = embedding_model
        self.filter_threshold = filter_threshold
        self._results = []

        compat_prompt = generate_tool_compatibility_json_template()
        json_parser = JsonOutputParser()
        self.compat_chain = compat_prompt | self.base_llm | json_parser

        self._generate_tool_embeddings()

    def generate_matches_for_tool(self, abstract_tool: AbstractTool) -> list[ConcreteToolBase]:
        # TODO: Improve embedding-based filtering
        query = f"{abstract_tool.name}: {abstract_tool.description}"
        hits = self.faiss_store.search(
            query,
            search_type="similarity_score_threshold",
            threshold=self.filter_threshold,
            k=100,  # should be no limit, but seems API forces one
        )
        hit_names = [hit.metadata["name"] for hit in hits]

        # Conform concrete tools to abstract tool schema
        abs_tool_json = abstract_tool.as_json()
        tools = []
        self._results = []
        for tool_name in hit_names:
            concrete_tool = self.tool_manager.get_by_name(tool_name)
            concrete_tool_json = concrete_tool.as_json()
            result = self.compat_chain.invoke({"abstract_tool": abs_tool_json, "concrete_tool": concrete_tool_json})

            self._results.append(result)

            if result["status"] == "success":
                input_mapping = result.get("input_mapping")
                output_mapping = result.get("output_mapping")

                if not input_mapping or not output_mapping:
                    continue

                # TODO: pydantic kind of requires constructor to use all fields
                # but we should see if we can make this cleaner with validators
                tools.append(
                    SchemaAdaptedTool(
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
                )

        return tools

    @property
    def results(self) -> list[dict]:
        return self._results

    def implement_plan(self, plan: AbstractPlan) -> dict[str, ConcreteToolBase]:
        # Naive: just pick the first match for each tool
        concrete_tools = []
        for abstract_tool in plan.abs_tools:
            matches = self.generate_matches_for_tool(abstract_tool)
            concrete_tools.append(matches[0])

        return {
            abstract_tool.name: concrete_tool for abstract_tool, concrete_tool in zip(plan.abs_tools, concrete_tools)
        }

    def _generate_tool_embeddings(self):
        tool_documents = [
            Document(page_content=tool.description, metadata={"name": tool.name}) for tool in self.tool_manager.tools
        ]

        self.faiss_store = FAISS.from_documents(tool_documents, self.embeddings)


class InfoFlowPlanner(SimpleConcretePlanner):
    def implement_plan(self, plan: AbstractPlan) -> dict[str, ConcreteToolBase]:
        matches = [self.generate_matches_for_tool(abstract_tool) for abstract_tool in plan.abs_tools]

        for proposal in itertools.product(*matches):
            memory = MemoryModel(
                lattice_type=SubsetLattice,
                dynamic_vars={},
                static_vars={
                    abstract_tool.name: SubsetLattice(concrete_tool.clearances)
                    for abstract_tool, concrete_tool in zip(plan.abs_tools, proposal)
                },
            )

            code = plan.compile_for_analysis()

            # Run information flow analysis
            analyzer = FlowAnalyzer(memory)
            analyzer.analyze_ast(code)
            if analyzer.valid:
                return {
                    abstract_tool.name: concrete_tool for abstract_tool, concrete_tool in zip(plan.abs_tools, proposal)
                }

        return None
