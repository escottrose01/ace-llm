import argparse
import json
from enum import Enum
from pathlib import Path

import requests
from ace.agent.ace_agent import AceAgent
from ace.execute import PlanOrchestrator
from ace.llm.embeddings import EmbeddingsEnum
from ace.llm.models import ModelsEnum
from ace.plan.abstract import AbstractPlanner
from ace.plan.concrete import SimpleConcretePlanner
from ace.tools.manager import ToolManager
from dotenv import load_dotenv
from langchain_benchmarks import registry
from langchain_benchmarks.schema import ToolUsageTask
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .ace_adapter import LangChainAceAdapter


class Benchmark(Enum):
    TYPEWRITER_1 = "typewriter1"
    TYPEWRITER_26 = "typewriter26"
    MULTIVERSE_MATH = "multiverse_math"
    RELATIONAL_DATA = "relational_data"


# Mapping from Enum to long-form benchmark keys
BENCHMARK_LONG_NAMES = {
    Benchmark.TYPEWRITER_1: "Tool Usage - Typewriter (1 tool)",
    Benchmark.TYPEWRITER_26: "Tool Usage - Typewriter (26 tools)",
    Benchmark.MULTIVERSE_MATH: "Multiverse Math",
    Benchmark.RELATIONAL_DATA: "Relational Data",
}

ROOT_DIR = Path("benchmarks") / "langchain" / "data"
BENCHMARK_DATA_PATHS = {
    Benchmark.TYPEWRITER_1: ROOT_DIR / "single_tools_cases.json",
    Benchmark.TYPEWRITER_26: ROOT_DIR / "multi_tools_cases.json",
    Benchmark.MULTIVERSE_MATH: ROOT_DIR / "multiverse_math.json",
    Benchmark.RELATIONAL_DATA: ROOT_DIR / "relational_data_cases.json",
}

EXTRA_CONTEXT = {
    Benchmark.TYPEWRITER_1: "You are a typewriter tool that can type letters one at a time. You have a single tool that can type a letter.",
    Benchmark.TYPEWRITER_26: "You are a typewriter tool that can type letters one at a time. You have 26 tools, each capable of typing a single letter from the standard alphabet.",
}


def run_case(agent: AceAgent, task: ToolUsageTask, case: dict):
    # Extract query and answer
    q = case["inputs"]["question"]
    case["outputs"]["reference"]
    query = (
        (
            "Repeat the given string using the provided tools. "
            "Do not write anything else or provide any explanations. "
            "For example, if the string is 'abc', you must print the letters "
            "'a', 'b', and 'c' one at a time and in that order. "
            "Please invoke the functions without any arguments."
            "The given string is: '"
        )
        + q
        + "'."
    )

    query = (
        "You are a typewriter tool that can type letters one at a time. You have a single tool that can type a letter. It takes the character to type as an argument.\n\n"
        + query
    )
    query += "BTW when you match tools please be lenient about the return type."

    # Phase 1: Abstract Planning
    abstract_plan = agent.abstract_planner.generate_abstract_plan(query)
    print(f"Abstract tools: {[f'{tool.name} - {tool.description}' for tool in abstract_plan.abs_tools]}")
    print(f"Abstract plan: {abstract_plan.script}")

    # Phase 2: Concrete Planning
    concrete_plan = agent.concrete_planner.implement_plan(plan=abstract_plan)
    print("Concrete plan:", concrete_plan)

    # Phase 3: Execution
    with PlanOrchestrator(abstract_plan, concrete_plan) as orchestrator:
        orchestrator.launch()
        orchestrator.join()

        result = orchestrator.result
        print(f"Result: {result}")

    # Evaluation
    eval_config = task.get_eval_config()
    print(task.eval_params)
    print(eval_config)
    for eval in eval_config:
        print(eval)


def main(args: argparse.Namespace):
    load_dotenv()

    # System setup
    abs_model = ModelsEnum("gpt-4o-2024-05-13")
    conc_model = ModelsEnum("gpt-4o-2024-05-13")
    embedding_model = EmbeddingsEnum("text-embedding-3-small")

    abs_llm = ChatOpenAI(model=abs_model.value, temperature=0.8)
    conc_llm = ChatOpenAI(model=conc_model.value, temperature=0.8)
    embedding = OpenAIEmbeddings(model=embedding_model.value)

    benchmark = Benchmark(args.benchmark_name)

    print(f"Running benchmark: {BENCHMARK_LONG_NAMES[benchmark]}")

    task: ToolUsageTask = registry[BENCHMARK_LONG_NAMES[benchmark]]  # type: ignore
    assert isinstance(task, ToolUsageTask), "Expected a ToolUsageTask instance"

    # Bridge between LangChain and ACE
    adapter = LangChainAceAdapter(task=task)
    adapter.start()
    service_url = f"http://localhost:{adapter.port}"
    print(f"Service started at {service_url}")

    tools = ToolManager(adapter.create_ace_tools())

    abstract_planner = AbstractPlanner(base_llm=abs_llm)
    concrete_planner = SimpleConcretePlanner(
        tool_manager=tools,
        base_llm=conc_llm,
        embedding_model=embedding,
        filter_threshold=0.9,
    )
    agent = AceAgent(
        abstract_planner=abstract_planner,
        concrete_planner=concrete_planner,
    )

    with open(BENCHMARK_DATA_PATHS[benchmark]) as f:
        cases = json.load(f)

    results = []
    for case in cases:
        # Reset environment
        requests.post(f"{service_url}/reset")

        # Verify the health of the service
        health_response = requests.get(f"{service_url}/health")
        if health_response.status_code != 200:
            raise RuntimeError(f"Service health check failed: {health_response.text}")

        # Run case
        result = run_case(agent, task, case)
        results.append(result)

        # Collect results and check correctness
        env = requests.get(f"{service_url}/state")
        print(f"Environment state: {env.json()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangChain benchmarks.")

    parser.add_argument(
        "--benchmark_name",
        type=str,
        default="typewriter1",
        choices=Benchmark.__members__.keys(),
        help="Name of the benchmark to run.",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the benchmark results.")

    args = parser.parse_args()

    main(args)
