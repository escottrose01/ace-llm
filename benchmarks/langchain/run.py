import argparse
import datetime
import uuid
from enum import Enum
from pathlib import Path

from ace.agent.ace_agent import AceAgent
from ace.llm.embeddings import EmbeddingsEnum
from ace.llm.models import ModelsEnum
from ace.plan.abstract import AbstractPlanner
from ace.plan.concrete import SimpleConcretePlanner
from ace.tools.manager import ToolManager
from dotenv import load_dotenv
from langchain_benchmarks import __version__ as langchain_version
from langchain_benchmarks import clone_public_dataset, registry
from langchain_benchmarks.schema import ToolUsageTask
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith.client import Client

from .ace_adapter import LangChainAceAdapter
from .agent_factory import AgentFactory


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


def main(args: argparse.Namespace):
    load_dotenv()

    # LangSmith client and experiment setup
    client = Client()
    today = datetime.date.today().isoformat()
    experiment_id = uuid.uuid4().hex[:]

    benchmark = Benchmark(args.benchmark_name)

    print(f"Running benchmark: {BENCHMARK_LONG_NAMES[benchmark]}")

    task: ToolUsageTask = registry[BENCHMARK_LONG_NAMES[benchmark]]  # type: ignore
    assert isinstance(task, ToolUsageTask), "Expected a ToolUsageTask instance"

    dataset_name = task.name + f" ({today})"
    clone_public_dataset(task.dataset_id, dataset_name=dataset_name)

    # ACE System setup
    abs_model = ModelsEnum("gpt-4o-2024-05-13")
    conc_model = ModelsEnum("gpt-4o-2024-05-13")
    embedding_model = EmbeddingsEnum("text-embedding-3-small")

    abs_llm = ChatOpenAI(model=abs_model.value, temperature=0.8)
    conc_llm = ChatOpenAI(model=conc_model.value, temperature=0.8)
    embedding = OpenAIEmbeddings(model=embedding_model.value)

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

    agent_factory = AgentFactory(agent, task, service_url)
    eval_config = task.get_eval_config()

    client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=agent_factory,
        evaluation=eval_config,
        verbose=False,
        project_name=f"{abs_model.value}-{conc_model.value}-{embedding_model.value}-{today}-{experiment_id}",
        concurrency_level=0,  # ACE binds a specific port, so cannot run concurrently
        project_metadata={
            "abs_model": abs_model.value,
            "conc_model": conc_model.value,
            "embedding_model": embedding_model.value,
            "id": experiment_id,
            "task": task.name,
            "date": today,
            "langchain_benchmarks_version": langchain_version,
        },
    )


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
