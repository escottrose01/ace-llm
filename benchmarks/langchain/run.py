import argparse
import datetime
import json
import logging
import uuid
from enum import Enum
from pathlib import Path

from ace.agent.ace_agent import AceAgent
from ace.llm.embeddings import EmbeddingsEnum
from ace.llm.models import ModelsEnum
from ace.logging_config import setup_logging
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
    Benchmark.RELATIONAL_DATA: "Tool Usage - Relational Data",
}

EXTRA_CONTEXT = {
    Benchmark.TYPEWRITER_1: "You are a case-insensitive typewriter tool that can type letters one at a time. You have a single tool that types a letter and returns string status. You must generate this tool to use it.",
    Benchmark.TYPEWRITER_26: "You are a case-insensitive typewriter tool that can type letters one at a time. You have 26 tools, each for a letter of the alphabet, and they return string status when a letter is typed. Generate only the tools you need to type the given string.",
    Benchmark.RELATIONAL_DATA: """\
You can access several tables in a relational database: User Data, Location Data, Food Data.

## Database Schema
1. Users
- id (integer): Primary key; unique user identifier
- name (string): User's name
- email (string): User's email address
- location (integer): Foreign key -> Locations.id
- favorite_color (string): User's preferred color
- favorite_foods (array<int>): List of food IDs -> Foods.id

Relationships:
- Users.location -> Locations.id
- Elements of Users.favorite_foods -> Foods.id

2. Locations
- id (integer): Primary key; unique location identifier
- city (string): City name
- current_time (datetime): Local time at the location (e.g. "2023-11-14 10:30 AM")
- current_weather (string): Weather description & temperature (e.g. "Partly Cloudy, 68°F")

3. Foods
- id (integer): Primary key; unique food identifier
- name (string): Food item name
- calories (integer): Caloric content per serving
- allergic_ingredients (array<string>): List of allergenic components

## Actions
You can use tools to perform basic database read operations.
Use primary keys to query information about specific records.
Do not use non-primary keys for lookups (e.g. city name).
You might first need to find the primary key of a record identified by a non-primary key.
Each action can only access a single field of a record at a time.
When looking record ID with string, you might get back multiple IDs. Not all will match. You should either use the first one or check each ID against the original query with a follow-up action.
Make sure the final output actually answers the original user question! Return what you display to the user.
Only generate / use the tools needed.

Example actions:
Find<RecordType>sBy<FieldName>(<field_value>: <field_type>) -> tuple[int]
  Example: FindUsersByName(name: str) -> tuple[int]
Get<RecordType><FieldName>(<record_id>: int) -> <field_type>
  Example: GetUserName(user_id: int) -> str
GetCurrentUserId() -> int
""",
}


def main(args: argparse.Namespace):
    load_dotenv()
    setup_logging(log_level="DEBUG", log_file="langchain_benchmarks.log")
    logging.getLogger("langchain").setLevel(logging.ERROR)
    logging.getLogger("langsmith").setLevel(logging.ERROR)

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
    abs_model = ModelsEnum(args.abs_model)
    conc_model = ModelsEnum(args.conc_model)
    embedding_model = EmbeddingsEnum(args.embedding_model)

    abs_llm = ChatOpenAI(model=abs_model.value, temperature=args.abs_temperature)
    conc_llm = ChatOpenAI(model=conc_model.value, temperature=args.conc_temperature)
    embedding = OpenAIEmbeddings(model=embedding_model.value)

    # Bridge between LangChain and ACE
    adapter = LangChainAceAdapter(task=task)
    adapter.start()
    service_url = f"http://localhost:{adapter.port}"
    print(f"Service started at {service_url}")

    tools = ToolManager(adapter.create_ace_tools())

    extra_context = EXTRA_CONTEXT.get(benchmark, "") if args.extra_context else ""
    abstract_planner = AbstractPlanner(base_llm=abs_llm, context=extra_context)
    concrete_planner = SimpleConcretePlanner(
        tool_manager=tools,
        base_llm=conc_llm,
        embedding_model=embedding,
        filter_threshold=args.embedding_threshold,
        context=extra_context,
    )
    agent = AceAgent(
        abstract_planner=abstract_planner,
        concrete_planner=concrete_planner,
    )

    agent_factory = AgentFactory(agent, task, service_url)
    eval_config = task.get_eval_config()

    results = client.run_on_dataset(
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

    print("Benchmark completed")

    # Save results to a file if specified
    # TODO: currently this will fail because of several nonserializable values
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Trace logs saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangChain benchmarks.")

    parser.add_argument(
        "--benchmark_name",
        type=str,
        default="typewriter1",
        choices=Benchmark._value2member_map_.keys(),
        help="Name of the benchmark to run.",
    )
    parser.add_argument(
        "--abs_model",
        type=str,
        default="gpt-4-1-nano-2025-04-14",
        choices=[m.value for m in ModelsEnum],
        help="Abstract model to use for the benchmark.",
    )
    parser.add_argument(
        "--conc_model",
        type=str,
        default="gpt-4-1-nano-2025-04-14",
        choices=[m.value for m in ModelsEnum],
        help="Concrete model to use for the benchmark.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-small",
        choices=[m.value for m in EmbeddingsEnum],
        help="Embedding model to use for the benchmark.",
    )
    parser.add_argument(
        "--abs_temperature",
        type=float,
        default=1.0,
        help="Temperature for the abstract LLM.",
    )
    parser.add_argument(
        "--conc_temperature",
        type=float,
        default=1.0,
        help="Temperature for the concrete LLM.",
    )
    parser.add_argument(
        "--embedding_threshold",
        type=float,
        default=0.0,
        help="Threshold for filtering embeddings in the concrete planner.",
    )
    parser.add_argument("--extra_context", action="store_true", help="Give extra task context to the agent.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory to save the benchmark results.")

    args = parser.parse_args()

    main(args)
