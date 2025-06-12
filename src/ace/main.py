import argparse
import logging

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from ace.agent import AceAgent
from ace.llm.base import create_llm
from ace.llm.models import ModelsEnum
from ace.plan.abstract import AbstractPlanner
from ace.plan.concrete import InfoFlowPlanner
from ace.tools.manager import ToolManager

load_dotenv()

logger = logging.getLogger(__name__)

ConcPlanner = InfoFlowPlanner


def main(args: argparse.Namespace):
    # Set up logging
    logging.basicConfig(level=logging.DEBUG if args.very_verbose else logging.INFO if args.verbose else logging.WARNING)

    # Load LLMs and tool manager
    base_llm = create_llm(ModelsEnum.GPT_4O_MINI_2024_07_18, temperature=0.0)  # GPT-4o Mini (July 2024)
    embedding_model = OpenAIEmbeddings()
    tool_manager = ToolManager.from_manifest("tools/manifest.json")

    # TODO: clean up abstract Planner class
    # Initialize planners
    abstract_planner = AbstractPlanner(base_llm=base_llm)
    concrete_planner = ConcPlanner(tool_manager=tool_manager, base_llm=base_llm, embedding_model=embedding_model)

    # Initialize agent
    agent = AceAgent(abstract_planner=abstract_planner, concrete_planner=concrete_planner)

    # Run queries
    while True:
        query = input("Query: ")
        agent.run_query(query)

        logger.debug(f"Most recent output: {agent.output_log[-1]}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args.add_argument("--very-verbose", "-vv", action="store_true", help="Enable very verbose logging")
    args = args.parse_args()

    main(args)
