import argparse
import logging

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ace.plan.abstract import AbstractPlanner
from ace.plan.concrete import InfoFlowPlanner
from ace.agent import AceAgent
from ace.tools.manager import ToolManager

load_dotenv()

logger = logging.getLogger(__name__)

ConcPlanner = InfoFlowPlanner


def main(args: argparse.Namespace):
    # Set up logging
    logging.basicConfig(level=logging.DEBUG if args.very_verbose else logging.INFO if args.verbose else logging.WARNING)

    # Load LLMs and tool manager
    base_llm = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        temperature=0.0,
    )
    embedding_model = OpenAIEmbeddings()
    tool_manager = ToolManager.from_manifest("tools/manifest.json")

    # TODO: clean up abstract Planner class
    # Initialize planners
    abstract_planner = AbstractPlanner()
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
