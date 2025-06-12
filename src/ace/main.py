import logging

import click
from dotenv import load_dotenv

from ace.llm.embeddings import EmbeddingsEnum
from ace.llm.models import ModelsEnum
from ace.setup import AppConfig, setup_app

load_dotenv()
logger = logging.getLogger(__name__)


@click.command()
@click.option("--llm-model", default="gpt-4o-mini-2024-07-18", show_default=True, help="LLM model to use")
@click.option("--embedding-model", default="text-embedding-3-small", show_default=True, help="Embedding model to use")
@click.option("--tool-manifest", default="tools/manifest.json", show_default=True, help="Path to tool manifest JSON")
@click.option("--temperature", default=0.0, type=float, show_default=True, help="LLM temperature")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--very-verbose", "-vv", is_flag=True, help="Enable very verbose logging")
def cli(llm_model, embedding_model, tool_manifest, temperature, verbose, very_verbose):
    """Run the ACE agent CLI."""
    # Set up logging
    if very_verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Map string to enum (fallback to value if not found)
    llm_enum = (
        ModelsEnum(llm_model) if llm_model in ModelsEnum._value2member_map_ else ModelsEnum.GPT_4O_MINI_2024_07_18
    )
    embedding_enum = (
        EmbeddingsEnum(embedding_model)
        if embedding_model in EmbeddingsEnum._value2member_map_
        else EmbeddingsEnum.OPENAI_3_SMALL
    )
    config = AppConfig(
        llm_model=llm_enum,
        embedding_model=embedding_enum,
        tool_manifest=tool_manifest,
        temperature=temperature,
    )
    agent = setup_app(config)

    click.echo("ACE agent CLI. Type your query and press Enter. Ctrl+C to exit.")
    while True:
        try:
            query = click.prompt("Query")
        except (EOFError, KeyboardInterrupt):
            click.echo("\nExiting.")
            break
        agent.run_query(query)
        logger.debug(f"Most recent output: {agent.output_log[-1]}")


if __name__ == "__main__":
    cli()
