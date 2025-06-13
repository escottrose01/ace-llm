import click
from dotenv import load_dotenv

from ace.llm.embeddings import EmbeddingsEnum
from ace.llm.models import ModelsEnum
from ace.logging_config import get_logger, setup_logging
from ace.setup import AppConfig, setup_app

load_dotenv()


@click.command()
@click.option("--llm-model", default="gpt-4o-mini-2024-07-18", show_default=True, help="LLM model to use")
@click.option("--embedding-model", default="text-embedding-3-small", show_default=True, help="Embedding model to use")
@click.option("--tool-manifest", default="tools/manifest.json", show_default=True, help="Path to tool manifest JSON")
@click.option("--temperature", default=0.0, type=float, show_default=True, help="LLM temperature")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--very-verbose", "-vv", is_flag=True, help="Enable very verbose logging")
def cli(llm_model, embedding_model, tool_manifest, temperature, verbose, very_verbose):
    """Run the ACE agent CLI."""
    # Set up comprehensive logging
    if very_verbose:
        log_level = "DEBUG"
    elif verbose:
        log_level = "INFO"
    else:
        log_level = "WARNING"

    # Configure logging to write to ace.log and optionally to console
    setup_logging(log_level=log_level, log_file="ace.log", console_output=verbose or very_verbose)

    logger = get_logger(__name__)
    logger.info(f"Starting ACE CLI with model={llm_model}, temp={temperature}, manifest={tool_manifest}")

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

    logger.info("ACE agent initialized successfully")
    logger.info(f"Agent configuration: LLM={config.llm_model.value}, Embedding={config.embedding_model.value}")

    click.echo("ACE agent CLI. Type your query and press Enter. Ctrl+C to exit.")
    while True:
        try:
            query = click.prompt("Query")
        except (EOFError, KeyboardInterrupt):
            click.echo("\nExiting.")
            logger.info("CLI session ended by user")
            break

        logger.info(f"Processing query: {query}")
        try:
            result = agent.run_query(query)
            logger.info("Query completed successfully")
            logger.debug(f"Query result type: {type(result).__name__}")
            if hasattr(agent, "output_log") and agent.output_log:
                logger.debug(f"Most recent output: {agent.output_log[-1]}")
        except Exception as e:
            logger.error(f"Query failed with error: {e}", exc_info=True)
            click.echo(f"Error: {e}")


if __name__ == "__main__":
    cli()
