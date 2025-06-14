from ace.agent import AceAgent
from ace.llm.base import create_llm
from ace.llm.embeddings import EmbeddingsEnum, create_embedding
from ace.llm.models import ModelsEnum
from ace.plan.abstract import AbstractPlanner
from ace.plan.concrete import InfoFlowPlanner
from ace.schema.events import AgentEventHandler
from ace.schema.handler_registry import HandlerRegistry
from ace.tools.manager import ToolManager


class AppConfig:
    def __init__(
        self,
        llm_model=ModelsEnum.GPT_4O_MINI_2024_07_18,
        embedding_model=EmbeddingsEnum.OPENAI_3_SMALL,
        tool_manifest="tools/manifest.json",
        temperature=0.0,
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.tool_manifest = tool_manifest
        self.temperature = temperature


def setup_app(config: AppConfig, event_registry: HandlerRegistry | AgentEventHandler | None = None) -> AceAgent:
    base_llm = create_llm(config.llm_model, temperature=config.temperature)
    embedding_model = create_embedding(config.embedding_model)
    tool_manager = ToolManager.from_manifest(config.tool_manifest)
    abstract_planner = AbstractPlanner(base_llm=base_llm)
    concrete_planner = InfoFlowPlanner(
        tool_manager=tool_manager,
        base_llm=base_llm,
        embedding_model=embedding_model,
    )
    return AceAgent(abstract_planner=abstract_planner, concrete_planner=concrete_planner, event_registry=event_registry)
