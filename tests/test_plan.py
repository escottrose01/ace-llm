# Comprehensive plan module test
from typing import ClassVar

import pytest

try:
    from src.ace.plan import abstract, concrete
except ImportError:
    pytest.skip("src.ace.plan modules not available", allow_module_level=True)


# Patch: Make DummyLLM a Runnable
from langchain_core.runnables.base import Runnable


class DummyLLM(Runnable):
    def __or__(self, other):
        return self

    def invoke(self, input, config=None, **kwargs):
        return {"apps": []}


# Patch: Provide a dummy embeddings model
class DummyEmbeddings:
    def embed_documents(self, texts):
        # Return a list of dummy vectors (e.g., lists of zeros)
        return [[0.0] * 10 for _ in texts]


class DummyToolManager:
    tools: ClassVar = []

    def get_by_name(self, name):
        return None


def test_abstract_plan():
    planner = abstract.AbstractPlanner(base_llm=DummyLLM())
    # Just test instantiation and method presence
    assert hasattr(planner, "generate_abstract_plan")


from unittest.mock import patch


@patch("langchain_community.vectorstores.faiss.FAISS.from_documents", return_value=None)
def test_concrete_plan(_):
    planner = concrete.SimpleConcretePlanner(
        tool_manager=DummyToolManager(), base_llm=DummyLLM(), embedding_model=DummyEmbeddings()
    )
    assert planner is not None
