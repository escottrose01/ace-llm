from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .models import MODEL_PROVIDERS, ModelsEnum


def create_llm(model: ModelsEnum, **kwargs: Any) -> BaseChatModel:
    """
    Factory: Instantiate and return a LangChain chat model for the given model enum, with backend-specific config.

    Args:
        model: ModelProviders enum value (e.g. ModelProviders.GPT_40_2024_07_18)
        **kwargs: Passed to the backend constructor (e.g. temperature, api_key, model_path, etc)

    Returns:
        LangChain BaseChatModel instance (e.g. ChatOpenAI, ChatAnthropic, etc)

    Raises:
        ValueError if the provider is unknown or required kwargs are missing.
    """
    provider = MODEL_PROVIDERS[model]
    if provider == "openai" or provider == "vllm":
        return ChatOpenAI(model=model.value, **kwargs)
    elif provider == "anthropic":
        return ChatAnthropic(model_name=model.value, **kwargs)
    elif provider == "meta":
        raise NotImplementedError("Meta/Llama provider support not yet implemented. Please add your backend.")
    else:
        raise ValueError(f"Unknown or unsupported provider: {provider}")
