import abc
import datetime
import uuid
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_benchmarks import (
    __version__,
    clone_public_dataset,
    registry,
)
from langchain_benchmarks.rate_limiting import RateLimiter, with_rate_limit
from langchain_benchmarks.schema import ToolUsageTask
from langchain_benchmarks.tool_usage.agents.adapters import apply_agent_executor_adapter
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langsmith.client import Client


class AgentFactory(abc.ABC):
    """Abstract class for agent factory"""

    @abc.abstractmethod
    def __call__(self) -> Runnable:
        """Create a new agent"""


class StandardAgentFactory(AgentFactory):
    """A standard agent factory.

    Use this factory with chat models that support the standard LangChain tool
    calling API where the chat model populates the tool_calls attribute on AIMessage.
    """

    def __init__(
        self,
        task: ToolUsageTask,
        model: BaseChatModel,
        prompt: ChatPromptTemplate,
        *,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """Create an agent factory for the given tool usage task.

        Args:
            task: The task to create an agent factory for
            model: chat model to use, must support tool usage
            prompt: This is a chat prompt at the moment.
                Must include an agent_scratchpad

                For example,

                ChatPromptTemplate.from_messages(
                    [
                        ("system", "{instructions}"),
                        ("human", "{input}"),
                        MessagesPlaceholder("agent_scratchpad"),
                    ]
                )
            rate_limiter: will be appended to the agent runnable
        """
        self.task = task
        self.model = model
        self.prompt = prompt
        self.rate_limiter = rate_limiter

    def __call__(self) -> Runnable:
        """Call the factory to create Runnable agent."""

        env = self.task.create_environment()

        if "instructions" in self.prompt.input_variables:
            finalized_prompt = self.prompt.partial(instructions=self.task.instructions)
        else:
            finalized_prompt = self.prompt

        agent = create_tool_calling_agent(self.model, env.tools, finalized_prompt)

        if self.rate_limiter:
            agent = with_rate_limit(agent, self.rate_limiter)

        executor = AgentExecutor(
            agent=agent,
            tools=env.tools,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        return apply_agent_executor_adapter(executor, state_reader=env.read_state).with_config(
            {"run_name": "Agent", "metadata": {"task": self.task.name}}
        )


load_dotenv()


tests = [
    ("gpt-4.1-2025-04-14", ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)),
    (
        "gpt-4o-2024-11-20",
        ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0),
    ),
]


# Create prompts for the agents
# Using two prompts because some chat models do not support SystemMessage.
without_system_message_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "{instructions}\n{input}",
        ),  # Populated from task.instructions automatically
        MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
    ]
)

with_system_message_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{instructions}"),
        ("human", "{input}"),  # Populated from task.instructions automatically
        MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
    ]
)

experiment_uuid = uuid.uuid4().hex[:4]

client = Client()  # Launch langsmith client for cloning datasets
today = datetime.date.today().isoformat()


for task in registry.tasks:
    if task.type != "ToolUsageTask":
        continue

    # This is a small test dataset that can be used to verify
    # that everything is set up correctly prior to running over
    # all results. We may remove it in the future.
    if task.name == "Multiverse Math (Tiny)":
        continue

    dataset_name = task.name + f" ({today})"
    clone_public_dataset(task.dataset_id, dataset_name=dataset_name)

    for model_name, model in tests:
        if model_name.startswith("gemini"):
            # google models don't use system prompt
            prompt = without_system_message_prompt
            rate_limiter = RateLimiter(requests_per_second=0.1)
        else:
            prompt = with_system_message_prompt
            rate_limiter = RateLimiter(requests_per_second=1)
        print()
        print(f"Benchmarking {task.name} with model: {model_name}")
        eval_config = task.get_eval_config()

        agent_factory = StandardAgentFactory(task, model, prompt, rate_limiter=rate_limiter)

        client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=agent_factory,
            evaluation=eval_config,
            verbose=False,
            project_name=f"{model_name}-{task.name}-{today}-{experiment_uuid}",
            concurrency_level=5,
            project_metadata={
                "model": model_name,
                "id": experiment_uuid,
                "task": task.name,
                "date": today,
                "langchain_benchmarks_version": __version__,
            },
        )
