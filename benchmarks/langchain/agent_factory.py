import requests
from ace.agent import AceAgent
from ace.execute import PlanOrchestrator
from langchain_core.agents import AgentAction
from langchain_core.runnables import RunnableLambda


class ACEAgentLangChain:
    def __init__(self, agent: AceAgent, task, service_url):
        print("Initializing ACEAgentLangChain...")
        self.agent = agent
        self.task = task
        self.service_url = service_url

    def invoke(self, inputs):
        # Reset environment before each run
        reset_response = requests.post(f"{self.service_url}/reset")
        if reset_response.status_code != 200:
            raise RuntimeError(f"Failed to reset environment: {reset_response.text}")

        # Health check
        health_response = requests.get(f"{self.service_url}/health")
        if health_response.status_code != 200:
            raise RuntimeError(f"Service health check failed: {health_response.text}")

        # Compose query
        q = inputs["question"]
        query = self.task.instructions + f" The given string is: '{q}'."
        query = (
            "You are a typewriter tool that can type letters one at a time. You have a single effectual tool that can type a letter. It takes the character to type as an argument and returns status code.\n\n"
            + query
        )

        # ACE pipeline
        abstract_plan = self.agent.abstract_planner.generate_abstract_plan(query)
        concrete_plan = self.agent.concrete_planner.implement_plan(plan=abstract_plan)
        with PlanOrchestrator(abstract_plan, concrete_plan) as orchestrator:
            orchestrator.launch()
            orchestrator.join()
            result = orchestrator.result

            # Try to get intermediate_steps and state if available
            intermediate_steps = None
            try:
                intermediate_steps = requests.get(f"{self.service_url}/history")
                if intermediate_steps.status_code == 200:
                    intermediate_steps = intermediate_steps.json().get("history", [])
                    intermediate_steps = [(AgentAction(**action), output) for action, output in intermediate_steps]
                else:
                    intermediate_steps = []
            except Exception as e:
                print(f"Error fetching intermediate steps: {e}")

            state = None
            try:
                env = requests.get(f"{self.service_url}/state")
                if env.status_code == 200:
                    state = env.json().get("state")
            except Exception as e:
                print(f"Error fetching state: {e}")

        return {
            "question": q,
            "output": result,
            "intermediate_steps": intermediate_steps,
            "state": state,
        }


class AgentFactory:
    def __init__(self, agent, task, service_url):
        self.agent = agent
        self.task = task
        self.service_url = service_url

    def __call__(self, *args):
        agent = ACEAgentLangChain(self.agent, self.task, self.service_url)
        return RunnableLambda(agent.invoke)
