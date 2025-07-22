import time
from enum import Enum

import requests
from ace.agent import AceAgent
from ace.execute import PlanOrchestrator
from langchain_core.agents import AgentAction
from langchain_core.runnables import RunnableLambda


class Status(Enum):
    OK = "Success"
    ABSTRACT_PLANNING_FAILURE = "Abstract planning failed"
    ABSTRACT_PLANNING_ERROR = "Abstract planning failed"
    CONCRETE_PLANNING_FAILURE = "No feasible concrete plan"
    CONCRETE_PLANNING_ERROR = "Concrete planning error"
    EXECUTION_ERROR = "Execution error"


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
        instructions = f"{self.task.instructions}\n\n" if self.task.instructions else ""
        query = instructions + f"The given string is: '{q}'."

        # ACE pipeline
        trace = dict()

        # Abstract planning
        abstract_plan = None
        try:
            start_time = time.perf_counter()
            abstract_plan = self.agent.abstract_planner.generate_abstract_plan(query)
            end_time = time.perf_counter()
            trace["Abstract Plan"] = abstract_plan.script
            trace["Abstract Tools"] = [t.as_dict() for t in abstract_plan.abs_tools]
            trace["Plan Time"] = end_time - start_time
        except Exception:
            # TODO: implement more nuanced custom error types
            trace["Status"] = Status.ABSTRACT_PLANNING_ERROR.name
            return {
                "question": q,
                "output": "",
                "intermediate_steps": self.get_steps(),
                "state": self.get_state(),
                "trace": trace,
            }
        finally:
            if not abstract_plan:
                trace["Status"] = Status.ABSTRACT_PLANNING_FAILURE.name
                return {
                    "question": q,
                    "output": "",
                    "intermediate_steps": self.get_steps(),
                    "state": self.get_state(),
                    "trace": trace,
                }

        # Concrete planning
        try:
            start_time = time.perf_counter()
            concrete_plan = self.agent.concrete_planner.implement_plan(plan=abstract_plan)
            end_time = time.perf_counter()
            trace["Match Time"] = end_time - start_time

            if not concrete_plan:
                trace["Status"] = Status.CONCRETE_PLANNING_FAILURE.name

                return {
                    "question": q,
                    "output": "",
                    "intermediate_steps": self.get_steps(),
                    "state": self.get_state(),
                    "trace": trace,
                }

            tool_mapping = concrete_plan  # Will become attribute of concrete_plan in future versions
            trace["Raw Feasible Matches"] = None  # TODO: Better concrete planner return type
            trace["Tool Mapping"] = {tool_name: tool_call.name for tool_name, tool_call in tool_mapping.items()}

        except Exception:
            # TODO: implement more nuanced custom error types
            trace["Status"] = Status.CONCRETE_PLANNING_ERROR.name
            return {
                "question": q,
                "output": "",
                "intermediate_steps": self.get_steps(),
                "state": self.get_state(),
                "trace": trace,
            }

        # Execution
        with PlanOrchestrator(abstract_plan, concrete_plan) as orchestrator:
            try:
                start_time = time.perf_counter()
                orchestrator.launch()
                orchestrator.join()
                end_time = time.perf_counter()

                output = orchestrator.result
                trace["Execution Time"] = end_time - start_time
            except Exception as e:
                trace["Status"] = Status.EXECUTION_ERROR.name
                trace["Execution Error"] = str(e)
                return {
                    "question": q,
                    "output": "",
                    "intermediate_steps": self.get_steps(),
                    "state": self.get_state(),
                    "trace": trace,
                }

            if not orchestrator.exception_queue.empty():
                trace["Status"] = Status.EXECUTION_ERROR.name
                trace["Execution Error"] = orchestrator.exception_queue.get()
                return {
                    "question": q,
                    "output": "",
                    "intermediate_steps": self.get_steps(),
                    "state": self.get_state(),
                    "trace": trace,
                }

        trace["Status"] = Status.OK.name
        trace["Output"] = output
        return {
            "question": q,
            "output": output,
            "intermediate_steps": self.get_steps(),
            "state": self.get_state(),
            "trace": trace,
        }

    def get_steps(self):
        try:
            response = requests.get(f"{self.service_url}/history")
            if response.status_code == 200:
                steps = response.json().get("history", [])
                steps = [(AgentAction(**action), output) for action, output in steps]
                return steps
            else:
                return []
        except Exception as e:
            print(f"Error fetching steps: {e}")
            return []

    def get_state(self):
        try:
            response = requests.get(f"{self.service_url}/state")
            if response.status_code == 200:
                return response.json().get("state")
            else:
                return None
        except Exception as e:
            print(f"Error fetching state: {e}")
            return None


class AgentFactory:
    def __init__(self, agent, task, service_url):
        self.agent = agent
        self.task = task
        self.service_url = service_url

    def __call__(self, *args):
        agent = ACEAgentLangChain(self.agent, self.task, self.service_url)
        return RunnableLambda(agent.invoke)
