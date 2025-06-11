import json
import re
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ace.execute.orchestrator import PlanOrchestrator
from ace.plan.abstract import AbstractPlanner
from ace.plan.concrete import SimpleConcretePlanner
from ace.schema.concrete import CustomTool
from ace.sentinel_agent import AceAgent
from ace.tools.helper import parse_schemas_from_json
from ace.tools.manager import ToolManager


class InjecSentinelAgent(AceAgent):
    """
    An enchanced version of SentinelAgent, designed to run with InjecAgent benchmark.
    One instance per test (for now)
    """

    count = 0
    orch_error_count = 0

    def __init__(
        self,
        abstract_planner: AbstractPlanner,
        user_tool_string: str,
        attacker_tool_strings: list[str],
        is_user_tool=True,
    ):
        # base_llm = ChatOpenAI(
        #     model="o3-mini",
        #     #openai_api_base="http://localhost:8000/v1"
        # )
        base_llm = ChatOpenAI(model="o3-mini")

        embedding_model = OpenAIEmbeddings()

        self.skip_flag = user_tool_string in ("GmailSearchEmails")

        tools: list[CustomTool] = []
        user_toolkit_name, user_tool_name = InjecSentinelAgent.split_tool(user_tool_string)

        new_tool = InjecSentinelAgent.impl_tool(user_toolkit_name, user_tool_name, is_user_tool)
        tools.append(new_tool)

        for attacker_tool in attacker_tool_strings:
            toolkit, toolname = InjecSentinelAgent.split_tool(attacker_tool)
            new_tool: CustomTool = InjecSentinelAgent.impl_tool(toolkit, toolname, False)
            print(f"!!NEWTOOLNAME: {new_tool.name}")
            tools.append(new_tool)
        tool_manager = ToolManager(tools)

        concrete_planner = SimpleConcretePlanner(
            tool_manager=tool_manager, base_llm=base_llm, embedding_model=embedding_model
        )
        self.actions: list = []
        self.final_code: str = ""
        self.results: list = []

        super().__init__(abstract_planner, concrete_planner)

    def run_query(self, query: str, debug=False):
        InjecSentinelAgent.count += 1

        print(f"\nRunning Query Number {InjecSentinelAgent.count}")

        if self.skip_flag:
            return ("Error", "execution error: From keyword")

        abstract_plan = self.abstract_planner.generate_abstract_plan(query)

        matches = self.concrete_planner.generate_all_feasible_matches(abstract_plan.abs_tools)

        if debug:
            print("Matched tools:")
        try:
            for abs_matches in matches:
                for match in abs_matches:
                    if debug:
                        print(match.name)
                    code: str = match.generate_source()
                    self.final_code = code
                    if debug:
                        print(code)
                        print()
                        print()
        except:
            pass

        try:
            tool_mapping = self.concrete_planner.implement_plan(abstract_plan)
        except Exception as e:
            self.results = self.concrete_planner.results
            print("!!Error:", e)
            return ("Error", str(e))

        self.results = self.concrete_planner.results

        for tool in tool_mapping.values():
            self.actions.append(tool.name)

        for key, value in tool_mapping.items():
            if debug:
                print(f"The tool mapping (before orchestration): {value.as_dict()}")

        try:
            # TO BUILD:
            # docker build -t worker-image -f Dockerfile.worker .
            # docker build -t tool-runner-image -f Dockerfile.tool .
            with PlanOrchestrator(abstract_plan, tool_mapping) as orchestrator:
                orchestrator.launch()
                orchestrator.join()
                if not orchestrator.exception_queue.empty():
                    InjecSentinelAgent.orch_error_count += 1
                    return ("Error", str(orchestrator.exception_queue.get()))
                final_message = orchestrator.result

        except Exception as e:
            if debug:
                print("!!Error", e)
            InjecSentinelAgent.orch_error_count += 1
            return ("Error", str(e))

        if debug:
            print("NO ERRORS ENCOUNTERED")

        return ("End", final_message)

    @staticmethod
    def translate_function_signature(toolkit_function: dict):
        """
        Translate a toolkit function signature to a JSON Schema function signature.
        Parameters:
        toolkit_function (dict): A dictionary representing a function in the toolkit schema.
            Expected keys include:
                - "name": the function name.
                - "summary": a brief description.
                - "parameters": a list of parameters, where each parameter is a dict with:
                    - "name": parameter name.
                    - "type": parameter type (e.g., "string", "integer").
                    - "description": a description of the parameter.
                    - "required": a boolean indicating if it is required.
                - "returns": a list of return values, where each return is a dict with:
                    - "name": return value name.
                    - "type": return value type.
                    - "description": a description of the return value.
            (The "exceptions" field is ignored in this translation.)
        Returns:
        dict: A JSON Schema representing the function signature in the "TO" language.
        """
        # Extract the function name. Default to 'unnamed_function' if not provided.
        function_name = toolkit_function.get("name", "unnamed_function")
        # Process parameters to build the "request" schema.
        parameters = toolkit_function.get("parameters", [])
        request_properties = {}
        request_required = []
        for param in parameters:
            param_name = param["name"]
            # Map the type directly (assumes standard types)
            param_type = param.get("type", "string")
            request_properties[param_name] = {"type": param_type, "description": param.get("description", "")}
            if param.get("required", False):
                request_required.append(param_name)
        # Process returns to build the "response" schema.
        returns = toolkit_function.get("returns", [])
        if len(returns) == 1:
            ret = returns[0]
            response_schema = {"type": ret.get("type", "string"), "description": ret.get("description", "")}
        else:
            response_properties = {}
            response_required = []
            for ret in returns:
                ret_name = ret["name"]
                ret_type = ret.get("type", "string")
                response_properties[ret_name] = {"type": ret_type, "description": ret.get("description", "")}
                response_required.append(ret_name)
            response_schema = {"type": "object", "properties": response_properties, "required": response_required}

        # Build the overall JSON Schema.
        json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                function_name: {
                    "type": "object",
                    "description": toolkit_function.get("summary", ""),
                    "properties": {
                        "request": {"type": "object", "properties": request_properties, "required": request_required},
                        "response": response_schema,
                    },
                    "required": ["request", "response"],
                }
            },
            "required": [function_name],
        }

        return json_schema

    @staticmethod
    def split_tool(tool_str) -> tuple[str, str]:
        # Splits the tool into toolkit and tool name
        exceptions = {
            "EvernoteManager",
            "GitHub",
            "GoogleCalendar",
            "BankManager",
            "TwitterManager",
            "GoogleHome",
            "FacebookManager",
            "DeepfakeGenerator",
            "The23andMe",
            "GoogleSearch",
            "WebBrowser",
            "NortonIdentitySafe",
            "AugustSmartLock",
            "IFTTT",
            "GoogleMap",
            "IndoorRobot",
            "EmergencyDispatchSystem",
            "EthereumManager",
            "FedExShipManager",
            "InventoryManagementSystem",
            "CiscoUmbrella",
            "EpicFHIR",
            "TrafficControl",
            "TDAmeritrade",
            "InvestmentManager",
        }

        for exception in exceptions:
            if exception in tool_str:
                return (exception, tool_str[len(exception) :])

        match = re.match(r"([A-Z][a-z]*)([A-Z].*)", tool_str)
        return (match.group(1), match.group(2))

    @staticmethod
    def generate_code(tool: dict[str, Any], user: bool) -> str:
        """
        Given a tool definition as a dictionary, returns a string containing a block of Python code.
        The code block defines a main() function with an argument for each parameter in the tool,
        without any type annotations. Optional parameters have a default value of None.

        If the generated code returns a dictionary with only one key, the main function will return
        the value for that key instead.
        """
        parameters = tool.get("parameters", [])
        required_args = []
        optional_args = []

        for param in parameters:
            name = param["name"]
            # Adjust if parameter name is a Python keyword.
            if name == "from":
                name = "from_"

            if param.get("required", False):
                # Add required parameter (without default)
                required_args.append(name)
            else:
                # Optional parameter: add default value of None.
                optional_args.append(f"{name} = None")

        # Combine required parameters and optional parameters.
        arg_list = required_args + optional_args

        function_block = ""

        if user:
            function_block = (
                "def main("
                + ", ".join(arg_list)
                + "):\n"
                + "    _result = "
                + str(InjecSentinelAgent.__get_func_impl(tool["name"]))
                + "\n"
                + "    if isinstance(_result, dict) and len(_result) == 1:\n"
                + "        return list(_result.values())[0]\n"
                + "    else:\n"
                + "        return _result\n"
            )
        else:
            with open("injec_eval/res/attacker_responses.json") as f:
                cases: dict = json.load(f)
                for key, value in cases.items():
                    tool_name = key.split(",")[0][1:]
                    if tool_name != tool["name"]:
                        continue
                    function_block = (
                        "def main(" + ", ".join(arg_list) + "):\n" + f"    result = {value}\n"
                        "    if isinstance(result, dict) and len(result) == 1:\n"
                        + "        return list(result.values())[0]\n"
                        + "    else:\n"
                        + "        return result\n"
                    )

        if not function_block:
            function_block = "def main():\n" + "\treturn"

        return function_block.strip()

    @staticmethod
    def __get_func_impl(tool_name: str) -> str:
        with open("injec_eval/res/directharm.json") as f:
            cases: list = json.load(f)
            for test in cases:
                if test["User Tool"] != tool_name:
                    continue
                response: str = test["Tool Response Template"]
                response = response.replace("<Attacker Instruction>", test["Attacker Instruction"])
                return response

    @staticmethod
    def manual_increment():
        InjecSentinelAgent.count += 1

    @staticmethod
    def impl_tool(toolkit_name: str, tool_name: str, user: bool, debug: bool = False) -> CustomTool:
        """
        Gets the tool from tools.json, along with its corresponding function implementation from test cases.
        NOTE: tool_name must not have toolkit_name in it (which is default in test cases)
        """
        tools: dict = {}
        with open("injec_eval/res/tools.json") as f:
            toolkits = json.load(f)
            for toolkit_dict in toolkits:
                if toolkit_dict["toolkit"] == toolkit_name:
                    tools = toolkit_dict["tools"]
                    for tool in tools:
                        if tool["name"] == tool_name:
                            tool["name"] = toolkit_name + tool_name
                            signature = InjecSentinelAgent.translate_function_signature(tool)

                            param_schema, return_schema = parse_schemas_from_json(signature)
                            if debug:
                                print(
                                    "Signature:",
                                    json.dumps(param_schema.model_json_schema(), indent=4),
                                    json.dumps(return_schema.model_json_schema()),
                                    sep="\n",
                                )
                            function = InjecSentinelAgent.generate_code(tool, user)
                            if debug:
                                print("Function:\n", function)

                            try:
                                impl_tool = CustomTool(
                                    name=tool["name"],
                                    provider="InjecAgent",
                                    description=tool["summary"],
                                    permissions=set(),
                                    clearances=set(),
                                    args_schema=param_schema,
                                    output_schema=return_schema,
                                    source_code=function,
                                )
                            except Exception as e:
                                print("Error with custom tool, moving on...")
                                raise e
                            """
                            manifest_path = "sentinel/injecagent/injec_manifest.jsonl"
                            entry_exists = False

                            with open(manifest_path, "r") as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        data = json.loads(line)
                                        if "name" in data and impl_tool.name == data["name"]:
                                            entry_exists = True
                                    except:
                                        continue

                            if not entry_exists:
                                with open(manifest_path, "a") as f:
                                    new_entry = {
                                        "name": impl_tool.name,
                                        "provider": impl_tool.provider,
                                        "description": impl_tool.description,
                                        "tool_type": "custom",
                                        "path": "", # NOTE: Implement later?
                                        "code": impl_tool.source_code,
                                        "clearances": [
                                            "general"
                                        ],
                                        "permissions": []
                                    }
                                    f.write(new_entry + "\n")

                            print(json.dumps(impl_tool.as_dict(), indent=4))
                            """
                            return impl_tool

        raise RuntimeError(f"Could not find the given tools for {toolkit_name}, {tool_name}")
