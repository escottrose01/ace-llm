import socket
import threading
from typing import Optional

from ace.schema.concrete import ConcreteToolBase
from flask import Flask, jsonify, request
from langchain_benchmarks.schema import ToolUsageTask


class LangChainAceAdapter:
    def __init__(self, task: ToolUsageTask):
        self.task = task

        # Create the initial environment
        self.env = task.create_environment()
        self.tools = self.env.tools
        self.tool_lookup = {tool.name: tool for tool in self.tools}
        self.history: list[tuple] = []

        # Ensure all tool names are unique
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("All tool names must be unique.")

        # Setup the Flask app
        self.app = Flask(__name__)
        self.port = None
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok"}), 200

        @self.app.route("/tools", methods=["GET"])
        def tools():
            return jsonify(
                {
                    "functions": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "args_schema": tool.args_schema.model_json_schema(),
                            "output_schema": tool.output_schema.model_json_schema(),
                        }
                        for tool in self.tools
                    ]
                }
            ), 200

        @self.app.route("/history", methods=["GET"])
        def history():
            return jsonify({"history": self.history}), 200

        @self.app.route("/invoke", methods=["POST"])
        def invoke():
            # Safely access JSON payload
            data = request.get_json(force=True) or {}
            tool_name = data.get("tool")
            args = data.get("args", {})

            # Validate tool name
            if not tool_name:
                return jsonify({"success": False, "error": "No tool specified"}), 400
            elif not isinstance(tool_name, str):
                return jsonify({"success": False, "error": "Tool name must be a string"}), 400

            # Find the tool by name
            try:
                tool = self.tool_lookup[tool_name]
            except KeyError:
                return jsonify({"success": False, "error": f"Tool '{tool_name}' not found"}), 404

            # Run the tool with the provided arguments
            try:
                result = tool.invoke(args)
            except Exception as e:
                return jsonify({"success": False, "error": f"Error executing tool code: {e!s}"})

            self.history.append(
                (
                    {
                        "tool": tool_name,
                        "tool_input": args,
                        "log": "",
                    },
                    result,
                )
            )

            return jsonify({"success": True, "result": result}), 200

        @self.app.route("/state", methods=["GET"])
        def state():
            return jsonify({"state": self.env.read_state()})  # type: ignore

        @self.app.route("/reset", methods=["POST"])
        def reset():
            # Reset the environment by re-creating it
            self.env = self.task.create_environment()
            self.tools = self.env.tools
            self.tool_lookup = {tool.name: tool for tool in self.tools}
            self.history = []

            return jsonify({"status": "reset"}), 200

    def start(self, host: str = "0.0.0.0", port: Optional[int] = None) -> int:
        """
        Start the Flask service on a random available port (if port is None),
        returning the port number bound.
        """
        # Determine a free port if not provided
        # There is an obvious race condition / TOCTOU issue, but acceptable for now
        if port is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, 0))
            port = sock.getsockname()[1]
            sock.close()
        self.port = port

        # Run the Flask app in a background thread
        server = threading.Thread(target=lambda: self.app.run(host=host, port=port), daemon=True)
        server.start()

        # At this point, port has been set to an int
        assert self.port is not None, "Service port must be set before returning"
        return self.port

    def create_ace_tools(self) -> list[ConcreteToolBase]:
        """
        Construct ACE-compatible tool wrappers that forward invocations
        to the running REST service.
        """
        wrappers = []
        for tool in self.tools:
            wrappers.append(
                LangChainProxyTool(
                    name=tool.name,
                    description=tool.description,
                    function_name=tool.name,
                    service_url=f"http://172.17.0.1:{self.port}",
                    args_schema=tool.args_schema,
                    output_schema=tool.output_schema,
                )
            )
            pass
        return wrappers


class LangChainProxyTool(ConcreteToolBase):
    function_name: str = ""
    service_url: str = ""

    def __init__(
        self,
        name: str,
        description: str,
        function_name: str,
        service_url: str,
        args_schema=None,
        output_schema=None,
    ):
        super().__init__(
            name=name,
            provider="langchain",
            description=description,
            clearances=set(),
            permissions=set(),
            args_schema=args_schema,
            output_schema=output_schema,
        )
        object.__setattr__(self, "function_name", function_name)
        object.__setattr__(self, "service_url", service_url)

    def generate_source(self) -> str:
        return (
            f"def main(*args, **kwargs):\n"
            f"    import urllib.request\n"
            f"    import json\n"
            f"    url = '{self.service_url}/invoke'\n"
            f"    payload = {{'tool': '{self.function_name}', 'args': kwargs}}\n"
            f"    data = json.dumps(payload).encode('utf-8')\n"
            f"    req = urllib.request.Request(url, data=data, headers={{'Content-Type': 'application/json'}}, method='POST')\n"
            f"    try:\n"
            f"        with urllib.request.urlopen(req, timeout=30) as resp:\n"
            f"            resp_data = resp.read().decode('utf-8')\n"
            f"            data = json.loads(resp_data)\n"
            f"            if resp.status == 200 and data.get('success'):\n"
            f"                return data.get('result')\n"
            f"            else:\n"
            f"                raise RuntimeError('LangChain function failed: ' + str(data.get('error', 'HTTP ' + str(resp.status))))\n"
            f"    except Exception as e:\n"
            f"        raise RuntimeError('LangChainProxyTool error: ' + str(e))\n"
        )
