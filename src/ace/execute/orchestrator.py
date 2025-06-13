import base64
import json
import logging
import socket
import threading
from queue import Queue
from typing import Any

from ..schema.abstract import AbstractPlan
from ..schema.concrete import ConcreteToolBase
from .sandbox import PlanExecutionSandbox, ToolExecutionSandbox

logger = logging.getLogger(__name__)


class PlanOrchestrator:
    plan: AbstractPlan
    tools: dict[str, ConcreteToolBase]
    plan_execution_sandbox: PlanExecutionSandbox | None
    tool_execution_sandbox: ToolExecutionSandbox | None
    listener_thread: threading.Thread | None
    shutdown: threading.Event
    toolrunner_ids: set[str]
    exception_queue: Queue
    result: Any | None
    tool_use_history: list[str]

    def __init__(self, plan: AbstractPlan, tools: dict[str, ConcreteToolBase]):
        self.plan = plan
        self.tools = tools
        self.result = None
        self.shutdown = threading.Event()
        self.plan_execution_sandbox = None
        self.tool_execution_sandbox = None
        self.toolrunner_ids = set()
        self.exception_queue = Queue()
        self.result = None
        self.tool_use_history = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.kill()

    def launch(self) -> None:
        # Start listener thread
        self.listener_thread = threading.Thread(target=self.__start_listener, args=("0.0.0.0", 65432), daemon=True)
        self.listener_thread.start()

        # Launch plan execution container and wait for port to open
        self.plan_execution_sandbox = PlanExecutionSandbox(self.plan)
        self.plan_execution_sandbox.launch()
        self.plan_execution_sandbox.wait_for_port()

    def join(self) -> None:
        try:
            if self.listener_thread:
                self.listener_thread.join()
        finally:
            self.kill()
        if not self.exception_queue.empty():
            exception = self.exception_queue.get()
            raise exception

    def kill(self) -> None:
        self.shutdown.set()

        if self.plan_execution_sandbox:
            self.plan_execution_sandbox.kill()
        if self.tool_execution_sandbox:
            self.tool_execution_sandbox.kill()

    def handle_invoke(self, tool_name: str, tool_args: list, tool_kwargs: dict) -> None:
        # Execute tool and retrieve result
        tool = self.tools[tool_name]
        self.tool_execution_sandbox = ToolExecutionSandbox(tool)
        self.tool_use_history.append(tool_name)

        try:
            self.tool_execution_sandbox.launch(tool_args, tool_kwargs)
            if self.tool_execution_sandbox:
                self.toolrunner_ids.add(self.tool_execution_sandbox.id)
        except Exception as e:
            logger.error(f"Failed to execute tool {tool_name}: {e}")
            self.exception_queue.put(ValueError(f"Failed to execute tool {tool_name}: {e}"))
            error = "Tool failed to start."
            error_b64 = base64.b64encode(error.encode()).decode()
            if self.plan_execution_sandbox:
                self.plan_execution_sandbox.send(f"ERROR:{error_b64}")

    def handle_print(self, message: str) -> None:
        print("PRINT:", message)

    def handle_terminate(self) -> None:
        if self.plan_execution_sandbox:
            self.plan_execution_sandbox.kill()
        self.shutdown.set()

    def __handle_client(self, client_socket: socket.socket, addr: tuple[str, int]) -> None:
        logger.info(f"New connection from {addr}")

        try:
            while True:
                # TODO: Implement message buffering
                message = client_socket.recv(4096).decode()
                if not message:
                    break

                logger.debug(f"Received message from {addr}: {message}")
                id, message = message.split(":", 1)

                plan_id = self.plan_execution_sandbox.id if self.plan_execution_sandbox else None
                if id == plan_id:
                    if message.startswith("INVOKE:"):
                        msg_b64 = message[len("INVOKE:") :]
                        msg_json = base64.b64decode(msg_b64).decode()
                        tool_name, tool_args, tool_kwargs = json.loads(msg_json)
                        self.handle_invoke(tool_name, tool_args, tool_kwargs)
                    elif message.startswith("PRINT:"):
                        msg_b64 = message[len("PRINT:") :]
                        msg_str = base64.b64decode(msg_b64).decode()
                        if "Error executing tool code" in message:
                            exception = RuntimeError(message)
                            self.exception_queue.put(exception)
                            raise exception
                        self.handle_print(msg_str)
                    elif message.startswith("TERMINATE"):
                        # Extract final output
                        msg_b64 = message[len("TERMINATE:") :]
                        msg_str = base64.b64decode(msg_b64).decode()
                        self.result = json.loads(msg_str)

                        # Cleanup orchestrator
                        self.handle_terminate()
                        break
                    else:
                        logger.warning(f"Unknown message from {addr}: {message}")
                elif id in self.toolrunner_ids:
                    if message.startswith("RESULT:"):
                        result_b64 = message[len("RESULT:") :]
                        if self.tool_execution_sandbox:
                            self.tool_execution_sandbox.kill()

                        # Encode result and send to worker
                        if self.plan_execution_sandbox:
                            self.plan_execution_sandbox.send(f"RESPONSE:{result_b64}")
                    elif message.startswith("ERROR:"):
                        error = message[len("ERROR:") :]
                        error_b64 = base64.b64encode(error.encode()).decode()
                        if self.tool_execution_sandbox:
                            self.tool_execution_sandbox.kill()

                        if self.plan_execution_sandbox:
                            self.plan_execution_sandbox.send(f"ERROR:{error_b64}")
                        exception = RuntimeError("Error in execution:" + str(error))
                        self.exception_queue.put(exception)
                        raise exception
        except ConnectionResetError:
            logger.warning(f"Connection lost with {addr}")
        except RuntimeError as e:
            logger.error("Exception in client handler", exc_info=True)
            self.exception_queue.put(e)
            raise
        finally:
            client_socket.close()

    def __start_listener(self, ip: str, port: int, backlog: int = 5) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # allow fast rebinding
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(1.0)
        server.bind((ip, port))
        server.listen(backlog)
        logger.info(f"Listener thread listening on {ip}:{port}")

        try:
            while not self.shutdown.is_set():
                try:
                    client_socket, addr = server.accept()
                except TimeoutError:
                    continue
                threading.Thread(target=self.__handle_client, args=(client_socket, addr), daemon=True).start()
        finally:
            server.close()
            logger.info("Listener thread shutting down")
