import base64
import json
import socket
import threading
from queue import Queue
from typing import Any

from ..logging_config import get_logger
from ..schema.abstract import AbstractPlan
from ..schema.concrete import ConcreteToolBase
from .sandbox import PlanExecutionSandbox, ToolExecutionSandbox

logger = get_logger(__name__)


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
        logger.info("Initializing PlanOrchestrator")
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

        logger.debug(f"Plan has {len(plan.abs_tools)} abstract tools")
        logger.debug(f"Tool mapping has {len(tools)} concrete tools")
        logger.debug(f"Tools: {list(tools.keys())}")

        # Log detailed execution plan using enhanced logger
        logger.log_execution("ORCHESTRATOR_INIT", f"tools={len(tools)} abs_tools={len(plan.abs_tools)}")
        logger.log_plan(plan.script, plan.abs_tools)

        # Log tool mappings
        for abs_name, concrete_tool in tools.items():
            logger.log_tool_match(
                abs_name,
                concrete_tool.name,
                {
                    "provider": concrete_tool.provider,
                    "clearances": str(concrete_tool.clearances),
                    "permissions": str(concrete_tool.permissions),
                },
            )

    def __enter__(self):
        logger.debug("Entering PlanOrchestrator context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug("Exiting PlanOrchestrator context")
        if exc_type:
            logger.error(f"PlanOrchestrator context exited with exception: {exc_type.__name__}: {exc_value}")
        self.kill()

    def launch(self) -> None:
        logger.info("Launching plan orchestrator")

        # Start listener thread
        logger.debug("Starting orchestrator listener thread")
        self.listener_thread = threading.Thread(target=self.__start_listener, args=("0.0.0.0", 65432), daemon=True)
        self.listener_thread.start()

        # Launch plan execution container and wait for port to open
        logger.debug("Launching plan execution sandbox")
        try:
            self.plan_execution_sandbox = PlanExecutionSandbox(self.plan)
            self.plan_execution_sandbox.launch()
            logger.debug("Plan execution sandbox launched, waiting for port")
            self.plan_execution_sandbox.wait_for_port()
            logger.info("Plan execution sandbox ready")
        except Exception as e:
            logger.error(f"Failed to launch plan execution sandbox: {e}", exc_info=True)
            raise

    def join(self) -> None:
        logger.debug("Joining orchestrator threads")
        try:
            if self.listener_thread:
                self.listener_thread.join()
                logger.debug("Listener thread joined")
        finally:
            self.kill()

        if not self.exception_queue.empty():
            exception = self.exception_queue.get()
            logger.error(f"Orchestrator completed with exception: {exception}")
            raise exception
        else:
            logger.info("Orchestrator completed successfully")

    def kill(self) -> None:
        logger.debug("Killing orchestrator and cleaning up sandboxes")
        self.shutdown.set()

        if self.plan_execution_sandbox:
            logger.debug("Killing plan execution sandbox")
            self.plan_execution_sandbox.kill()
        if self.tool_execution_sandbox:
            logger.debug("Killing tool execution sandbox")
            self.tool_execution_sandbox.kill()

    def handle_invoke(self, tool_name: str, tool_args: list, tool_kwargs: dict) -> None:
        logger.info(f"Handling tool invocation: {tool_name}")
        logger.debug(f"Tool args: {tool_args}")
        logger.debug(f"Tool kwargs: {tool_kwargs}")

        # Log detailed invocation using enhanced logger
        logger.log_execution("TOOL_INVOKE", f"tool={tool_name} args={tool_args} kwargs={tool_kwargs}")

        if tool_name not in self.tools:
            error_msg = f"Tool {tool_name} not found in available tools: {list(self.tools.keys())}"
            logger.error(error_msg)
            logger.log_execution("TOOL_NOT_FOUND", error_msg)
            self.exception_queue.put(ValueError(error_msg))
            return

        # Execute tool and retrieve result
        tool = self.tools[tool_name]
        logger.debug(f"Executing tool: {tool.name} (provider: {tool.provider})")
        logger.log_execution(
            "TOOL_EXECUTE",
            f"concrete_tool={tool.name} provider={tool.provider} clearances={tool.clearances} permissions={tool.permissions}",
        )

        self.tool_execution_sandbox = ToolExecutionSandbox(tool)
        self.tool_use_history.append(tool_name)

        try:
            self.tool_execution_sandbox.launch(tool_args, tool_kwargs)
            if self.tool_execution_sandbox:
                self.toolrunner_ids.add(self.tool_execution_sandbox.id)
                logger.debug(f"Tool {tool_name} execution started with container ID: {self.tool_execution_sandbox.id}")
        except Exception as e:
            logger.error(f"Failed to execute tool {tool_name}: {e}", exc_info=True)
            self.exception_queue.put(ValueError(f"Failed to execute tool {tool_name}: {e}"))
            error = "Tool failed to start."
            error_b64 = base64.b64encode(error.encode()).decode()
            if self.plan_execution_sandbox:
                self.plan_execution_sandbox.send(f"ERROR:{error_b64}")

    def handle_print(self, message: str) -> None:
        logger.debug(f"Received PRINT message: {message[:100]}...")
        print("PRINT:", message)

    def handle_terminate(self) -> None:
        logger.info("Received TERMINATE message")
        if self.plan_execution_sandbox:
            self.plan_execution_sandbox.kill()
        self.shutdown.set()

    def __handle_client(self, client_socket: socket.socket, addr: tuple[str, int]) -> None:
        logger.debug(f"New connection from {addr}")

        try:
            while True:
                # TODO: Implement message buffering
                message = client_socket.recv(4096).decode()
                if not message:
                    logger.debug(f"Empty message received from {addr}, closing connection")
                    break

                logger.debug(f"Received message from {addr}: {message[:200]}...")
                try:
                    id, message = message.split(":", 1)
                except ValueError:
                    logger.warning(f"Invalid message format from {addr}: {message}")
                    continue

                plan_id = self.plan_execution_sandbox.id if self.plan_execution_sandbox else None
                logger.debug(f"Processing message from ID: {id}, plan_id: {plan_id}")

                if id == plan_id:
                    if message.startswith("INVOKE:"):
                        logger.debug("Processing INVOKE message")
                        msg_b64 = message[len("INVOKE:") :]
                        msg_json = base64.b64decode(msg_b64).decode()
                        tool_name, tool_args, tool_kwargs = json.loads(msg_json)
                        logger.info(f"Plan invoking tool: {tool_name}")
                        self.handle_invoke(tool_name, tool_args, tool_kwargs)
                    elif message.startswith("PRINT:"):
                        logger.debug("Processing PRINT message")
                        msg_b64 = message[len("PRINT:") :]
                        msg_str = base64.b64decode(msg_b64).decode()
                        if "Error executing tool code" in message:
                            logger.error(f"Tool execution error detected in PRINT: {msg_str}")
                            exception = RuntimeError(message)
                            self.exception_queue.put(exception)
                            raise exception
                        self.handle_print(msg_str)
                    elif message.startswith("TERMINATE"):
                        logger.info("Processing TERMINATE message")
                        # Extract final output
                        msg_b64 = message[len("TERMINATE:") :]
                        msg_str = base64.b64decode(msg_b64).decode()
                        self.result = json.loads(msg_str)
                        logger.info(f"Plan execution completed with result: {type(self.result).__name__}")

                        # Cleanup orchestrator
                        self.handle_terminate()
                        break
                    else:
                        logger.warning(f"Unknown message type from plan {id}: {message[:100]}")
                elif id in self.toolrunner_ids:
                    logger.debug(f"Processing message from tool runner {id}")
                    if message.startswith("RESULT:"):
                        logger.debug("Processing tool RESULT message")
                        result_b64 = message[len("RESULT:") :]
                        if self.tool_execution_sandbox:
                            self.tool_execution_sandbox.kill()

                        # Encode result and send to worker
                        if self.plan_execution_sandbox:
                            self.plan_execution_sandbox.send(f"RESPONSE:{result_b64}")
                            logger.debug("Sent tool result back to plan")
                    elif message.startswith("ERROR:"):
                        logger.error(f"Tool execution error from {id}")
                        error = message[len("ERROR:") :]
                        error_b64 = base64.b64encode(error.encode()).decode()
                        if self.tool_execution_sandbox:
                            self.tool_execution_sandbox.kill()

                        if self.plan_execution_sandbox:
                            self.plan_execution_sandbox.send(f"ERROR:{error_b64}")
                        exception = RuntimeError("Error in execution:" + str(error))
                        self.exception_queue.put(exception)
                        raise exception
                else:
                    logger.warning(f"Received message from unknown ID {id}: {message[:100]}")
        except ConnectionResetError:
            logger.warning(f"Connection lost with {addr}")
        except RuntimeError as e:
            logger.error(f"Runtime error in client handler for {addr}: {e}", exc_info=True)
            self.exception_queue.put(e)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in client handler for {addr}: {e}", exc_info=True)
            self.exception_queue.put(e)
            raise
        finally:
            client_socket.close()
            logger.debug(f"Closed connection to {addr}")

    def __start_listener(self, ip: str, port: int, backlog: int = 5) -> None:
        logger.info(f"Starting orchestrator listener on {ip}:{port}")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # allow fast rebinding
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(1.0)

        try:
            server.bind((ip, port))
            server.listen(backlog)
            logger.info(f"Listener thread listening on {ip}:{port}")
        except Exception as e:
            logger.error(f"Failed to start listener on {ip}:{port}: {e}", exc_info=True)
            raise

        try:
            while not self.shutdown.is_set():
                try:
                    client_socket, addr = server.accept()
                    logger.debug(f"Accepted connection from {addr}")
                    threading.Thread(target=self.__handle_client, args=(client_socket, addr), daemon=True).start()
                except TimeoutError:
                    continue
                except Exception as e:
                    if not self.shutdown.is_set():
                        logger.error(f"Error accepting connection: {e}")
        finally:
            server.close()
            logger.info("Listener thread shutting down")
