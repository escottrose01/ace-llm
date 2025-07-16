import base64
import json
import logging
import threading
import time
from enum import Enum
from typing import Any

import docker
import docker.models
import docker.models.containers

from ..schema.abstract import AbstractPlan
from ..schema.concrete import ConcreteToolBase
from .helper import safe_container_cleanup, write_to_socket

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    NOT_STARTED = 0
    PENDING = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4


class PlanExecutionSandbox:
    plan: AbstractPlan
    container: docker.models.containers.Container | None
    port_ready: threading.Event

    def __init__(self, plan: AbstractPlan):
        self.plan = plan
        self.container = None
        self.port_ready = threading.Event()

    def launch(self) -> None:
        # Prepare container inputs
        plan_code = self.plan.compile_for_protocol()
        plan_code_b64 = base64.b64encode(plan_code.encode()).decode()
        docker_env = {"SCRIPT": plan_code_b64}

        # Spawn plan execution container
        docker_client = docker.from_env()
        container = docker_client.containers.run(
            image="worker-image",
            environment=docker_env,
            detach=True,
            ports={"8080/tcp": None},
        )

        self.container = container

    def wait_for_port(self, timeout: float = 30.0, interval: float = 0.5) -> None:
        if not self.container:
            raise RuntimeError("Container not launched")

        start_time = time.time()
        while True:
            self.container.reload()
            ports = self.container.ports.get("8080/tcp")
            if ports and len(ports) > 0 and ports[0].get("HostPort"):
                self.port_ready.set()
                break
            if self.container.status == "exited":
                raise RuntimeError("Container exited unexpectedly")
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for container port")
            time.sleep(interval)

    def send(self, message: str, timeout: float = 30.0) -> None:
        if not self.container:
            raise RuntimeError("Container not launched")

        if not self.port_ready.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for container port")

        writer_thread = threading.Thread(target=write_to_socket, args=(self.addr, message))
        writer_thread.start()

    def kill(self) -> None:
        # TODO: think about how to handle errors
        if not self.container:
            raise RuntimeError("Container not launched")

        safe_container_cleanup(self.container)

    @property
    def status(self) -> ExecutionStatus:
        if not self.container:
            return ExecutionStatus.NOT_STARTED
        elif self.container.status == "exited":
            return ExecutionStatus.COMPLETED
        elif self.container.status == "running":
            return ExecutionStatus.RUNNING
        else:
            return ExecutionStatus.FAILED

    @property
    def port(self) -> int:
        if not self.container:
            raise RuntimeError("Container not launched")
        if not self.container.ports:
            raise RuntimeError("Container port not open")
        ports = self.container.ports.get("8080/tcp")
        if not ports:
            raise RuntimeError("Port mapping not found")
        return int(ports[0]["HostPort"])

    @property
    def ip(self) -> str:
        return "127.0.0.1"

    @property
    def addr(self) -> tuple[str, int]:
        return (self.ip, self.port)

    @property
    def id(self) -> str:
        if not self.container:
            raise RuntimeError("Container not launched")
        container_id = self.container.id
        if not container_id:
            raise RuntimeError("Container ID not available")
        return container_id[:12]


class ToolExecutionSandbox:
    tool: ConcreteToolBase
    container: docker.models.containers.Container | None

    def __init__(self, tool: ConcreteToolBase):
        self.tool = tool
        self.container = None

    def launch(self, args: list[Any] = [], kwargs: dict[str, Any] = {}) -> None:
        # Enforce input typing
        # TODO: enforce input typing according to tool schema

        # Prepare container inputs
        src_code = self.tool.generate_source()
        kwargs_json = json.dumps(kwargs)
        args_json = json.dumps(args)
        src_code_b64 = base64.b64encode(src_code.encode()).decode()
        args_b64 = base64.b64encode(args_json.encode()).decode()
        kwargs_b64 = base64.b64encode(kwargs_json.encode()).decode()
        docker_env = {
            "TOOL_CODE": src_code_b64,
            "ARGS": args_b64,
            "KWARGS": kwargs_b64,
        }

        # Spawn tool environment
        docker_client = docker.from_env()
        self.container = docker_client.containers.run(
            image="tool-runner-image",
            environment=docker_env,
            detach=True,
            ports={"8080/tcp": None},
        )

    def wait_for_result(self, timeout: float | None = None) -> None:
        if not self.container:
            raise RuntimeError("Container not launched")
        self.container.wait(timeout=timeout)

    def kill(self) -> None:
        if not self.container:
            return

        safe_container_cleanup(self.container)

    @property
    def status(self) -> ExecutionStatus:
        if not self.container:
            return ExecutionStatus.NOT_STARTED
        elif self.container.status == "exited":
            return ExecutionStatus.COMPLETED
        elif self.container.status == "running":
            return ExecutionStatus.RUNNING
        else:
            return ExecutionStatus.FAILED

    @property
    def result(self) -> Any:
        if not self.container:
            raise RuntimeError("Container not launched")
        logs = self.container.logs()
        if isinstance(logs, bytes):
            output_b64 = logs.decode()
        else:
            output_b64 = str(logs)
        output = json.loads(base64.b64decode(output_b64).decode())
        return output

    @property
    def id(self) -> str:
        if not self.container:
            raise RuntimeError("Container not launched")
        container_id = self.container.id
        if not container_id:
            raise RuntimeError("Container ID not available")
        return container_id[:12]
