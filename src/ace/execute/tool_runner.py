# TODO: clean up this code

import base64
import json
import os
import socket
import sys
import threading
from typing import Any

ORCHESTRATOR_IP = "172.17.0.1"
ORCHESTRATOR_PORT = 65432
ORCHESTRATOR_ADDR = (ORCHESTRATOR_IP, ORCHESTRATOR_PORT)


def handle_write(addr, message):
    client_socket = None
    try:
        print("attempting to write to", addr)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(addr)
        message = socket.gethostname() + ":" + message
        print("Attempting to send " + message)
        client_socket.send(message.encode())
        print("tool msg:", message)
    except Exception as e:
        print(f"[-] Error sending message to {addr}: {e}")
    finally:
        if client_socket:
            client_socket.close()


def main() -> None:
    tool_code_b64: str = os.environ.get("TOOL_CODE", "")
    args_json_b64: str = os.environ.get("ARGS", "[]")
    kwargs_json_b64: str = os.environ.get("KWARGS", "{}")
    tool_code: str = base64.b64decode(tool_code_b64).decode()
    args_json: str = base64.b64decode(args_json_b64).decode()
    kwargs_json: str = base64.b64decode(kwargs_json_b64).decode()

    try:
        args: list[Any] = json.loads(args_json)
        kwargs: dict[str, Any] = json.loads(kwargs_json)
    except Exception as e:
        print("Error decoding ARGS:", e, flush=True)
        sys.exit(1)
    if not tool_code.strip():
        print("No TOOL_CODE provided.", flush=True)
        sys.exit(1)
    try:
        local_ns = {}
        exec(tool_code, {}, local_ns)

        result: Any = local_ns["main"](*args, **kwargs)

        result_b64: str = base64.b64encode(json.dumps(result).encode()).decode()

        result_b64 = "RESULT:" + result_b64

        # Send message to orchestrator
        writer_thread = threading.Thread(target=handle_write, args=(ORCHESTRATOR_ADDR, result_b64), daemon=True)
        writer_thread.start()

        writer_thread.join()

    except Exception as e:
        error_msg: str = f"Error executing tool code: {e}"
        error_msg_b64: str = "ERROR:" + base64.b64encode(error_msg.encode()).decode()
        print(error_msg, flush=True)
        writer_thread = threading.Thread(target=handle_write, args=(ORCHESTRATOR_ADDR, error_msg_b64), daemon=True)
        writer_thread.start()
        writer_thread.join()

        sys.exit(1)


if __name__ == "__main__":
    main()
