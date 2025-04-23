# worker.py
from typing import Any

import os
import ast
import queue
import sys
import socket
import threading
import base64
import json

listener_ip = socket.gethostbyname(socket.gethostname())
listener_port = 8080
listener_addr = (listener_ip, listener_port)

ORCHESTRATOR_IP = "172.17.0.1"
ORCHESTRATOR_PORT = 65432
ORCHESTRATOR_ADDR = (ORCHESTRATOR_IP, ORCHESTRATOR_PORT)

response_queue = queue.Queue()

orchestrator_response = ''


def display(data: str) -> None:
    if isinstance(data, dict):
        data = json.dumps(data)
    encoded = base64.b64encode(data.encode()).decode()
    writer_thread = threading.Thread(
        target=handle_write,
        args=(ORCHESTRATOR_ADDR, f"PRINT:{encoded}"), daemon=True
    )
    writer_thread.start()


def invoke(tool_name: str, *args: Any, **kwargs: Any) -> Any:
    # Prepare message
    msg_json = json.dumps([tool_name, args, kwargs])
    msg_b64 = base64.b64encode(msg_json.encode()).decode()
    message = f"INVOKE:{msg_b64}"

    # Send message to orchestrator
    writer_thread = threading.Thread(
        target=handle_write,
        args=(ORCHESTRATOR_ADDR, message),
        daemon=True
    )
    writer_thread.start()

    response_b64 = response_queue.get()
    if response_b64.startswith("ERROR:"):
        response_b64 = response_b64[len("ERROR:"):]
        err_msg = base64.b64decode(response_b64).decode()
        raise RuntimeError(err_msg)
    elif response_b64.startswith("RESPONSE:"):
        response_b64 = response_b64[len("RESPONSE:"):]
        response = base64.b64decode(response_b64).decode()
        print("[WORKER] Received response:", response)
        return json.loads(response)
    else:
        raise ValueError(f"Unexpected response format: {response_b64}")


def start_listener():
    """Starts the server and listens for connections until shutdown."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(listener_addr)
    server.listen(5)
    # Set timeout so we can periodically check for shutdown
    server.settimeout(1.0)
    print(f"[WORKER] listening on {listener_ip}:{listener_port}")

    while True:
        try:
            client_socket, addr = server.accept()
        except socket.timeout:
            continue  # Check the shutdown event again
        threading.Thread(
            target=handle_client,
            args=(client_socket, addr),
            daemon=True
        ).start()


def handle_client(client_socket, address):
    """Handles incoming messages from a connected client."""
    print(f"[WORKER] New connection from {address}")

    try:
        while True:
            message = client_socket.recv(4096).decode()
            if not message:
                break

            if message.startswith("RESPONSE:"):
                response_queue.put(message)
            elif message.startswith("ERROR:"):
                error_b64 = f"ERROR:{message.split(':', 1)[1]}"
                response_queue.put(error_b64)

            print(f"[WORKER] Received {message} from {address}")
    except ConnectionResetError:
        print(f"[-] Connection lost with {address}")
    finally:
        client_socket.close()


def handle_write(addr, message):
    try:
        print("attempting to write to", addr)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(addr)
        message = socket.gethostname() + ':' + message
        client_socket.send(message.encode())
        print("worker msg:", message)
    except Exception as e:
        print(f"[-] Error sending message to {addr}: {e}")
    finally:
        client_socket.close()


def main() -> None:
    # Start listener thread
    listener_thread = threading.Thread(target=start_listener, daemon=True)
    listener_thread.start()

    # Get script from environment variable
    script_b64 = os.environ.get("SCRIPT", "")
    script = base64.b64decode(script_b64).decode() if script_b64 else ""
    if not script.strip():
        msg_b64 = base64.b64encode(
            "No script provided in $SCRIPT.".encode()).decode()
        writer_thread = threading.Thread(
            target=handle_write,
            args=(ORCHESTRATOR_ADDR, f"PRINT:{msg_b64}"),
            daemon=True
        )
        writer_thread.start()
        sys.exit(1)

    # Execute script
    try:
        code = compile(
            ast.parse(script, mode="exec"),
            filename="<script>",
            mode="exec"
        )

        local_ns = {}
        exec(code, globals(), local_ns)
    except Exception as e:
        e_msg = f"Error executing script: {e}"
        e_msg_b64 = base64.b64encode(e_msg.encode()).decode()
        writer_thread = threading.Thread(
            target=handle_write,
            args=(ORCHESTRATOR_ADDR, f"PRINT:{e_msg_b64}"),
            daemon=True
        )
        writer_thread.start()

    # Send termination message
    final_output: Any = local_ns.get("final_output")
    final_output_json = json.dumps(final_output)
    print("[WORKER] Final output:", final_output_json)
    final_output_b64 = base64.b64encode(final_output_json.encode()).decode()
    term_msg = f"TERMINATE:{final_output_b64}"
    writer_thread = threading.Thread(
        target=handle_write,
        args=(ORCHESTRATOR_ADDR, term_msg),
        daemon=True
    )
    writer_thread.start()

    # Signal the listener to shutdown and wait for it to finish.
    writer_thread.join()
    listener_thread.join()


if __name__ == "__main__":
    main()
