import socket
import logging
import docker.models.containers

logger = logging.getLogger(__name__)


def write_to_socket(addr: tuple[str, int], message: str) -> None:
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(addr)
        client_socket.send(message.encode())
        client_socket.close()
    except Exception as e:
        logger.error(f"Error sending {message} to {addr}: {e}")
    finally:
        client_socket.close()


def safe_container_cleanup(container: docker.models.containers.Container) -> None:
    try:
        container.kill()
    except Exception as e:
        pass

    try:
        container.remove(force=True)
    except Exception as e:
        pass
