
from __future__ import annotations

import json
import socket
import threading
from queue import Queue, Empty
from typing import Any, Dict, Optional

HOST = "0.0.0.0"
PORT = 5000

node_sockets: Dict[str, socket.socket] = {}
_node_lock = threading.Lock()
_message_queue: "Queue[Dict[str, Any]]" = Queue()
_server_socket: Optional[socket.socket] = None


def _recv_lines(sock: socket.socket):
    buffer = b""
    while True:
        try:
            data = sock.recv(1024)
        except OSError:
            break
        if not data:
            break
        buffer += data
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            yield line


def _register_node(node: str, sock: socket.socket) -> None:
    with _node_lock:
        node_sockets[node] = sock
    print(f"[NET] Registered node '{node}'")


def _handle_client(sock: socket.socket, addr) -> None:
    node_name: Optional[str] = None
    print(f"[NET] New connection from {addr}")
    try:
        for raw_line in _recv_lines(sock):
            if not raw_line.strip():
                continue
            try:
                message = json.loads(raw_line.decode("utf-8"))
            except json.JSONDecodeError:
                print(f"[NET] Failed to parse JSON from {addr}: {raw_line!r}")
                continue

            if message.get("type") == "hello":
                node = message.get("node")
                if node:
                    node_name = node
                    _register_node(node_name, sock)
                continue

            if node_name and "node" not in message:
                message["node"] = node_name

            _message_queue.put(message)
    finally:
        print(f"[NET] Connection closed {addr}")
        if node_name:
            with _node_lock:
                node_sockets.pop(node_name, None)
        try:
            sock.close()
        except OSError:
            pass


def _accept_loop(server_sock: socket.socket) -> None:
    while True:
        try:
            client_sock, addr = server_sock.accept()
        except OSError:
            break
        thread = threading.Thread(target=_handle_client, args=(client_sock, addr), daemon=True)
        thread.start()


def start_server(host: str = HOST, port: int = PORT) -> None:
    global _server_socket
    if _server_socket is not None:
        return
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen()
    _server_socket = server_sock
    print(f"[NET] Listening on {host}:{port}")
    accept_thread = threading.Thread(target=_accept_loop, args=(server_sock,), daemon=True)
    accept_thread.start()


def get_next_message(timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
    try:
        return _message_queue.get(timeout=timeout)
    except Empty:
        return None


def _send(sock: socket.socket, message: Dict[str, Any]) -> None:
    payload = json.dumps(message).encode("utf-8") + b"\n"
    try:
        sock.sendall(payload)
    except OSError as exc:
        print(f"[NET] Failed to send message: {exc}")


def send_to_node(node_name: str, message: Dict[str, Any]) -> bool:
    with _node_lock:
        sock = node_sockets.get(node_name)
    if not sock:
        print(f"[NET] No socket for node '{node_name}'")
        return False
    _send(sock, message)
    print(f"[NET] Sent message to {node_name}")
    return True


def broadcast(message: Dict[str, Any]) -> None:
    with _node_lock:
        sockets = list(node_sockets.items())
    for node_name, sock in sockets:
        _send(sock, message)
        print(f"[NET] Broadcast to {node_name}")
