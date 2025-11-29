#!/usr/bin/env python3
"""
Simple live dashboard for the sleep-env project.

- Connects to the brain_server TCP endpoint as a "dashboard" client.
- Shows:
    * Latest sensor values published by each ESP32 / node
    * Latest decisions (plan "state") from the brain
    * Latest config/parameters sent to each node (via plan["config"])
    * Model version and overall status
- Tracks a small rolling event log.

Assumptions:
- Brain server is running on HOST:PORT and accepts JSON messages line-by-line.
- ESP32 nodes send feature messages of the form:
      {"node": "bedside", "ts": ..., "sensors": {...}}
- Brain sends plan messages of the form:
      {"type": "plan", "node": "...", "state": "...",
       "model_version": N, "config": { ...optional... }}

Adjust HOST/PORT as needed to match your network_server setup.
"""

import json
import socket
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import tkinter as tk
from tkinter import ttk


# ---------------------------------------------------------------------------
# Config: where to connect
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"   # brain server IP (Pi or PC)
PORT = 5000          # must match SERVER_PORT used by ESP32s


# ---------------------------------------------------------------------------
# Shared state model
# ---------------------------------------------------------------------------

@dataclass
class NodeState:
    node: str
    last_seen: float = 0.0
    last_sensors: Dict[str, Any] = field(default_factory=dict)
    last_plan_state: str = ""
    last_plan_time: float = 0.0
    last_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.last_seen <= 0:
            return "never seen"
        age = time.time() - self.last_seen
        if age < 15:
            return "online"
        elif age < 60:
            return f"stale ({int(age)}s)"
        else:
            return f"offline ({int(age)}s)"


@dataclass
class DashboardState:
    nodes: Dict[str, NodeState] = field(default_factory=dict)
    model_version: int = 0
    connected: bool = False
    last_connect_error: str = ""
    last_message_time: float = 0.0
    log: deque = field(default_factory=lambda: deque(maxlen=100))


STATE = DashboardState()
STATE_LOCK = threading.Lock()


def get_or_create_node(node_name: str) -> NodeState:
    with STATE_LOCK:
        if node_name not in STATE.nodes:
            STATE.nodes[node_name] = NodeState(node=node_name)
        return STATE.nodes[node_name]


def log_event(msg: str) -> None:
    with STATE_LOCK:
        ts = time.strftime("%H:%M:%S", time.localtime())
        STATE.log.appendleft(f"[{ts}] {msg}")


# ---------------------------------------------------------------------------
# Network client thread
# ---------------------------------------------------------------------------

def network_loop() -> None:
    global STATE
    while True:
        try:
            log_event("Connecting to brain server...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((HOST, PORT))
            sock.settimeout(None)

            with STATE_LOCK:
                STATE.connected = True
                STATE.last_connect_error = ""

            # Send a hello message like the ESP32s do
            hello = {"type": "hello", "node": "dashboard"}
            sock.sendall((json.dumps(hello) + "\n").encode("utf-8"))
            log_event("Connected and sent dashboard hello")

            buffer = ""
            while True:
                data = sock.recv(4096)
                if not data:
                    raise ConnectionError("Socket closed by server")
                buffer += data.decode("utf-8", errors="ignore")

                # Split on newline; assume one JSON message per line
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    _handle_incoming_json(line)

        except Exception as e:
            with STATE_LOCK:
                STATE.connected = False
                STATE.last_connect_error = str(e)
            log_event(f"Connection error: {e!r}")
            time.sleep(5)  # retry delay


def _handle_incoming_json(line: str) -> None:
    with STATE_LOCK:
        STATE.last_message_time = time.time()

    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        log_event(f"Failed to parse JSON: {line[:80]}...")
        return

    node = msg.get("node", "unknown")

    # Feature message? (has 'sensors')
    if "sensors" in msg:
        ns = get_or_create_node(node)
        sensors = msg.get("sensors", {})
        with STATE_LOCK:
            ns.last_seen = msg.get("ts", time.time())
            ns.last_sensors = sensors
        log_event(f"Feature from {node}: {list(sensors.keys())}")

    # Plan message? (has 'type' == 'plan' or 'state' field)
    if msg.get("type") == "plan" or "state" in msg:
        ns = get_or_create_node(node)
        state = msg.get("state", "")
        cfg = msg.get("config", {})

        with STATE_LOCK:
            ns.last_plan_state = state
            ns.last_plan_time = time.time()
            if isinstance(cfg, dict):
                ns.last_config = cfg.copy()
            mv = msg.get("model_version")
            if isinstance(mv, int):
                STATE.model_version = mv

        log_event(f"Plan for {node}: state={state}, cfg_keys={list(cfg.keys())}")


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------

class DashboardGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Sleep-Env Dashboard")

        self._build_ui()
        self._schedule_update()

    def _build_ui(self) -> None:
        # Top summary frame
        summary_frame = ttk.LabelFrame(self.root, text="Brain Status")
        summary_frame.pack(fill="x", padx=8, pady=4)

        self.lbl_connection = ttk.Label(summary_frame, text="Connection: unknown")
        self.lbl_connection.pack(anchor="w", padx=4, pady=2)

        self.lbl_model_version = ttk.Label(summary_frame, text="Model version: 0")
        self.lbl_model_version.pack(anchor="w", padx=4, pady=2)

        self.lbl_last_msg = ttk.Label(summary_frame, text="Last message: -")
        self.lbl_last_msg.pack(anchor="w", padx=4, pady=2)

        # Node status frame
        nodes_frame = ttk.LabelFrame(self.root, text="Nodes")
        nodes_frame.pack(fill="both", expand=True, padx=8, pady=4)

        # Treeview: one row per node
        cols = ("status", "plan_state", "last_seen", "sensors", "config")
        self.tree = ttk.Treeview(
            nodes_frame,
            columns=cols,
            show="headings",
            height=8,
        )
        self.tree.heading("status", text="Status")
        self.tree.heading("plan_state", text="Plan state")
        self.tree.heading("last_seen", text="Last seen")
        self.tree.heading("sensors", text="Last sensors")
        self.tree.heading("config", text="Last config")

        self.tree.column("status", width=100, anchor="w")
        self.tree.column("plan_state", width=120, anchor="w")
        self.tree.column("last_seen", width=120, anchor="w")
        self.tree.column("sensors", width=300, anchor="w")
        self.tree.column("config", width=300, anchor="w")

        self.tree.pack(fill="both", expand=True, padx=4, pady=4)

        # Event log
        log_frame = ttk.LabelFrame(self.root, text="Event log")
        log_frame.pack(fill="both", expand=False, padx=8, pady=4)

        self.txt_log = tk.Text(log_frame, height=10, wrap="none")
        self.txt_log.pack(fill="both", expand=True, padx=4, pady=4)
        self.txt_log.configure(state="disabled", font=("Consolas", 9))

    def _schedule_update(self) -> None:
        self._update_ui()
        # update every 500 ms
        self.root.after(500, self._schedule_update)

    def _update_ui(self) -> None:
        # Snapshot state under lock
        with STATE_LOCK:
            connected = STATE.connected
            last_err = STATE.last_connect_error
            model_version = STATE.model_version
            last_msg_time = STATE.last_message_time
            nodes_copy = {name: ns for name, ns in STATE.nodes.items()}
            log_lines = list(STATE.log)

        # --- Summary labels ---
        if connected:
            conn_text = "Connection: CONNECTED"
        else:
            if last_err:
                conn_text = f"Connection: DISCONNECTED ({last_err})"
            else:
                conn_text = "Connection: DISCONNECTED"
        self.lbl_connection.configure(text=conn_text)

        self.lbl_model_version.configure(
            text=f"Model version: {model_version}"
        )

        if last_msg_time > 0:
            age = time.time() - last_msg_time
            self.lbl_last_msg.configure(
                text=f"Last message: {int(age)}s ago"
            )
        else:
            self.lbl_last_msg.configure(text="Last message: -")

        # --- Nodes table ---
        # Rebuild tree for simplicity
        for item in self.tree.get_children():
            self.tree.delete(item)

        for name, ns in sorted(nodes_copy.items(), key=lambda kv: kv[0]):
            # Format last seen
            if ns.last_seen > 0:
                t_str = time.strftime("%H:%M:%S", time.localtime(ns.last_seen))
            else:
                t_str = "-"

            # Sensors / config compressed into key:value; only first few
            sensors_str = ", ".join(
                f"{k}={v}"
                for k, v in list(ns.last_sensors.items())[:6]
            )
            config_str = ", ".join(
                f"{k}={v}"
                for k, v in list(ns.last_config.items())[:6]
            )

            self.tree.insert(
                "",
                "end",
                values=(
                    ns.status,
                    ns.last_plan_state or "-",
                    t_str,
                    sensors_str or "-",
                    config_str or "-",
                ),
                text=name,
            )

        # Label row headings as node names (Treeview doesn't show "text" in headings mode,
        # so we can set tags/values, but simplest is just to visually map via sensors column).
        # If you want node name visible, you can add a "node" column too.

        # --- Event log ---
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        for line in log_lines:
            self.txt_log.insert("end", line + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    # Start network thread
    t = threading.Thread(target=network_loop, daemon=True)
    t.start()

    # Start GUI
    root = tk.Tk()
    app = DashboardGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
