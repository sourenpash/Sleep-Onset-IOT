"""
Networked simulation driver for the PC brain.

- Starts a TCP server (via network_server).
- Ingests feature messages from ESP32 nodes (bedside, window, etc.).
- Maintains a rolling ~30-minute history of sensor data.
- Logs all incoming sensor data to CSV for offline analysis.
- Periodically calls compute_sleep_plan(recent_features) to generate a plan.
- Broadcasts the plan back to all connected nodes.
- Handles optional "actuation intent" messages from nodes (for Kasa stubs).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import network_server
from kasa_control_stub import set_device_state
from sleep_model_stub import compute_sleep_plan

# How much history (in seconds) to keep in memory for the model
HISTORY_SECONDS = 30 * 60  # 30 minutes

# How often to compute and broadcast a new plan
PLAN_INTERVAL_SEC = 10

# Logging config
LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)
LOG_FILE = LOG_PATH / "sensor_log.csv"


def main() -> None:
    recent_features: List[Dict[str, Any]] = []
    last_plan_time = 0.0

    network_server.start_server()
    print("[BRAIN] Server started; waiting for ESP32 clients...")

    try:
        while True:
            # Poll for a new message from any connected node
            message = network_server.get_next_message(timeout=0.1)
            if message:
                _handle_message(message, recent_features)

            # Periodically compute and broadcast a plan
            now = time.time()
            if now - last_plan_time >= PLAN_INTERVAL_SEC:
                if recent_features:
                    plan = compute_sleep_plan(recent_features)
                else:
                    # No data yet; ask the stub for a default plan
                    plan = compute_sleep_plan([])

                print("[BRAIN] Broadcasting plan:")
                print(json.dumps(plan, indent=2))

                network_server.broadcast(plan)
                last_plan_time = now
    except KeyboardInterrupt:
        print("[BRAIN] Shutting down")


def _handle_message(message: Dict[str, Any], recent_features: List[Dict[str, Any]]) -> None:
    """Process incoming feature or actuation messages."""
    if "sensors" in message:
        _handle_feature(message, recent_features)
    elif "desired" in message:
        _handle_actuation_intent(message)
    else:
        print(f"[BRAIN] Unrecognized message: {message}")


def _handle_feature(message: Dict[str, Any], recent_features: List[Dict[str, Any]]) -> None:
    """
    Handle an incoming sensor feature message from a node.

    We:
    - Normalize/attach a real timestamp.
    - Append to the rolling 30-minute history.
    - Log to CSV for offline analysis.
    """
    # Normalize timestamp: if 'ts' looks like a real Unix timestamp (seconds), use it;
    # otherwise, replace with the current wall-clock time.
    raw_ts = message.get("ts")
    if isinstance(raw_ts, (int, float)) and raw_ts > 10_000_000:
        msg_time = float(raw_ts)
    else:
        msg_time = time.time()
        message["ts"] = msg_time

    # Append to in-memory history
    recent_features.append(message)

    # Drop old entries outside the sliding window
    cutoff = msg_time - HISTORY_SECONDS
    while recent_features and recent_features[0].get("ts", 0.0) < cutoff:
        recent_features.pop(0)

    # Log to CSV
    _log_feature_to_csv(message)

    node = message.get("node", "unknown")
    print(f"[BRAIN] Feature received from {node} at ts={message['ts']}")


def _log_feature_to_csv(message: Dict[str, Any]) -> None:
    """Append a flattened version of the feature message to sensor_log.csv."""
    node = message.get("node", "unknown")
    ts = message.get("ts", time.time())
    sensors = message.get("sensors", {})

    # Flatten sensors into s_<name> columns
    row: Dict[str, Any] = {
        "ts": ts,
        "node": node,
    }
    for k, v in sensors.items():
        row[f"s_{k}"] = v

    file_exists = LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _handle_actuation_intent(message: Dict[str, Any]) -> None:
    """
    Handle an actuation intent message from a node.

    Example message:
    {
        "node": "bedside",
        "desired": {
            "bedside_lamp": "off",
            "humidifier": "on"
        }
    }
    """
    desired = message.get("desired", {})
    node = message.get("node", "unknown")
    print(f"[BRAIN] Actuation intent from {node}")
    for name, state in desired.items():
        set_device_state(name, state)


if __name__ == "__main__":
    main()
