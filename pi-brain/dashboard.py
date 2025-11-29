"""
dashboard.py

Live dashboard for the sleep-env project.

Tabs:
  1) Overview
     - Current global sleep state / probabilities.
     - Latest per-node sensor values (window / bedside / door / camera / weather).
  2) Model Internals
     - Features fed into logistic regression (current + ODE-predicted future).
     - Logistic regression output p_sleep_model.
     - ODE parameters (tau_temp_s, tau_hum_s, cooldown_time_s, targets).
     - Training/model status (version, # training samples).

Data sources (all written by brain_server.py):
  - logs/sensor_log.csv       : flattened sensor features by node.
  - logs/plan_log.jsonl       : one JSON sleep plan per line.
  - logs/model_meta.json      : model_version + trained_on_samples.
  - logs/training_data.csv    : camera-labelled training rows.

This dashboard is read-only: it does NOT send any commands back to the ESP32s
yet. Calibration / remote control could be built on top of this.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import PySimpleGUI as sg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
SENSOR_LOG_FILE = LOG_DIR / "sensor_log.csv"
PLAN_LOG_FILE = LOG_DIR / "plan_log.jsonl"
MODEL_META_FILE = LOG_DIR / "model_meta.json"
TRAIN_DATA_FILE = LOG_DIR / "training_data.csv"

REFRESH_INTERVAL_SEC = 2.0

# Same feature names as sleep_model_stub.py
FEATURE_NAMES = [
    "temp_win_c",
    "hum_win_pct",
    "light_win_lux",
    "temp_bed_c",
    "hum_bed_pct",
    "lux_bed",
    "light_bed_lux",
    "temp_door_c",
    "hum_door_pct",
    "mic_v",
    "light_door_v",
    "temp_outdoor_c",
    "hum_outdoor_pct",
]

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "temp_win_c": "Window temperature (°C)",
    "hum_win_pct": "Window humidity (%)",
    "light_win_lux": "Window light (lux)",
    "temp_bed_c": "Bedside temperature (°C)",
    "hum_bed_pct": "Bedside humidity (%)",
    "lux_bed": "Bedside light level (lux)",
    "light_bed_lux": "Bedside light (lux, alt key)",
    "temp_door_c": "Door area temperature (°C)",
    "hum_door_pct": "Door area humidity (%)",
    "mic_v": "Door microphone (V)",
    "light_door_v": "Door area light (V)",
    "temp_outdoor_c": "Outdoor temperature (°C)",
    "hum_outdoor_pct": "Outdoor humidity (%)",
}

# ---------------------------------------------------------------------------
# Helpers to read logs
# ---------------------------------------------------------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def read_latest_sensor_rows() -> Dict[str, Dict[str, Any]]:
    """
    Read sensor_log.csv and return latest row per node.

    Returns:
        {
          "bedside": {"ts": ..., "sensors": {"temp_bed_c": ..., ...}},
          "window": {...},
          "door": {...},
          "camera": {...},
          "weather": {...},
          ...
        }
    """
    if not SENSOR_LOG_FILE.exists():
        return {}

    latest_by_node: Dict[str, Dict[str, Any]] = {}

    with SENSOR_LOG_FILE.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = row.get("node", "unknown")
            ts = _safe_float(row.get("ts"))
            if ts is None:
                continue
            # Always keep the newest row per node
            prev = latest_by_node.get(node)
            if prev is not None and _safe_float(prev.get("ts")) is not None:
                if ts <= _safe_float(prev.get("ts")):
                    continue

            # Build sensors dict from s_<name> columns
            sensors: Dict[str, Any] = {}
            for k, v in row.items():
                if not isinstance(k, str):
                    continue
                if not k.startswith("s_"):
                    continue
                name = k[2:]  # drop "s_"
                sensors[name] = _safe_float(v)

            latest_by_node[node] = {
                "ts": ts,
                "sensors": sensors,
            }

    return latest_by_node


def read_latest_plan() -> Optional[Dict[str, Any]]:
    """Return the last JSON object from plan_log.jsonl, or None."""
    if not PLAN_LOG_FILE.exists():
        return None

    last_line = None
    with PLAN_LOG_FILE.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last_line = line

    if last_line is None:
        return None

    try:
        return json.loads(last_line)
    except Exception:
        return None


def read_model_meta() -> Dict[str, Any]:
    """Load model_meta.json, or defaults if missing."""
    if not MODEL_META_FILE.exists():
        return {
            "version": 0,
            "trained_on_samples": 0,
            "updated_at": None,
        }
    try:
        data = json.loads(MODEL_META_FILE.read_text())
    except Exception:
        data = {}
    return {
        "version": int(data.get("version", 0)),
        "trained_on_samples": int(data.get("trained_on_samples", 0)),
        "updated_at": data.get("updated_at"),
    }


def read_training_stats() -> Dict[str, Any]:
    """
    Very small summary of training_data.csv:
      - n_rows
      - last_label
      - last_ts
    """
    if not TRAIN_DATA_FILE.exists():
        return {
            "n_rows": 0,
            "last_label": None,
            "last_ts": None,
        }

    n = 0
    last_label = None
    last_ts = None

    with TRAIN_DATA_FILE.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            last_label = row.get("label", last_label)
            ts_raw = row.get("ts_label")
            ts_val = _safe_float(ts_raw)
            if ts_val is not None:
                last_ts = ts_val

    return {
        "n_rows": n,
        "last_label": last_label,
        "last_ts": last_ts,
    }

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def make_overview_tab_layout() -> List[List[Any]]:
    # Global state summary
    global_frame = [
        [
            sg.Text("Current time:", size=(14, 1)),
            sg.Text("?", key="-OV_NOW-", size=(24, 1)),
        ],
        [
            sg.Text("Global State:", size=(14, 1)),
            sg.Text("?", key="-OV_GLOBAL_STATE-", size=(12, 1), text_color="yellow"),
        ],
        [
            sg.Text("p_sleep (final):", size=(14, 1)),
            sg.Text("?", key="-OV_P_SLEEP-", size=(12, 1)),
        ],
        [
            sg.Text("p_sleep_model:", size=(14, 1)),
            sg.Text("?", key="-OV_P_SLEEP_MODEL-", size=(12, 1)),
        ],
        [
            sg.Text("Camera label:", size=(14, 1)),
            sg.Text("?", key="-OV_CAM_LABEL-", size=(20, 1)),
        ],
        [
            sg.Text("Camera conf:", size=(14, 1)),
            sg.Text("?", key="-OV_CAM_CONF-", size=(12, 1)),
        ],
    ]

    node_frame = [
        [sg.Text("Window node")],
        [sg.Text("Temp in (°C):"), sg.Text("?", key="-OV_WIN_TEMP_IN-")],
        [sg.Text("Hum in (%):"), sg.Text("?", key="-OV_WIN_HUM_IN-")],
        [sg.Text("Temp out (°C):"), sg.Text("?", key="-OV_WIN_TEMP_OUT-")],
        [sg.Text("Hum out (%):"), sg.Text("?", key="-OV_WIN_HUM_OUT-")],
        [sg.Text("Light (lux):"), sg.Text("?", key="-OV_WIN_LIGHT-")],
        [sg.HorizontalSeparator()],
        [sg.Text("Bedside node")],
        [sg.Text("Temp (°C):"), sg.Text("?", key="-OV_BED_TEMP-")],
        [sg.Text("Hum (%):"), sg.Text("?", key="-OV_BED_HUM-")],
        [sg.Text("Light (lux):"), sg.Text("?", key="-OV_BED_LIGHT-")],
        [sg.HorizontalSeparator()],
        [sg.Text("Door node")],
        [sg.Text("Temp (°C):"), sg.Text("?", key="-OV_DOOR_TEMP-")],
        [sg.Text("Hum (%):"), sg.Text("?", key="-OV_DOOR_HUM-")],
        [sg.Text("Mic (V):"), sg.Text("?", key="-OV_DOOR_MIC-")],
        [sg.Text("Light (V):"), sg.Text("?", key="-OV_DOOR_LIGHT-")],
    ]

    env_frame = [
        [sg.Text("Outdoor (weather node)")],
        [sg.Text("Temp (°C):"), sg.Text("?", key="-OV_OUT_TEMP-")],
        [sg.Text("Hum (%):"), sg.Text("?", key="-OV_OUT_HUM-")],
        [sg.HorizontalSeparator()],
        [sg.Text("Camera node")],
        [sg.Text("Raw sensors dict:")],
        [
            sg.Multiline(
                "",
                key="-OV_CAM_RAW-",
                size=(30, 5),
                disabled=True,
            )
        ],
    ]

    layout = [
        [
            sg.Frame("Sleep plan", global_frame, pad=(5, 5), expand_x=True),
        ],
        [
            sg.Frame(
                "Nodes (window / bedside / door)",
                node_frame,
                pad=(5, 5),
            ),
            sg.Frame("Weather + camera", env_frame, pad=(5, 5)),
        ],
    ]
    return layout


def make_model_tab_layout() -> List[List[Any]]:
    # Features list
    feature_rows = []
    for name in FEATURE_NAMES:
        disp = FEATURE_DISPLAY_NAMES.get(name, name)
        feature_rows.append(
            [
                sg.Text(disp + ":", size=(28, 1)),
                sg.Text("?", key=f"-MOD_FEAT_NOW_{name}-", size=(12, 1)),
                sg.Text("→ future:", size=(8, 1)),
                sg.Text("?", key=f"-MOD_FEAT_FUT_{name}-", size=(12, 1)),
            ]
        )

    features_frame = [
        [sg.Text("Features into logistic regression")],
        [
            sg.Column(
                feature_rows,
                scrollable=True,
                vertical_scroll_only=True,
                size=(500, 260),
            )
        ],
    ]

    ode_frame = [
        [sg.Text("ODE dynamics (self-tuning)")],
        [
            sg.Text("tau_temp_s:", size=(14, 1)),
            sg.Text("?", key="-MOD_TAU_TEMP-", size=(12, 1)),
        ],
        [
            sg.Text("tau_hum_s:", size=(14, 1)),
            sg.Text("?", key="-MOD_TAU_HUM-", size=(12, 1)),
        ],
        [
            sg.Text("cooldown_time_s:", size=(14, 1)),
            sg.Text("?", key="-MOD_COOLDOWN-", size=(12, 1)),
        ],
        [
            sg.Text("Target temp (°C):", size=(14, 1)),
            sg.Text("?", key="-MOD_TARGET_TEMP-", size=(12, 1)),
        ],
        [
            sg.Text("Target hum (%):", size=(14, 1)),
            sg.Text("?", key="-MOD_TARGET_HUM-", size=(12, 1)),
        ],
    ]

    model_frame = [
        [sg.Text("Logistic regression status")],
        [
            sg.Text("p_sleep_model:", size=(14, 1)),
            sg.Text("?", key="-MOD_P_SLEEP_MODEL-", size=(12, 1)),
        ],
        [
            sg.Text("Model version:", size=(14, 1)),
            sg.Text("?", key="-MOD_MODEL_VERSION-", size=(12, 1)),
        ],
        [
            sg.Text("# training rows:", size=(14, 1)),
            sg.Text("?", key="-MOD_TRAIN_ROWS-", size=(12, 1)),
        ],
        [
            sg.Text("Last label:", size=(14, 1)),
            sg.Text("?", key="-MOD_LAST_LABEL-", size=(16, 1)),
        ],
    ]

    layout = [
        [
            sg.Frame(
                "Features (now vs future)",
                features_frame,
                pad=(5, 5),
                expand_x=True,
                expand_y=True,
            ),
            sg.Column(
                [
                    [sg.Frame("ODE dynamics", ode_frame, pad=(5, 5))],
                    [sg.Frame("Model / training", model_frame, pad=(5, 5))],
                ]
            ),
        ],
    ]
    return layout

# ---------------------------------------------------------------------------
# Main update logic
# ---------------------------------------------------------------------------

def format_val(v: Any, digits: int = 2) -> str:
    if v is None:
        return "–"
    try:
        f = float(v)
    except Exception:
        return str(v)
    return f"{f:.{digits}f}"


def update_overview_tab(
    window: sg.Window,
    nodes: Dict[str, Dict[str, Any]],
    plan: Optional[Dict[str, Any]],
) -> None:
    # Current time
    now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    window["-OV_NOW-"].update(now_str)

    # Plan summary
    if plan is None:
        window["-OV_GLOBAL_STATE-"].update("NO PLAN")
        window["-OV_P_SLEEP-"].update("–")
        window["-OV_P_SLEEP_MODEL-"].update("–")
        window["-OV_CAM_LABEL-"].update("–")
        window["-OV_CAM_CONF-"].update("–")
    else:
        window["-OV_GLOBAL_STATE-"].update(plan.get("global_state", "??"))
        window["-OV_P_SLEEP-"].update(format_val(plan.get("p_sleep"), 3))
        window["-OV_P_SLEEP_MODEL-"].update(
            format_val(plan.get("p_sleep_model"), 3)
        )
        window["-OV_CAM_LABEL-"].update(str(plan.get("camera_label")))
        window["-OV_CAM_CONF-"].update(format_val(plan.get("camera_conf"), 3))

    # Helper to pull from nodes dict
    def get_sensor(node: str, key: str) -> Optional[float]:
        info = nodes.get(node)
        if not info:
            return None
        return info.get("sensors", {}).get(key)

    # ----- Window node: inside + outside -----
    # Inside temp/hum: prefer *_in_ keys, fall back to original names
    temp_win_in = get_sensor("window", "temp_win_in_c")
    if temp_win_in is None:
        temp_win_in = get_sensor("window", "temp_win_c")

    hum_win_in = get_sensor("window", "hum_win_in_pct")
    if hum_win_in is None:
        hum_win_in = get_sensor("window", "hum_win_pct")

    # Outside temp/hum from window node (if firmware sends them)
    temp_win_out = get_sensor("window", "temp_win_out_c")
    hum_win_out = get_sensor("window", "hum_win_out_pct")

    window["-OV_WIN_TEMP_IN-"].update(format_val(temp_win_in))
    window["-OV_WIN_HUM_IN-"].update(format_val(hum_win_in))
    window["-OV_WIN_TEMP_OUT-"].update(format_val(temp_win_out))
    window["-OV_WIN_HUM_OUT-"].update(format_val(hum_win_out))
    window["-OV_WIN_LIGHT-"].update(
        format_val(get_sensor("window", "light_win_lux"))
    )

    # ----- Bedside -----
    window["-OV_BED_TEMP-"].update(
        format_val(get_sensor("bedside", "temp_bed_c"))
    )
    window["-OV_BED_HUM-"].update(
        format_val(get_sensor("bedside", "hum_bed_pct"))
    )
    # try both lux_bed and light_bed_lux
    light_bed = get_sensor("bedside", "lux_bed")
    if light_bed is None:
        light_bed = get_sensor("bedside", "light_bed_lux")
    window["-OV_BED_LIGHT-"].update(format_val(light_bed))

    # ----- Door -----
    window["-OV_DOOR_TEMP-"].update(
        format_val(get_sensor("door", "temp_door_c"))
    )
    window["-OV_DOOR_HUM-"].update(
        format_val(get_sensor("door", "hum_door_pct"))
    )
    window["-OV_DOOR_MIC-"].update(
        format_val(get_sensor("door", "mic_v"), 3)
    )
    window["-OV_DOOR_LIGHT-"].update(
        format_val(get_sensor("door", "light_door_v"), 3)
    )

    # ----- Weather -----
    window["-OV_OUT_TEMP-"].update(
        format_val(get_sensor("weather", "temp_outdoor_c"))
    )
    window["-OV_OUT_HUM-"].update(
        format_val(get_sensor("weather", "hum_outdoor_pct"))
    )

    # ----- Camera raw sensors -----
    cam_info = nodes.get("camera")
    if cam_info:
        raw = cam_info.get("sensors", {})
        window["-OV_CAM_RAW-"].update(json.dumps(raw, indent=2))
    else:
        window["-OV_CAM_RAW-"].update("")


def update_model_tab(
    window: sg.Window,
    plan: Optional[Dict[str, Any]],
    model_meta: Dict[str, Any],
    train_stats: Dict[str, Any],
) -> None:
    feats_now = {}
    feats_future = {}
    p_sleep_model = None
    tau_temp = tau_hum = cooldown = None
    target_temp = target_hum = None

    if plan is not None:
        feats_now = plan.get("features_now", {}) or {}
        feats_future = plan.get("features_future", {}) or {}
        p_sleep_model = plan.get("p_sleep_model")
        tau_temp = plan.get("tau_temp_s")
        tau_hum = plan.get("tau_hum_s")
        cooldown = plan.get("cooldown_time_s")
        target_temp = plan.get("target_temp_c")
        target_hum = plan.get("target_hum_pct")

    # Features
    for name in FEATURE_NAMES:
        v_now = feats_now.get(name)
        v_fut = feats_future.get(name)
        window[f"-MOD_FEAT_NOW_{name}-"].update(format_val(v_now))
        window[f"-MOD_FEAT_FUT_{name}-"].update(format_val(v_fut))

    # ODE
    window["-MOD_TAU_TEMP-"].update(format_val(tau_temp))
    window["-MOD_TAU_HUM-"].update(format_val(tau_hum))
    window["-MOD_COOLDOWN-"].update(format_val(cooldown, 1))
    window["-MOD_TARGET_TEMP-"].update(format_val(target_temp, 1))
    window["-MOD_TARGET_HUM-"].update(format_val(target_hum, 1))

    # Model / training
    window["-MOD_P_SLEEP_MODEL-"].update(format_val(p_sleep_model, 3))
    window["-MOD_MODEL_VERSION-"].update(str(model_meta.get("version", 0)))
    window["-MOD_TRAIN_ROWS-"].update(str(train_stats.get("n_rows", 0)))
    window["-MOD_LAST_LABEL-"].update(str(train_stats.get("last_label")))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sg.theme("DarkBlue3")

    layout = [
        [
            sg.TabGroup(
                [
                    [
                        sg.Tab(
                            "Overview",
                            make_overview_tab_layout(),
                            key="-TAB_OV-",
                        ),
                        sg.Tab(
                            "Model Internals",
                            make_model_tab_layout(),
                            key="-TAB_MOD-",
                        ),
                    ]
                ],
                expand_x=True,
                expand_y=True,
            )
        ],
        [sg.Button("Exit")],
    ]

    window = sg.Window(
        "Sleep Env Dashboard",
        layout,
        resizable=True,
        finalize=True,
    )

    last_refresh = 0.0

    while True:
        event, values = window.read(timeout=200)
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        now = time.time()
        if now - last_refresh >= REFRESH_INTERVAL_SEC:
            last_refresh = now

            nodes = read_latest_sensor_rows()
            plan = read_latest_plan()
            model_meta = read_model_meta()
            train_stats = read_training_stats()

            update_overview_tab(window, nodes, plan)
            update_model_tab(window, plan, model_meta, train_stats)

    window.close()


if __name__ == "__main__":
    main()
