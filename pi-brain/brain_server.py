

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests  

import network_server
from kasa_control_stub import set_device_state
from sleep_model_stub import compute_sleep_plan, train_model_from_csv

HISTORY_SECONDS = 30 * 60  

PLAN_INTERVAL_SEC = 10

WEATHER_UPDATE_INTERVAL_SEC = 10 * 60  

FALLBACK_LAT = 43.65
FALLBACK_LON = -79.38

LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)
LOG_FILE = LOG_PATH / "sensor_log.csv"

# Training data + model metadata
TRAIN_LOG_FILE = LOG_PATH / "training_data.csv"
MODEL_META_PATH = LOG_PATH / "model_meta.json"
MODEL_EVENT_LOG = LOG_PATH / "model_events.csv"

PLAN_LOG_FILE = LOG_PATH / "plan_log.jsonl"

MODEL_VERSION: int = 0
LAST_TRAINED_SAMPLES: int = 0
MIN_SAMPLES_FOR_RETRAIN: int = 50  

TRAIN_FIELDNAMES = [
    "ts_label",
    "label",
    "label_conf",

    # Window node features
    "temp_win_c",
    "hum_win_pct",
    "light_win_lux",

    # Bedside node features
    "temp_bed_c",
    "hum_bed_pct",
    "lux_bed",
    "light_bed_lux",

    # Door node features
    "temp_door_c",
    "hum_door_pct",
    "mic_v",
    "light_door_v",

    # Outdoor weather
    "temp_outdoor_c",
    "hum_outdoor_pct",
]


# --------------------------------------------------------------------------

def _load_model_meta() -> None:
    
    global MODEL_VERSION, LAST_TRAINED_SAMPLES
    if MODEL_META_PATH.exists():
        try:
            data = json.loads(MODEL_META_PATH.read_text())
            MODEL_VERSION = int(data.get("version", 0))
            LAST_TRAINED_SAMPLES = int(data.get("trained_on_samples", 0))
            print(
                f"[MODEL] Loaded meta: version={MODEL_VERSION}, "
                f"trained_on_samples={LAST_TRAINED_SAMPLES}"
            )
        except Exception as e:
            print(f"[MODEL] Failed to load model_meta.json: {e}")
            MODEL_VERSION = 0
            LAST_TRAINED_SAMPLES = 0
    else:
        MODEL_VERSION = 0
        LAST_TRAINED_SAMPLES = 0
        print("[MODEL] No model_meta.json; starting at version=0.")


def _save_model_meta() -> None:
    
    meta = {
        "version": MODEL_VERSION,
        "trained_on_samples": LAST_TRAINED_SAMPLES,
        "updated_at": time.time(),
    }
    MODEL_META_PATH.write_text(json.dumps(meta))


def _log_model_event(event_type: str) -> None:
    
    file_exists = MODEL_EVENT_LOG.exists()
    with MODEL_EVENT_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ts", "event", "version", "trained_on_samples"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "ts": time.time(),
                "event": event_type,
                "version": MODEL_VERSION,
                "trained_on_samples": LAST_TRAINED_SAMPLES,
            }
        )


def _count_training_samples() -> int:
    
    if not TRAIN_LOG_FILE.exists():
        return 0
    n = 0
    with TRAIN_LOG_FILE.open("r", newline="") as f:
        reader = csv.reader(f)
        # skip header if present
        header_seen = False
        for row in reader:
            if not header_seen:
                header_seen = True
                continue
            if row:
                n += 1
    return n


def _maybe_retrain_model_from_training_csv() -> None:
   
    global MODEL_VERSION, LAST_TRAINED_SAMPLES

    n_samples = _count_training_samples()
    if n_samples < MIN_SAMPLES_FOR_RETRAIN:
        return
    if n_samples <= LAST_TRAINED_SAMPLES:
        return

    print(
        f"[MODEL] Retraining from training_data.csv: "
        f"n_samples={n_samples}, prev_trained={LAST_TRAINED_SAMPLES}"
    )

    model_path = LOG_PATH / "sleep_model.json"
    ok = train_model_from_csv(TRAIN_LOG_FILE, model_out_path=model_path)

    if not ok:
        print("[MODEL] Training failed; keeping old model/version.")
        return

    LAST_TRAINED_SAMPLES = n_samples
    MODEL_VERSION += 1
    _save_model_meta()
    _log_model_event("retrained")
    print(f"[MODEL] RETRAINED → version={MODEL_VERSION}")



def main() -> None:
    recent_features: List[Dict[str, Any]] = []
    last_plan_time = 0.0

    # Outdoor weather state
    last_weather_update = 0.0
    lat_lon: Optional[Tuple[float, float]] = None

    _load_model_meta()

    network_server.start_server()
    print("[BRAIN] Server started; waiting for ESP32 clients...")

    try:
        while True:
            # Poll for a new message from any connected node
            message = network_server.get_next_message(timeout=0.1)
            if message:
                _handle_message(message, recent_features)

            now = time.time()

            # Periodically refresh outdoor weather and inject as a "weather" node
            if now - last_weather_update >= WEATHER_UPDATE_INTERVAL_SEC:
                if lat_lon is None:
                    lat_lon = _resolve_location_from_ip()
                lat, lon = lat_lon
                weather_sensors = _fetch_outdoor_weather(lat, lon)
                if weather_sensors is not None:
                    weather_msg = {
                        "node": "weather",
                        "ts": time.time(),
                        "sensors": weather_sensors,
                    }
                    _handle_feature(weather_msg, recent_features)
                    print(
                        f"[BRAIN] Updated outdoor weather: "
                        f"T={weather_sensors.get('temp_outdoor_c')} °C, "
                        f"RH={weather_sensors.get('hum_outdoor_pct')} %"
                    )
                last_weather_update = now

            # Periodically compute and broadcast a plan
            if now - last_plan_time >= PLAN_INTERVAL_SEC:
                plan = compute_sleep_plan(recent_features, model_version=MODEL_VERSION)

                print("[BRAIN] Broadcasting plan:")
                print(json.dumps(plan, indent=2))

                # Log plan to JSONL for dashboard
                try:
                    PLAN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
                    with PLAN_LOG_FILE.open("a") as pf:
                        pf.write(json.dumps(plan) + "\n")
                except Exception as e:
                    print(f"[BRAIN] Failed to log plan: {e}")

                network_server.broadcast(plan)
                last_plan_time = now

    except KeyboardInterrupt:
        print("[BRAIN] Shutting down")


def _resolve_location_from_ip() -> Tuple[float, float]:
   
    try:
        print("[BRAIN] Resolving location from IP...")
        resp = requests.get("http://ip-api.com/json/", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            lat = data.get("lat")
            lon = data.get("lon")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                print(f"[BRAIN] Location from IP: lat={lat}, lon={lon}")
                return float(lat), float(lon)
            else:
                print("[BRAIN] IP API did not return numeric lat/lon, using fallback.")
        else:
            print(f"[BRAIN] IP API error status={resp.status_code}, using fallback.")
    except Exception as e:
        print(f"[BRAIN] IP geolocation failed: {e}. Using fallback.")

    print(
        f"[BRAIN] Fallback location: Toronto, ON (lat={FALLBACK_LAT}, lon={FALLBACK_LON})"
    )
    return FALLBACK_LAT, FALLBACK_LON




def _fetch_outdoor_weather(lat: float, lon: float) -> Optional[Dict[str, float]]:

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m",
        "timezone": "auto",
    }

    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200:
            print(f"[BRAIN] Open-Meteo HTTP {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        current = data.get("current", {})

        temp_c = current.get("temperature_2m")
        rh = current.get("relative_humidity_2m")

        if temp_c is None or rh is None:
            print("[BRAIN] Open-Meteo response missing temperature or humidity.")
            return None

        return {
            "temp_outdoor_c": float(temp_c),
            "hum_outdoor_pct": float(rh),
        }

    except Exception as e:
        print(f"[BRAIN] Failed to fetch outdoor weather: {e}")
        return None




def _handle_message(message: Dict[str, Any], recent_features: List[Dict[str, Any]]) -> None:
    """Process incoming feature or actuation messages."""
    if "sensors" in message:
        _handle_feature(message, recent_features)
    elif "desired" in message:
        _handle_actuation_intent(message)
    else:
        print(f"[BRAIN] Unrecognized message: {message}")


def _handle_feature(message: Dict[str, Any], recent_features: List[Dict[str, Any]]) -> None:

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

    # Log to sensor_log.csv
    _log_feature_to_csv(message)

    node = message.get("node", "unknown")
    print(f"[BRAIN] Feature received from {node} at ts={message['ts']}")

    # If this is a camera label, also log to training_data.csv and maybe retrain
    if node == "camera":
        _handle_camera_label(message, recent_features)


def _log_feature_to_csv(message: Dict[str, Any]) -> None:
   
    node = message.get("node", "unknown")
    ts = message.get("ts", time.time())
    sensors = message.get("sensors", {})

    row: Dict[str, Any] = {
        "ts": ts,
        "node": node,
    }
    for k, v in sensors.items():
        row[f"s_{k}"] = v

    if not LOG_FILE.exists():
        # First time: write file with this row's keys as header
        fieldnames = sorted(row.keys())
        with LOG_FILE.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return

    # If file already exists, we may need to add new columns
    # 1) Read existing header and all rows
    with LOG_FILE.open("r", newline="") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = reader.fieldnames or []
        existing_rows = list(reader)

    # 2) Compute union of existing fieldnames and new row keys
    new_fieldnames_set = set(existing_fieldnames) | set(row.keys())
    # Preserve order: ts, node first; then the rest sorted
    base_order = ["ts", "node"]
    rest = sorted(fn for fn in new_fieldnames_set if fn not in base_order)
    fieldnames = base_order + rest

    # 3) Rewrite entire file with new header if needed
    with LOG_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write old rows, filling missing keys with empty string
        for old in existing_rows:
            out = {fn: old.get(fn, "") for fn in fieldnames}
            writer.writerow(out)

        # Write new row with any missing columns filled
        out_new = {fn: row.get(fn, "") for fn in fieldnames}
        writer.writerow(out_new)


# --------------------------------------------------------------------------
# Training data: camera labels + recent sensor context
# --------------------------------------------------------------------------

def _latest_feature_for_node(
    recent_features: List[Dict[str, Any]],
    node_name: str,
    max_age_sec: float,
    ref_ts: float,
) -> Optional[Dict[str, Any]]:
    """Return the latest message for a given node within max_age_sec of ref_ts."""
    for msg in reversed(recent_features):
        if msg.get("node") != node_name:
            continue
        ts = msg.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        if ref_ts - ts > max_age_sec:
            continue
        return msg
    return None


def _build_training_example_from_context(
    camera_msg: Dict[str, Any],
    recent_features: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:

    sensors = camera_msg.get("sensors", {})
    ts_label = float(camera_msg.get("ts", time.time()))

    # ---------- FLEXIBLE LABEL / CONF EXTRACTION ----------
    # 1) Try common keys explicitly
    label = (
        sensors.get("state")
        or sensors.get("label")
        or sensors.get("camera_state")
        or sensors.get("class")
        or sensors.get("pose")
        or sensors.get("sleep_state")
    )
    conf = (
        sensors.get("conf")
        or sensors.get("confidence")
        or sensors.get("prob")
        or sensors.get("p")
        or sensors.get("score")
    )

    # 2) If still no label, scan all keys for something that LOOKS like a state/label
    if label is None:
        for k, v in sensors.items():
            if not isinstance(v, str):
                continue
            kl = k.lower()
            if any(sub in kl for sub in ["state", "label", "pose", "sleep"]):
                label = v
                break

    if label is None:
        print("[TRAIN] Camera message missing label-like field in sensors; got:", sensors)
        return None

    try:
        label_conf = float(conf) if conf is not None else float("nan")
    except Exception:
        label_conf = float("nan")


    try:
        label_conf = float(conf) if conf is not None else float("nan")
    except Exception:
        label_conf = float("nan")

    # Look back up to 60 seconds for context
    MAX_AGE_SEC = 60.0*30

    win_msg = _latest_feature_for_node(recent_features, "window", MAX_AGE_SEC, ts_label)
    bed_msg = _latest_feature_for_node(recent_features, "bedside", MAX_AGE_SEC, ts_label)
    door_msg = _latest_feature_for_node(recent_features, "door", MAX_AGE_SEC, ts_label)
    weather_msg = _latest_feature_for_node(recent_features, "weather", MAX_AGE_SEC, ts_label)

    def sensor_val(msg: Optional[Dict[str, Any]], key: str) -> float:
        if msg is None:
            return float("nan")
        s = msg.get("sensors", {})
        v = s.get(key)
        try:
            return float(v)
        except Exception:
            return float("nan")

    row_full: Dict[str, Any] = {
        "ts_label": ts_label,
        "label": label,
        "label_conf": label_conf,

        # Window
        "temp_win_c": sensor_val(win_msg, "temp_win_c"),
        "hum_win_pct": sensor_val(win_msg, "hum_win_pct"),
        "light_win_lux": sensor_val(win_msg, "light_win_lux"),

        # Bedside
        "temp_bed_c": sensor_val(bed_msg, "temp_bed_c"),
        "hum_bed_pct": sensor_val(bed_msg, "hum_bed_pct"),
        "lux_bed": sensor_val(bed_msg, "lux_bed"),
        "light_bed_lux": sensor_val(bed_msg, "light_bed_lux"),

        # Door
        "temp_door_c": sensor_val(door_msg, "temp_door_c"),
        "hum_door_pct": sensor_val(door_msg, "hum_door_pct"),
        "mic_v": sensor_val(door_msg, "mic_v"),
        "light_door_v": sensor_val(door_msg, "light_door_v"),

        # Weather
        "temp_outdoor_c": sensor_val(weather_msg, "temp_outdoor_c"),
        "hum_outdoor_pct": sensor_val(weather_msg, "hum_outdoor_pct"),
    }

    return row_full


def _append_training_row(row: Dict[str, Any]) -> None:
    """Append a single training example to training_data.csv."""
    file_exists = TRAIN_LOG_FILE.exists()

    # Ensure all expected keys present, fill missing with NaN / empty
    out_row: Dict[str, Any] = {}
    for key in TRAIN_FIELDNAMES:
        val = row.get(key, float("nan") if key != "label" else "")
        out_row[key] = val

    with TRAIN_LOG_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(out_row)


def _handle_camera_label(
    camera_msg: Dict[str, Any],
    recent_features: List[Dict[str, Any]],
) -> None:
    """Handle camera label: build training row, append CSV, maybe retrain."""
    row = _build_training_example_from_context(camera_msg, recent_features)
    if row is None:
        return

    _append_training_row(row)
    print("[TRAIN] Appended training example to training_data.csv")

    # Check if it's time to retrain
    _maybe_retrain_model_from_training_csv()


# --------------------------------------------------------------------------
# Actuation
# --------------------------------------------------------------------------

def _handle_actuation_intent(message: Dict[str, Any]) -> None:

    desired = message.get("desired", {})
    node = message.get("node", "unknown")
    print(f"[BRAIN] Actuation intent from {node}")
    for name, state in desired.items():
        set_device_state(name, state)


if __name__ == "__main__":
    main()
