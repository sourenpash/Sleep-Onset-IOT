"""
sleep_model_stub.py

Unified sleep + environment model:

- Logistic regression hazard model:
    p_sleep_soon = P(sleep within BASE_HORIZON_MIN minutes | features)

- First-order ODE environment model for:
    - temperature at bed
    - humidity at bed
  with optional online tuning of k_temp, k_hum.

- compute_sleep_plan(recent_features):
    Used by brain_server. Combines:
      * logistic hazard
      * simple heuristic state (AWAKE / WINDING_DOWN / ASLEEP)
      * environment time-to-target estimates
      * simple target setpoints

- train_on_night_csv(csv_path):
    Offline training from a CSV log (e.g. fake_night.csv or real night logs).

- Optional ONLINE learning:
    - Logistic: if ENABLE_ONLINE_LOGISTIC is True and a special label
      message is present in recent_features, we do an SGD step.
    - Env ODE: if ENABLE_ENV_ONLINE_TUNING is True, we update k_temp/k_hum
      from successive temp/humidity samples and persist periodically.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

# Base hazard horizon for training / interpretation (minutes)
BASE_HORIZON_MIN = 30.0

# Paths for model persistence
MODEL_PATH = Path(__file__).with_name("sleep_model_state.json")
ENV_MODEL_PATH = Path(__file__).with_name("env_model_state.json")

# Online learning toggles
ENABLE_ONLINE_LOGISTIC = False        # set to True to enable online SGD updates
ENABLE_ENV_ONLINE_TUNING = False      # set to True to tune k_temp/k_hum online

# For logistic online learning (small learning rate)
ONLINE_LR = 1e-3

# Node name + key for online labels
# If you push a message like:
#   {"node": "__sleep_label", "sensors": {"y_sleep_within_horizon": 1}}
# into recent_features, we'll use it as an online training target.
LABEL_NODE_NAME = "__sleep_label"
LABEL_KEY = "y_sleep_within_horizon"

# For env model online tuning
ENV_SAVE_INTERVAL_SEC = 60.0  # don't hammer disk
ENV_ETA = 0.05                # how fast to blend new k samples
ENV_K_TEMP_MIN = 1.0 / 240.0  # time constant between ~4h and...
ENV_K_TEMP_MAX = 1.0 / 5.0    # ...5 minutes
ENV_K_HUM_MIN = 1.0 / 240.0
ENV_K_HUM_MAX = 1.0 / 5.0

# State for env online tuning (module-level)
_last_env_sample: Dict[str, Optional[float]] = {
    "ts": None,
    "temp_bed": None,
    "temp_out": None,
    "hum_bed": None,
    "hum_out": None,
}
_last_env_save_time: float = 0.0


# ---------------------------------------------------------------------
# Utility: safe sigmoid
# ---------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


# ---------------------------------------------------------------------
# Logistic model persistence
# ---------------------------------------------------------------------

def _load_logistic_model(n_features: int) -> Tuple[np.ndarray, float]:
    """
    Load logistic regression parameters from JSON, or initialize zeros.

    Model: p = sigmoid(w^T x + b)
    """
    if MODEL_PATH.exists():
        try:
            with MODEL_PATH.open("r") as f:
                state = json.load(f)
            w_list = state.get("w", [])
            b_val = state.get("b", 0.0)
            w = np.array(w_list, dtype=float)
            b = float(b_val)
            if w.shape[0] != n_features:
                # feature dimension changed; re-init
                print("[SLEEP_MODEL] Feature size changed; re-initializing logistic model.")
                w = np.zeros(n_features, dtype=float)
                b = 0.0
        except Exception as e:
            print(f"[SLEEP_MODEL] Warning: failed to load logistic model: {e}")
            w = np.zeros(n_features, dtype=float)
            b = 0.0
    else:
        w = np.zeros(n_features, dtype=float)
        b = 0.0

    return w, b


def _save_logistic_model(w: np.ndarray, b: float) -> None:
    state = {"w": w.tolist(), "b": float(b)}
    try:
        with MODEL_PATH.open("w") as f:
            json.dump(state, f, indent=2)
        print("[SLEEP_MODEL] Saved logistic model to", MODEL_PATH)
    except Exception as e:
        print(f"[SLEEP_MODEL] Warning: failed to save logistic model: {e}")


def _online_update_logistic(w: np.ndarray, b: float,
                            x: np.ndarray, y: float,
                            lr: float = ONLINE_LR) -> Tuple[np.ndarray, float]:
    """
    Single SGD step for logistic regression on one labeled example (x, y).
    """
    z = float(np.dot(w, x) + b)
    p = _sigmoid(z)
    err = p - y
    w = w - lr * err * x
    b = b - lr * err
    return w, b


def _find_online_label(recent_features: List[Dict[str, Any]]) -> Optional[float]:
    """
    Look for a special label node in recent_features of the form:

      {
        "node": "__sleep_label",
        "sensors": {"y_sleep_within_horizon": 0 or 1}
      }

    Return the label as float if found, else None.
    """
    for f in reversed(recent_features):
        if f.get("node") == LABEL_NODE_NAME:
            sensors = f.get("sensors", {})
            if LABEL_KEY in sensors:
                try:
                    return float(sensors[LABEL_KEY])
                except Exception:
                    return None
    return None


# ---------------------------------------------------------------------
# Environment first-order ODE model
# ---------------------------------------------------------------------

class EnvModel:
    """
    Very simple first-order environment model for temperature and humidity.

    We assume:
      dT/dt = -k_T (T - T_out)
      dH/dt = -k_H (H - H_out)

    For now, k_T and k_H are scalars per system. With ENABLE_ENV_ONLINE_TUNING,
    we adapt them online from observed trajectories.
    """

    def __init__(self, k_temp: float = 1.0 / 45.0, k_hum: float = 1.0 / 60.0):
        """
        k_temp, k_hum are approx inverse time constants in 1/min.

        Example:
          k_temp = 1/45 ~ time constant ~45 minutes
          k_hum  = 1/60 ~ time constant ~60 minutes
        """
        self.k_temp = k_temp
        self.k_hum = k_hum

    def estimate_time_to_target_temp(
        self,
        T0: float,
        T_out: float,
        T_target: float,
    ) -> float:
        """
        Estimate time in minutes to reach T_target, using continuous-time
        solution of first-order system:

          T(t) = T_out + (T0 - T_out) * exp(-k_temp * t)

        Solve for t such that T(t) = T_target:

          (T_target - T_out)/(T0 - T_out) = exp(-k_temp * t)
          t = -1/k_temp * ln(ratio)
        """
        if self.k_temp <= 0.0:
            return float("inf")

        denom = (T0 - T_out)
        if abs(denom) < 0.1:
            # No usable gradient (room already near outside temp)
            return 0.0

        num = (T_target - T_out)
        ratio = num / denom
        # If ratio not in (0,1), target not reachable in simple model
        if ratio <= 0.0 or ratio >= 1.0:
            return float("inf")

        t_min = -math.log(ratio) / self.k_temp
        if t_min < 0.0:
            t_min = 0.0
        return t_min

    def estimate_time_to_target_hum(
        self,
        H0: float,
        H_out: float,
        H_target: float,
    ) -> float:
        """
        Same idea as temperature, but for humidity.
        """
        if self.k_hum <= 0.0:
            return float("inf")

        denom = (H0 - H_out)
        if abs(denom) < 0.1:
            return 0.0

        num = (H_target - H_out)
        ratio = num / denom
        if ratio <= 0.0 or ratio >= 1.0:
            return float("inf")

        t_min = -math.log(ratio) / self.k_hum
        if t_min < 0.0:
            t_min = 0.0
        return t_min


def _load_env_model() -> EnvModel:
    """
    Load environment model parameters (k_temp, k_hum) from JSON,
    or initialize defaults if not present.
    """
    if ENV_MODEL_PATH.exists():
        try:
            with ENV_MODEL_PATH.open("r") as f:
                state = json.load(f)
            k_temp = float(state.get("k_temp", 1.0 / 45.0))
            k_hum = float(state.get("k_hum", 1.0 / 60.0))
            return EnvModel(k_temp=k_temp, k_hum=k_hum)
        except Exception as e:
            print(f"[SLEEP_MODEL] Warning: failed to load env model: {e}")
    # defaults
    return EnvModel()


def _save_env_model(env_model: EnvModel) -> None:
    state = {"k_temp": env_model.k_temp, "k_hum": env_model.k_hum}
    try:
        with ENV_MODEL_PATH.open("w") as f:
            json.dump(state, f, indent=2)
        print("[SLEEP_MODEL] Saved env model to", ENV_MODEL_PATH)
    except Exception as e:
        print(f"[SLEEP_MODEL] Warning: failed to save env model: {e}")


def _update_env_model_online(env_model: EnvModel, dbg: Dict[str, Any], ts: float) -> None:
    """
    Online adaptation of k_temp / k_hum from successive samples.

    We look at bed & window temp/hum at consecutive times, estimate an
    effective exponential decay rate k_sample, and blend it into k_temp/k_hum.
    """
    global _last_env_sample, _last_env_save_time

    temp_bed = dbg["temp_bed"]
    hum_bed = dbg["hum_bed"]
    temp_out = dbg["temp_win"]
    hum_out = dbg["hum_win"]

    # First call: just store and return
    if _last_env_sample["ts"] is None:
        _last_env_sample = {
            "ts": ts,
            "temp_bed": temp_bed,
            "temp_out": temp_out,
            "hum_bed": hum_bed,
            "hum_out": hum_out,
        }
        return

    ts_prev = _last_env_sample["ts"]
    dt_sec = ts - ts_prev
    if dt_sec <= 1.0:
        # too close in time; skip
        return

    dt_min = dt_sec / 60.0

    T_prev = _last_env_sample["temp_bed"]
    T_out_prev = _last_env_sample["temp_out"]
    H_prev = _last_env_sample["hum_bed"]
    H_out_prev = _last_env_sample["hum_out"]

    # -------- temp update --------
    try:
        denom_T = (T_prev - T_out_prev)
        if abs(denom_T) > 0.3:
            ratio_T = (temp_bed - temp_out) / denom_T
            # ratio should be between 0 and 1 for decay towards T_out
            if 0.01 < ratio_T < 0.99:
                k_sample_T = -math.log(ratio_T) / dt_min
                # clamp to reasonable band
                k_sample_T = max(ENV_K_TEMP_MIN, min(ENV_K_TEMP_MAX, k_sample_T))
                env_model.k_temp = (
                    (1.0 - ENV_ETA) * env_model.k_temp + ENV_ETA * k_sample_T
                )
    except Exception:
        pass

    # -------- humidity update --------
    try:
        denom_H = (H_prev - H_out_prev)
        if abs(denom_H) > 0.5:
            ratio_H = (hum_bed - hum_out) / denom_H
            if 0.01 < ratio_H < 0.99:
                k_sample_H = -math.log(ratio_H) / dt_min
                k_sample_H = max(ENV_K_HUM_MIN, min(ENV_K_HUM_MAX, k_sample_H))
                env_model.k_hum = (
                    (1.0 - ENV_ETA) * env_model.k_hum + ENV_ETA * k_sample_H
                )
    except Exception:
        pass

    # Store current as last
    _last_env_sample = {
        "ts": ts,
        "temp_bed": temp_bed,
        "temp_out": temp_out,
        "hum_bed": hum_bed,
        "hum_out": hum_out,
    }

    # Persist occasionally
    now = time.time()
    if now - _last_env_save_time > ENV_SAVE_INTERVAL_SEC:
        _save_env_model(env_model)
        _last_env_save_time = now


# ---------------------------------------------------------------------
# Helpers: recent_features parsing
# ---------------------------------------------------------------------

def _find_latest_node(
    recent_features: List[Dict[str, Any]],
    node_name: str,
) -> Optional[Dict[str, Any]]:
    for f in reversed(recent_features):
        if f.get("node") == node_name and "sensors" in f:
            return f
    return None


def _safe_float(sensors: Dict[str, Any], key: str, default: float) -> float:
    val = sensors.get(key, default)
    try:
        return float(val)
    except Exception:
        return default


# ---------------------------------------------------------------------
# Feature extraction for logistic model (now includes camera)
# ---------------------------------------------------------------------

def _extract_features(
    recent_features: List[Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a feature vector x_t from recent_features.

    Features (in order):
      0:  bias = 1
      1:  temp_bed
      2:  hum_bed
      3:  light_bed
      4:  temp_grad (bed - window)
      5:  hum_grad  (bed - window)
      6:  dT_bed_dt (stub 0)
      7:  dL_bed_dt (stub 0)
      8:  noise_level
      9:  motion_index
     10: in_bed_flag (heuristic from light & motion, OR camera)
     11: time_since_in_bed (stub 0)
     12: time_of_day_sin
     13: time_of_day_cos
     14: cam_in_room_flag
     15: cam_in_bed_flag
     16: cam_asleep_like_flag
    """
    now = time.time()

    bedside = _find_latest_node(recent_features, "bedside")
    window = _find_latest_node(recent_features, "window")
    camera = _find_latest_node(recent_features, "camera")

    # Defaults
    temp_bed = 23.0
    hum_bed = 40.0
    light_bed = 100.0
    noise_level = 0.5
    motion_index = 0.5

    temp_win = 23.0
    hum_win = 40.0

    cam_in_room = 0.0
    cam_in_bed = 0.0
    cam_asleep_like = 0.0

    # Bedside sensors
    if bedside is not None:
        s = bedside.get("sensors", {})
        temp_bed = _safe_float(s, "temp_bed_c", temp_bed)
        hum_bed = _safe_float(s, "hum_bed_pct", hum_bed)
        light_bed = _safe_float(s, "light_bed_lux", light_bed)
        noise_level = _safe_float(s, "noise_level", noise_level)
        motion_index = _safe_float(s, "motion_index", motion_index)

    # Window sensors
    if window is not None:
        s = window.get("sensors", {})
        temp_win = _safe_float(s, "temp_win_c", temp_win)
        hum_win = _safe_float(s, "hum_win_pct", hum_win)

    # Camera features as direct inputs
    if camera is not None:
        s = camera.get("sensors", {})
        # Option A: numeric cam_state_code
        if "cam_state_code" in s:
            try:
                code = int(s["cam_state_code"])
            except Exception:
                code = 0
            cam_in_room = 1.0 if code == 0 else 0.0
            cam_in_bed = 1.0 if code == 1 else 0.0
            cam_asleep_like = 1.0 if code == 2 else 0.0
        # Option B: string cam_state
        elif "cam_state" in s:
            st = str(s["cam_state"]).upper()
            cam_in_room = 1.0 if st in ("IN_ROOM", "AWAKE") else 0.0
            cam_in_bed = 1.0 if st in ("IN_BED", "LYING_IN_BED", "ON_BED") else 0.0
            cam_asleep_like = 1.0 if st in ("ASLEEP", "SLEEP_LIKE") else 0.0
        # Option C: probabilistic fields
        else:
            cam_in_bed = _safe_float(s, "in_bed_prob", cam_in_bed)
            cam_asleep_like = _safe_float(s, "sleep_like_prob", cam_asleep_like)

    temp_grad = temp_bed - temp_win
    hum_grad = hum_bed - hum_win

    # TODO: real derivatives over a short history window
    dT_bed_dt = 0.0
    dL_bed_dt = 0.0

    # Time of day
    lt = time.localtime(now)
    tod_hours = lt.tm_hour + lt.tm_min / 60.0
    angle = 2.0 * math.pi * tod_hours / 24.0
    tod_sin = math.sin(angle)
    tod_cos = math.cos(angle)

    # Heuristic in-bed flag:
    # either camera says "in bed" or (low light & low motion)
    in_bed_flag = 0.0
    if cam_in_bed > 0.5:
        in_bed_flag = 1.0
    elif light_bed < 60.0 and motion_index < 0.4:
        in_bed_flag = 1.0

    # Stub: we don't track this explicitly yet
    time_since_in_bed = 0.0

    feats = np.array(
        [
            1.0,
            temp_bed,
            hum_bed,
            light_bed,
            temp_grad,
            hum_grad,
            dT_bed_dt,
            dL_bed_dt,
            noise_level,
            motion_index,
            in_bed_flag,
            time_since_in_bed,
            tod_sin,
            tod_cos,
            cam_in_room,
            cam_in_bed,
            cam_asleep_like,
        ],
        dtype=float,
    )

    debug = {
        "temp_bed": temp_bed,
        "hum_bed": hum_bed,
        "light_bed": light_bed,
        "temp_grad": temp_grad,
        "hum_grad": hum_grad,
        "noise_level": noise_level,
        "motion_index": motion_index,
        "in_bed_flag": in_bed_flag,
        "time_since_in_bed": time_since_in_bed,
        "time_of_day_hours": tod_hours,
        "temp_win": temp_win,
        "hum_win": hum_win,
        "cam_in_room": cam_in_room,
        "cam_in_bed": cam_in_bed,
        "cam_asleep_like": cam_asleep_like,
    }

    return feats, debug


# ---------------------------------------------------------------------
# State classification (AWAKE / WINDING_DOWN / ASLEEP)
# ---------------------------------------------------------------------

def _classify_state_from_sensors(
    temp_bed: float,
    hum_bed: float,
    light_bed: float,
    noise_level: float,
    motion_index: float,
) -> str:
    LIGHT_LOW = 50.0
    LIGHT_VERY_LOW = 10.0
    MOTION_LOW = 0.2
    NOISE_LOW = 0.3

    if (light_bed > LIGHT_LOW) or (motion_index > MOTION_LOW) or (noise_level > NOISE_LOW):
        return "AWAKE"
    elif (light_bed < LIGHT_VERY_LOW) and (motion_index < MOTION_LOW) and (noise_level < NOISE_LOW):
        return "ASLEEP"
    else:
        return "WINDING_DOWN"


# ---------------------------------------------------------------------
# Public: compute_sleep_plan (used by brain_server)
# ---------------------------------------------------------------------

def compute_sleep_plan(recent_features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Called by brain_server in a loop.

    Steps:
      1) Extract features from recent_features (including camera if present).
      2) Run logistic hazard model to get p_sleep_within_base_horizon.
      3) Optional: online SGD update if a label message is present.
      4) Use heuristic to classify current state AWAKE/WINDING_DOWN/ASLEEP.
      5) Use EnvModel to estimate time to hit temp/humidity targets at the bed,
         with optional online tuning of k_temp/k_hum.
      6) Produce a plan dict with state, predicted sleep time, targets, and debug info.
    """
    now = time.time()

    # 1) Features
    feats, dbg = _extract_features(recent_features)
    n_features = feats.shape[0]

    # 2) Logistic hazard
    w, b = _load_logistic_model(n_features)
    z = float(np.dot(w, feats) + b)
    p_sleep_soon = _sigmoid(z)  # P(sleep within BASE_HORIZON_MIN)

    # 3) Optional online update if label present
    if ENABLE_ONLINE_LOGISTIC:
        y_label = _find_online_label(recent_features)
        if y_label is not None:
            w, b = _online_update_logistic(w, b, feats, y_label, lr=ONLINE_LR)
            _save_logistic_model(w, b)  # persist after online step

    # 4) State heuristic
    state = _classify_state_from_sensors(
        temp_bed=dbg["temp_bed"],
        hum_bed=dbg["hum_bed"],
        light_bed=dbg["light_bed"],
        noise_level=dbg["noise_level"],
        motion_index=dbg["motion_index"],
    )

    # 5) Environment ODE estimates (time to targets)
    env_model = _load_env_model()

    temp_bed = dbg["temp_bed"]
    hum_bed = dbg["hum_bed"]
    temp_out = dbg["temp_win"]
    hum_out = dbg["hum_win"]

    # Targets (tune as desired)
    temp_target = max(18.0, min(24.0, temp_bed - 2.0))
    hum_target = 45.0

    tau_temp_min = env_model.estimate_time_to_target_temp(
        T0=temp_bed, T_out=temp_out, T_target=temp_target
    )
    tau_hum_min = env_model.estimate_time_to_target_hum(
        H0=hum_bed, H_out=hum_out, H_target=hum_target
    )

    # Optional env online tuning
    if ENABLE_ENV_ONLINE_TUNING:
        # Use the latest 'ts' from bedside (or fallback to now)
        # We'll just grab default TS from the latest bedside message if available.
        bedside = _find_latest_node(recent_features, "bedside")
        ts = float(bedside["ts"]) if bedside and "ts" in bedside else now
        _update_env_model_online(env_model, dbg, ts)

    # Effective horizon: must be long enough for env to catch up
    effective_horizon_min = max(BASE_HORIZON_MIN, tau_temp_min, tau_hum_min)

    # 6) Predicted sleep time + confidence
    if state == "ASLEEP":
        t_sleep_pred = int(now)  # already asleep
        confidence = max(p_sleep_soon, 0.9)
    else:
        # Map p_sleep_soon into an offset inside the effective horizon.
        if p_sleep_soon >= 0.8:
            dt_min = 0.5 * effective_horizon_min
        elif p_sleep_soon >= 0.5:
            dt_min = 1.0 * effective_horizon_min
        else:
            dt_min = 1.5 * effective_horizon_min

        t_sleep_pred = int(now + dt_min * 60.0)
        confidence = float(p_sleep_soon)

    # Light target is low when winding down or asleep
    if state in ("WINDING_DOWN", "ASLEEP"):
        max_light_lux = 30.0
    else:
        max_light_lux = 150.0

    plan: Dict[str, Any] = {
        "state": state,
        "t_sleep_pred": t_sleep_pred,
        "confidence": confidence,
        "targets": {
            "temp_c": temp_target,
            "humidity_pct": hum_target,
            "max_light_lux": max_light_lux,
        },
        "debug": {
            "p_sleep_within_base_horizon": p_sleep_soon,
            "base_horizon_min": BASE_HORIZON_MIN,
            "effective_horizon_min": effective_horizon_min,
            "tau_temp_min": tau_temp_min,
            "tau_hum_min": tau_hum_min,
            "features": dbg,
            "z_raw": z,
            "k_temp": env_model.k_temp,
            "k_hum": env_model.k_hum,
        },
    }

    return plan


# ---------------------------------------------------------------------
# Offline training from a CSV night log (self-learning part)
# ---------------------------------------------------------------------

def train_on_night_csv(
    csv_path: Path,
    lr: float = 1e-3,
    horizon_min: float = BASE_HORIZON_MIN,
    epochs: int = 1,
) -> None:
    """
    Offline training of the logistic hazard model from a single-night CSV.

    Expected CSV columns (you can adapt for real logs):
      minute, cam_state,
      temp_bed_c, hum_bed_pct, light_bed_lux,
      noise_level, motion_index,
      temp_win_c, hum_win_pct

    We:
      1) Find first minute where cam_state == "ASLEEP" -> t_sleep.
      2) For each minute t < t_sleep:
            y_t = 1 if (t_sleep - t) <= horizon_min, else 0
            x_t = features built from history up to t (like in test script)
      3) Do SGD on logistic model parameters w, b.
    """
    import csv  # local import

    if not csv_path.exists():
        print(f"[TRAIN] CSV {csv_path} does not exist.")
        return

    # Load rows
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["minute"] = int(row["minute"])
            row["cam_state"] = row["cam_state"]
            for k in [
                "temp_bed_c",
                "hum_bed_pct",
                "light_bed_lux",
                "noise_level",
                "motion_index",
                "temp_win_c",
                "hum_win_pct",
            ]:
                row[k] = float(row[k])
            rows.append(row)

    if not rows:
        print(f"[TRAIN] CSV {csv_path} is empty.")
        return

    # Find sleep onset (first ASLEEP in cam_state)
    t_sleep: Optional[int] = None
    for row in rows:
        if row["cam_state"].upper() == "ASLEEP":
            t_sleep = row["minute"]
            break

    if t_sleep is None:
        print("[TRAIN] No ASLEEP state found; nothing to train on.")
        return

    print(f"[TRAIN] Sleep onset at minute {t_sleep}")

    # Helper: build recent_features structure up to minute index i
    def build_recent_features(idx: int) -> List[Dict[str, Any]]:
        rf: List[Dict[str, Any]] = []
        for j in range(0, idx + 1):
            r = rows[j]
            ts = float(r["minute"] * 60)
            bedside_msg = {
                "node": "bedside",
                "ts": ts,
                "sensors": {
                    "temp_bed_c": r["temp_bed_c"],
                    "hum_bed_pct": r["hum_bed_pct"],
                    "light_bed_lux": r["light_bed_lux"],
                    "noise_level": r["noise_level"],
                    "motion_index": r["motion_index"],
                },
            }
            window_msg = {
                "node": "window",
                "ts": ts,
                "sensors": {
                    "temp_win_c": r["temp_win_c"],
                    "hum_win_pct": r["hum_win_pct"],
                },
            }
            # Camera node as feature (cam_state)
            camera_msg = {
                "node": "camera",
                "ts": ts,
                "sensors": {
                    "cam_state": r["cam_state"],
                },
            }
            rf.append(bedside_msg)
            rf.append(window_msg)
            rf.append(camera_msg)
        return rf

    # Build list of (x_t, y_t)
    feature_list: List[np.ndarray] = []
    label_list: List[float] = []

    for idx, row in enumerate(rows):
        t = row["minute"]
        if t >= t_sleep:
            break  # don't train on or after sleep

        dt_to_sleep = t_sleep - t
        y = 1.0 if dt_to_sleep <= horizon_min else 0.0

        recent_features = build_recent_features(idx)
        x_vec, _dbg = _extract_features(recent_features)

        feature_list.append(x_vec)
        label_list.append(y)

    if not feature_list:
        print("[TRAIN] No training samples found (check t_sleep / horizon).")
        return

    X = np.stack(feature_list, axis=0)
    y_arr = np.array(label_list, dtype=float)

    print(f"[TRAIN] {X.shape[0]} samples, {X.shape[1]} features.")

    # Load model or init
    w, b = _load_logistic_model(n_features=X.shape[1])

    # SGD
    for ep in range(epochs):
        total_loss = 0.0
        # simple online pass in order (you can shuffle if desired)
        for xi, yi in zip(X, y_arr):
            z = float(np.dot(w, xi) + b)
            p = _sigmoid(z)
            # logistic loss
            loss = -(yi * math.log(max(p, 1e-8)) +
                     (1.0 - yi) * math.log(max(1.0 - p, 1e-8)))
            total_loss += loss

            err = p - yi
            w -= lr * err * xi
            b -= lr * err

        avg_loss = total_loss / len(y_arr)
        print(f"[TRAIN] epoch {ep+1}/{epochs}, avg_loss={avg_loss:.4f}")

    _save_logistic_model(w, b)
    print("[TRAIN] Done training on", csv_path)
