"""
sleep_model_stub.py

Logistic-regression-based sleep model + self-tuning ODEs + camera override.

Pipeline:

- Training:
    * training_data.csv rows come from camera labels + context (window/bedside/
      door/weather) built in brain_server.py.
    * We map camera labels -> binary target (SLEEP vs AWAKE).
    * We train a logistic regression in pure numpy.
    * We save weights + normalization stats to logs/sleep_model.json.

- Dynamics (ODEs):
    * We maintain first-order ODE time constants for:
        dT/dt = -(T - T_out) / tau_T
        dH/dt = -(H - H_out) / tau_H
      where T is indoor bed temp, H is indoor bed humidity, and T_out/H_out come
      from the weather node.
    * Each call, we estimate a fresh tau_T, tau_H from recent temperature/
      humidity history and update them via an exponential moving average.
    * ODE is used to estimate how long it will take to reach target temp /
      humidity (cooldown_time_s).

- Inference (compute_sleep_plan):
    * Load trained logistic regression model if present.
    * Extract current feature vector (window/bedside/door/weather sensors).
    * Estimate dynamics (tau_T, tau_H) and predicted cooldown_time_s.
    * Predict a FUTURE feature vector at t = cooldown_time_s using the ODE
      for temp_bed_c and hum_bed_pct (others left as current).
    * Run logistic regression on that future feature vector -> p_sleep.
    * Combine with latest camera label:
        - camera strongly says "in bed / sleeping" => override to SLEEP
        - else, state = SLEEP if p_sleep >= 0.5, else AWAKE.
    * Return a rich plan dict.

This file is intentionally self-contained and uses only numpy + stdlib.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FEATURE_NAMES = [
    # Window node
    "temp_win_c",
    "hum_win_pct",
    "light_win_lux",

    # Bedside node
    "temp_bed_c",
    "hum_bed_pct",
    "lux_bed",
    "light_bed_lux",

    # Door node
    "temp_door_c",
    "hum_door_pct",
    "mic_v",
    "light_door_v",

    # Outdoor weather
    "temp_outdoor_c",
    "hum_outdoor_pct",
]

# Mapping from feature_name -> (node_name, sensor_key)
FEATURE_SPEC: Dict[str, Tuple[str, str]] = {
    "temp_win_c": ("window", "temp_win_c"),
    "hum_win_pct": ("window", "hum_win_pct"),
    "light_win_lux": ("window", "light_win_lux"),

    "temp_bed_c": ("bedside", "temp_bed_c"),
    "hum_bed_pct": ("bedside", "hum_bed_pct"),
    "lux_bed": ("bedside", "lux_bed"),
    "light_bed_lux": ("bedside", "light_bed_lux"),

    "temp_door_c": ("door", "temp_door_c"),
    "hum_door_pct": ("door", "hum_door_pct"),
    "mic_v": ("door", "mic_v"),
    "light_door_v": ("door", "light_door_v"),

    "temp_outdoor_c": ("weather", "temp_outdoor_c"),
    "hum_outdoor_pct": ("weather", "hum_outdoor_pct"),
}

# Model + dynamics paths
LOG_DIR = Path("logs")
DEFAULT_MODEL_PATH = LOG_DIR / "sleep_model.json"
DYN_STATE_PATH = LOG_DIR / "sleep_dynamics.json"

# ---------------------------------------------------------------------------
# Comfort targets and ODE parameters
# ---------------------------------------------------------------------------

TARGET_TEMP_C = 21.0       # comfort temperature at bed (°C)
TARGET_HUM_PCT = 45.0      # comfort RH (%)
MIN_TAU_S = 60.0           # min plausible time constant (1 min)
MAX_TAU_S = 3 * 3600.0     # max (3 hours)
DYN_EMA_ALPHA = 0.2        # how fast we update tau estimates
MAX_PREDICTION_WINDOW_S = 2 * 3600.0  # clamp horizon to 2h

# ---------------------------------------------------------------------------
# Camera label → binary target (sleep vs awake)
# ---------------------------------------------------------------------------

def _label_to_target(label: str) -> Optional[int]:

    if not label:
        return None
    u = str(label).upper()

    # Sleepy / in-bed variants
    if "SLEEP" in u or "IN_BED" in u or "LYING" in u or "LAYING" in u:
        return 1

    # Awake variants
    if "AWAKE" in u or "OUT_OF_BED" in u or "STAND" in u or "UP" in u or "ROOM" in u:
        return 0

    # Unknown label → skip for training
    return None

# ---------------------------------------------------------------------------
# TRAINING API (logistic regression)
# ---------------------------------------------------------------------------

def train_model_from_csv(
    csv_path: Path,
    model_out_path: Path = DEFAULT_MODEL_PATH,
    min_conf: float = 0.5,
) -> bool:

    if not csv_path.exists():
        print(f"[MODEL] No training CSV at {csv_path}, cannot train.")
        return False

    X_rows: List[List[float]] = []
    y_rows: List[int] = []

    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_str = row.get("label", "")
                target = _label_to_target(label_str)
                if target is None:
                    continue

                conf_raw = row.get("label_conf", "")
                try:
                    conf_val = float(conf_raw) if conf_raw != "" else float("nan")
                except Exception:
                    conf_val = float("nan")

                if not math.isnan(conf_val) and conf_val < min_conf:
                    continue

                feats: List[float] = []
                for name in FEATURE_NAMES:
                    v_raw = row.get(name, "")
                    if v_raw is None or str(v_raw).strip() == "":
                        feats.append(float("nan"))
                        continue
                    try:
                        feats.append(float(v_raw))
                    except Exception:
                        feats.append(float("nan"))

                X_rows.append(feats)
                y_rows.append(int(target))

    except Exception as e:
        print(f"[MODEL] Error reading training CSV: {e}")
        return False

    if not X_rows:
        print("[MODEL] No usable training rows after filtering; aborting training.")
        return False

    X = np.asarray(X_rows, dtype=float)  # (N, D)
    y = np.asarray(y_rows, dtype=float)  # (N,)

    # Impute NaNs with per-feature mean, then standardize
    means = np.zeros(X.shape[1], dtype=float)
    stds = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isfinite(col)
        if not np.any(mask):
            means[j] = 0.0
            stds[j] = 1.0
            X[:, j] = 0.0
        else:
            m = float(col[mask].mean())
            means[j] = m
            col_imputed = col.copy()
            col_imputed[~mask] = m
            X[:, j] = col_imputed
            s = float(col_imputed.std())
            stds[j] = s if s > 1e-6 else 1.0

    X_norm = (X - means[None, :]) / stds[None, :]

    w, b = _train_logistic_gradient_descent(X_norm, y)

    model = {
        "feature_names": FEATURE_NAMES,
        "mean": means.tolist(),
        "std": stds.tolist(),
        "weights": w.tolist(),
        "bias": float(b),
        "trained_at": time.time(),
        "n_samples": int(len(y)),
    }

    try:
        model_out_path.parent.mkdir(parents=True, exist_ok=True)
        model_out_path.write_text(json.dumps(model))
        print(
            f"[MODEL] Trained logistic regression on {len(y)} samples; "
            f"saved to {model_out_path}"
        )
        return True
    except Exception as e:
        print(f"[MODEL] Failed to save model to {model_out_path}: {e}")
        return False


def _train_logistic_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    n_iter: int = 500,
) -> Tuple[np.ndarray, float]:
    N, D = X.shape
    w = np.zeros(D, dtype=float)
    b = 0.0

    for _ in range(n_iter):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

        error = p - y  # (N,)
        grad_w = (X.T @ error) / N
        grad_b = float(error.mean())

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b

# ---------------------------------------------------------------------------
# Model + dynamics loading
# ---------------------------------------------------------------------------

def _load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Optional[Dict[str, Any]]:
    if not model_path.exists():
        print(f"[MODEL] No model file at {model_path}; cannot predict.")
        return None
    try:
        return json.loads(model_path.read_text())
    except Exception as e:
        print(f"[MODEL] Failed to load model from {model_path}: {e}")
        return None


def _load_dyn_state(path: Path = DYN_STATE_PATH) -> Dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return {
                "tau_temp_s": float(data.get("tau_temp_s", 900.0)),
                "tau_hum_s": float(data.get("tau_hum_s", 900.0)),
                "updated_at": float(data.get("updated_at", time.time())),
            }
        except Exception as e:
            print(f"[DYN] Failed to load dynamics state: {e}")
    return {
        "tau_temp_s": 900.0,
        "tau_hum_s": 900.0,
        "updated_at": time.time(),
    }


def _save_dyn_state(state: Dict[str, Any], path: Path = DYN_STATE_PATH) -> None:
    state["updated_at"] = time.time()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state))
    except Exception as e:
        print(f"[DYN] Failed to save dynamics state: {e}")

# ---------------------------------------------------------------------------
# Helpers over recent_features
# ---------------------------------------------------------------------------

def _latest_msg_for_node(
    recent_features: List[Dict[str, Any]],
    node_name: str,
    max_age_sec: float,
    ref_ts: float,
) -> Optional[Dict[str, Any]]:
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


def _extract_feature_vector_from_recent(
    recent_features: List[Dict[str, Any]],
    feature_names: List[str],
    max_age_sec: float = 120.0,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if not recent_features:
        return None, {}

    now = time.time()
    debug: Dict[str, Any] = {}

    values: List[float] = []
    any_finite = False

    for fname in feature_names:
        node, skey = FEATURE_SPEC.get(fname, (None, None))
        if node is None:
            values.append(float("nan"))
            debug[fname] = None
            continue

        msg = _latest_msg_for_node(recent_features, node, max_age_sec, now)
        if msg is None:
            values.append(float("nan"))
            debug[fname] = None
            continue

        sensors = msg.get("sensors", {})
        v = sensors.get(skey)
        try:
            fv = float(v)
        except Exception:
            fv = float("nan")

        values.append(fv)
        debug[fname] = fv
        if math.isfinite(fv):
            any_finite = True

    if not any_finite:
        return None, debug

    x = np.asarray(values, dtype=float)
    return x, debug


def _latest_camera_info(
    recent_features: List[Dict[str, Any]],
    max_age_sec: float = 60.0,
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    if not recent_features:
        return None, None, None

    now = time.time()
    cam_msg = _latest_msg_for_node(recent_features, "camera", max_age_sec, now)
    if cam_msg is None:
        return None, None, None

    sensors = cam_msg.get("sensors", {})

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

    if label is None:
        for k, v in sensors.items():
            if not isinstance(v, str):
                continue
            kl = k.lower()
            if any(sub in kl for sub in ["state", "label", "pose", "sleep"]):
                label = v
                break

    try:
        conf_val = float(conf) if conf is not None else None
    except Exception:
        conf_val = None

    ts = cam_msg.get("ts", now)
    try:
        age = float(now - float(ts))
    except Exception:
        age = None

    return label, conf_val, age

# ---------------------------------------------------------------------------
# Dynamics estimation (ODE time constants)
# ---------------------------------------------------------------------------

def _extract_time_series(
    recent_features: List[Dict[str, Any]],
    node: str,
    key: str,
    max_age_sec: float = 1800.0,  # up to 30 min
) -> List[Tuple[float, float]]:
    series: List[Tuple[float, float]] = []
    now = time.time()
    for msg in recent_features:
        if msg.get("node") != node:
            continue
        ts = msg.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        if now - ts > max_age_sec:
            continue
        sensors = msg.get("sensors", {})
        v = sensors.get(key)
        try:
            fv = float(v)
        except Exception:
            continue
        series.append((float(ts), fv))
    series.sort(key=lambda x: x[0])
    return series


def _estimate_tau_from_series(
    series: List[Tuple[float, float]],
    eq_series: List[Tuple[float, float]],
) -> Optional[float]:
    if len(series) < 2 or len(eq_series) == 0:
        return None

    # last two indoor samples
    t0, x0 = series[-2]
    t1, x1 = series[-1]
    dt = t1 - t0
    if dt <= 1.0:
        return None

    # eq value: latest weather
    eq_ts, x_eq = eq_series[-1]

    diff0 = x0 - x_eq
    diff1 = x1 - x_eq
    if diff0 == 0 or diff1 == 0:
        return None
    if diff0 * diff1 <= 0:
        # changed sign (crossed equilibrum) -> unstable for tau estimation
        return None

    ratio = diff1 / diff0
    if ratio <= 0:
        return None

    try:
        tau = -dt / math.log(ratio)
    except (ValueError, ZeroDivisionError):
        return None

    if not math.isfinite(tau):
        return None
    if tau < MIN_TAU_S or tau > MAX_TAU_S:
        return None

    return float(tau)


def _update_dyn_state_from_recent(
    recent_features: List[Dict[str, Any]],
    dyn_state: Dict[str, Any],
) -> Dict[str, Any]:
    temp_series = _extract_time_series(recent_features, "bedside", "temp_bed_c")
    hum_series = _extract_time_series(recent_features, "bedside", "hum_bed_pct")
    temp_out_series = _extract_time_series(recent_features, "weather", "temp_outdoor_c")
    hum_out_series = _extract_time_series(recent_features, "weather", "hum_outdoor_pct")

    tau_temp = dyn_state.get("tau_temp_s", 900.0)
    tau_hum = dyn_state.get("tau_hum_s", 900.0)

    tau_temp_hat = _estimate_tau_from_series(temp_series, temp_out_series)
    if tau_temp_hat is not None:
        tau_temp = (1.0 - DYN_EMA_ALPHA) * tau_temp + DYN_EMA_ALPHA * tau_temp_hat

    tau_hum_hat = _estimate_tau_from_series(hum_series, hum_out_series)
    if tau_hum_hat is not None:
        tau_hum = (1.0 - DYN_EMA_ALPHA) * tau_hum + DYN_EMA_ALPHA * tau_hum_hat

    # clamp
    tau_temp = float(max(MIN_TAU_S, min(MAX_TAU_S, tau_temp)))
    tau_hum = float(max(MIN_TAU_S, min(MAX_TAU_S, tau_hum)))

    dyn_state["tau_temp_s"] = tau_temp
    dyn_state["tau_hum_s"] = tau_hum
    return dyn_state


def _predict_time_to_target(
    current: Optional[float],
    target: float,
    eq_val: Optional[float],
    tau: float,
) -> Optional[float]:
    if current is None or eq_val is None:
        return None
    if not (math.isfinite(current) and math.isfinite(eq_val)):
        return None

    num = target - eq_val
    den = current - eq_val
    if den == 0 or num == 0:
        return 0.0
    ratio = num / den
    if ratio <= 0:
        # target not between current and equilibrium in a clean way
        return None

    try:
        t = -tau * math.log(ratio)
    except (ValueError, ZeroDivisionError):
        return None
    if t < 0:
        return 0.0
    return float(t)


def _predict_env_at_time(
    current: Optional[float],
    eq_val: Optional[float],
    tau: float,
    t: float,
) -> Optional[float]:
    if current is None or eq_val is None:
        return None
    if not (math.isfinite(current) and math.isfinite(eq_val)):
        return None
    if tau <= 0 or t < 0:
        return current
    return float(eq_val + (current - eq_val) * math.exp(-t / tau))

# ---------------------------------------------------------------------------
# Logistic prediction
# ---------------------------------------------------------------------------

def _predict_prob_sleep(
    model: Dict[str, Any],
    x_raw: np.ndarray,
) -> float:
    means = np.asarray(model["mean"], dtype=float)
    stds = np.asarray(model["std"], dtype=float)
    w = np.asarray(model["weights"], dtype=float)
    b = float(model["bias"])

    x = x_raw.copy()
    mask = ~np.isfinite(x)
    if np.any(mask):
        x[mask] = means[mask]

    std_safe = stds.copy()
    std_safe[std_safe < 1e-6] = 1.0
    x_norm = (x - means) / std_safe

    z = float(x_norm @ w + b)
    z_clip = max(min(z, 40.0), -40.0)
    p = 1.0 / (1.0 + math.exp(-z_clip))
    return float(p)

# ---------------------------------------------------------------------------
# Public API: compute_sleep_plan
# ---------------------------------------------------------------------------

def compute_sleep_plan(
    recent_features: List[Dict[str, Any]],
    model_version: Optional[int] = None,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Main "brain" entry point.

    Steps:
      1) Load logistic model (if exists).
      2) Extract current feature vector and debug dict.
      3) Load & update ODE dynamics (tau_temp_s, tau_hum_s) from recent data.
      4) Compute ODE-based prediction window (cooldown_time_s) from how long
         it should take to reach TARGET_TEMP_C and TARGET_HUM_PCT.
      5) Build a future feature vector at t = cooldown_time_s by projecting
         temp_bed_c and hum_bed_pct forward using the ODEs.
      6) Run logistic regression on that future feature vector to get
         p_sleep_model (if model available).
      7) Camera label ALWAYS decides global_state when present:
           - camera → SLEEP label -> global_state = "SLEEP"
           - camera → AWAKE label -> global_state = "AWAKE"
         Model is used only as a side-quantity.
      8) If there is NO camera label, fall back to model / heuristics.
    """
    # 1) Load model
    model = _load_model(model_path)

    # 2) Current feature vector
    x_now, feats_now = _extract_feature_vector_from_recent(recent_features, FEATURE_NAMES)

    # 3) Load & update dynamics
    dyn_state = _load_dyn_state()
    dyn_state = _update_dyn_state_from_recent(recent_features, dyn_state)
    _save_dyn_state(dyn_state)

    tau_temp = dyn_state.get("tau_temp_s", 900.0)
    tau_hum = dyn_state.get("tau_hum_s", 900.0)

    # 4) Extract indoor/outdoor T/H for ODE
    def _get(feats: Dict[str, Any], key: str) -> Optional[float]:
        v = feats.get(key)
        try:
            return float(v)
        except Exception:
            return None

    T_in = _get(feats_now, "temp_bed_c")
    H_in = _get(feats_now, "hum_bed_pct")
    T_out = _get(feats_now, "temp_outdoor_c")
    H_out = _get(feats_now, "hum_outdoor_pct")

    # ODE-based prediction window
    t_temp = _predict_time_to_target(T_in, TARGET_TEMP_C, T_out, tau_temp)
    t_hum = _predict_time_to_target(H_in, TARGET_HUM_PCT, H_out, tau_hum)

    times = [t for t in [t_temp, t_hum] if t is not None]
    if times:
        cooldown_time_s = max(times)
        cooldown_time_s = float(
            max(0.0, min(MAX_PREDICTION_WINDOW_S, cooldown_time_s))
        )
    else:
        cooldown_time_s = 0.0

    # 5) Future feature vector at t = cooldown_time_s
    x_future = None
    feats_future: Dict[str, Any] = {}
    if x_now is not None:
        x_future = x_now.copy()
        feats_future = dict(feats_now)

        idx_temp_bed = FEATURE_NAMES.index("temp_bed_c")
        idx_hum_bed = FEATURE_NAMES.index("hum_bed_pct")

        if cooldown_time_s > 0:
            T_future = _predict_env_at_time(T_in, T_out, tau_temp, cooldown_time_s)
            H_future = _predict_env_at_time(H_in, H_out, tau_hum, cooldown_time_s)
        else:
            T_future = T_in
            H_future = H_in

        if T_future is not None:
            x_future[idx_temp_bed] = T_future
            feats_future["temp_bed_c"] = T_future
        if H_future is not None:
            x_future[idx_hum_bed] = H_future
            feats_future["hum_bed_pct"] = H_future

    # 6) Model probability 
    p_sleep_model = None
    model_reason = ""
    if model is not None and x_future is not None:
        p_sleep_model = _predict_prob_sleep(model, x_future)
        state_model = "SLEEP" if p_sleep_model >= 0.5 else "AWAKE"
        model_reason = (
            f"logistic on future env (t={cooldown_time_s:.0f}s) "
            f"p_sleep={p_sleep_model:.3f} → {state_model}"
        )
    elif model is None:
        model_reason = "no trained model yet"
    elif x_future is None:
        model_reason = "no usable features from sensors"

    # 7) Camera label: HARD OVERRIDE OF STATE
    cam_label, cam_conf, cam_age = _latest_camera_info(recent_features)

    cam_target = _label_to_target(cam_label) if cam_label is not None else None

    # Default state if we have nothing
    global_state = "AWAKE"
    p_sleep = 0.0
    reason_parts: List[str] = []

    # Case A: camera label present → trust it, always
    if cam_target is not None:
        if cam_target == 1:
            global_state = "SLEEP"
            # align probability with camera; if model exists and is higher, keep it
            if p_sleep_model is not None:
                p_sleep = max(0.9, p_sleep_model)
            else:
                p_sleep = 0.9
            reason_parts.append(
                f"camera label '{cam_label}' mapped to SLEEP (hard override)"
            )
        else:  # cam_target == 0
            global_state = "AWAKE"
            if p_sleep_model is not None:
                p_sleep = min(0.1, p_sleep_model)
            else:
                p_sleep = 0.1
            reason_parts.append(
                f"camera label '{cam_label}' mapped to AWAKE (hard override)"
            )

        reason_parts.append(
            f"camera_conf={cam_conf}, camera_age_s={cam_age}"
        )

        if model_reason:
            reason_parts.append(f"model: {model_reason}")

    # Case B: no camera label → fall back to model or heuristic
    else:
        if p_sleep_model is not None:
            p_sleep = p_sleep_model
            global_state = "SLEEP" if p_sleep_model >= 0.5 else "AWAKE"
            reason_parts.append(model_reason)
        else:
            # nothing to go on
            global_state = "AWAKE"
            p_sleep = 0.1
            reason_parts.append("no camera label and no usable model/features")

    plan: Dict[str, Any] = {
        "global_state": global_state,
        "p_sleep": p_sleep,
        "p_sleep_model": p_sleep_model,
        "camera_label": cam_label,
        "camera_conf": cam_conf,
        "camera_age_s": cam_age,
        "features_now": feats_now,
        "features_future": feats_future,
        "cooldown_time_s": cooldown_time_s,
        "tau_temp_s": tau_temp,
        "tau_hum_s": tau_hum,
        "target_temp_c": TARGET_TEMP_C,
        "target_hum_pct": TARGET_HUM_PCT,
        "reason": "; ".join(reason_parts),
    }
    if model_version is not None:
        plan["model_version"] = int(model_version)

    return plan
