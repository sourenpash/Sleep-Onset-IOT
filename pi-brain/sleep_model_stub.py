from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


def compute_sleep_plan(recent_features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sleep-plan model with simple fusion of:
      - bedside node (what user feels: temp, hum, light, motion, noise)
      - window node (room envelope + outside temp)
      - camera node (pose-based activity: IN_BED_LYING, AWAKE_IN_ROOM, AWAY)

    Outputs:
      {
        "state": "AWAKE" | "WINDING_DOWN" | "ASLEEP",
        "t_sleep_pred": <unix time>,
        "confidence": float,
        "targets": {
          "temp_c": float,
          "humidity_pct": float,
          "max_light_lux": float
        }
      }
    """

    if not recent_features:
        return _default_plan()

    # Latest readings from each node we care about
    bedside = _latest_by_node(recent_features, "bedside")
    window = _latest_by_node(recent_features, "window")
    camera = _latest_by_node(recent_features, "camera")

    # If no explicit bedside node yet, fall back to last feature
    if bedside is None:
        bedside = recent_features[-1]

    sensors_bed = bedside.get("sensors", {})
    sensors_win = window.get("sensors", {}) if window else {}
    sensors_cam = camera.get("sensors", {}) if camera else {}

    # ------------------------------------------------------------------
    # Bedside zone: what the person actually feels (primary comfort view)
    # ------------------------------------------------------------------
    light_bed = float(sensors_bed.get("light_bed_lux", 200.0))
    motion = float(sensors_bed.get("motion_index", 0.5))
    noise = float(sensors_bed.get("noise_level", 0.5))
    temp_bed = float(sensors_bed.get("temp_bed_c", 23.0))
    hum_bed = float(sensors_bed.get("hum_bed_pct", 40.0))

    # Window zone: room envelope + outside boundary
    temp_win = float(sensors_win.get("temp_win_c", temp_bed))
    hum_win = float(sensors_win.get("hum_win_pct", hum_bed))
    temp_out = float(sensors_win.get("temp_out_c", temp_win))
    # hum_out = float(sensors_win.get("hum_out_pct", hum_bed))  # if needed later

    # Camera activity: pose-based coarse state
    activity_state = sensors_cam.get("activity_state", None)
    activity_conf = float(sensors_cam.get("activity_conf", 0.0))

    # ------------------------------------------------------------------
    # 1) State classification using camera + light + motion + noise
    # ------------------------------------------------------------------
    # Light thresholds at bed
    LIGHT_DIM = 50.0    # "dim-ish" threshold
    LIGHT_DARK = 20.0   # "very dark" threshold

    # Activity helps a lot:
    # - IN_BED_LYING + dark + very low motion/noise -> ASLEEP
    # - IN_BED_LYING + dim + low motion/noise      -> WINDING_DOWN
    # - AWAKE_IN_ROOM or bright conditions         -> AWAKE
    # - AWAY                                       -> treat as AWAKE for now (not sleeping here)

    # Motion / noise thresholds
    MOTION_LOW = 0.2
    MOTION_VERY_LOW = 0.1
    NOISE_LOW = 0.3
    NOISE_VERY_LOW = 0.15

    state = "AWAKE"  # default

    if activity_state == "IN_BED_LYING" and activity_conf > 0.5:
        # Lying in bed: use environment to distinguish relaxing vs sleeping
        if (light_bed < LIGHT_DARK
            and motion < MOTION_VERY_LOW
            and noise < NOISE_VERY_LOW):
            state = "ASLEEP"
        elif (light_bed < LIGHT_DIM
              and motion < MOTION_LOW
              and noise < NOISE_LOW):
            state = "WINDING_DOWN"
        else:
            # Lying down but brighter or more active -> still AWAKE in bed
            state = "AWAKE"

    elif activity_state == "AWAKE_IN_ROOM" and activity_conf > 0.5:
        # Upright in camera view -> clearly AWAKE
        state = "AWAKE"

    elif activity_state == "AWAY" and activity_conf > 0.5:
        # Person is not in bed / out of frame; we can't assume sleep here.
        # For now treat as AWAKE (not sleeping in this environment).
        state = "AWAKE"

    else:
        # If camera info is missing or low-confidence, fall back to old heuristic
        state = _fallback_state_from_bed_signals(light_bed, motion, noise)

    # ------------------------------------------------------------------
    # 2) Sleep onset prediction (still simple, but now state-aware)
    # ------------------------------------------------------------------
    now = time.time()
    if state == "WINDING_DOWN":
        # If winding down in dark conditions, predict sleep sooner
        if light_bed < LIGHT_DARK and motion < MOTION_LOW and noise < NOISE_LOW:
            t_sleep_pred = int(now + 15 * 60)  # ~15 min
            confidence = 0.8
        else:
            t_sleep_pred = int(now + 30 * 60)  # ~30 min
            confidence = 0.7
    elif state == "ASLEEP":
        # Already asleep; approximate onset at last bedside timestamp
        t_sleep_pred = int(bedside.get("ts", now))
        confidence = 0.9
    else:  # AWAKE
        # Guess sleep in ~60 minutes with low confidence
        t_sleep_pred = int(now + 60 * 60)
        confidence = 0.3

    # ------------------------------------------------------------------
    # 3) Environment targets (temp, humidity, light)
    # ------------------------------------------------------------------

    # Temperature target: we want the bed zone a bit cooler,
    # but not below outside temperature + some margin.
    margin = 1.0  # °C above outside
    raw_temp_target = temp_bed - 2.0  # desired ~2°C cooler than current
    min_temp_target = temp_out + margin

    temp_target = max(min_temp_target, raw_temp_target)
    # Clamp to a reasonable comfort range
    temp_target = max(18.0, min(24.0, temp_target))

    # Humidity: keep it simple for now, can refine later based on hum_bed/hum_win
    humidity_target = 45.0

    # Light: we can be stricter when winding down/asleep
    if state in ("WINDING_DOWN", "ASLEEP"):
        max_light_lux = 20.0
    else:
        max_light_lux = 80.0  # allow brighter while awake

    plan = {
        "state": state,
        "t_sleep_pred": t_sleep_pred,
        "confidence": confidence,
        "targets": {
            "temp_c": temp_target,
            "humidity_pct": humidity_target,
            "max_light_lux": max_light_lux,
        },
    }

    return plan


def _fallback_state_from_bed_signals(
    light_bed: float,
    motion: float,
    noise: float,
) -> str:
    """Original heuristic using only bedside light/motion/noise."""
    LIGHT_LOW = 50.0       # lux
    LIGHT_VERY_LOW = 10.0  # lux
    MOTION_LOW = 0.2
    NOISE_LOW = 0.3

    if light_bed > LIGHT_LOW or motion > MOTION_LOW or noise > NOISE_LOW:
        return "AWAKE"
    elif light_bed < LIGHT_VERY_LOW and motion < MOTION_LOW and noise < NOISE_LOW:
        return "ASLEEP"
    else:
        return "WINDING_DOWN"


def _default_plan() -> Dict[str, Any]:
    """Fallback plan when we have no data at all."""
    now = time.time()
    return {
        "state": "AWAKE",
        "t_sleep_pred": int(now + 60 * 60),
        "confidence": 0.1,
        "targets": {
            "temp_c": 22.0,
            "humidity_pct": 45.0,
            "max_light_lux": 30.0,
        },
    }


def _latest_by_node(
    features: List[Dict[str, Any]], node_name: str
) -> Optional[Dict[str, Any]]:
    """
    Return the most recent feature dict for the given node name,
    or None if not found.
    """
    for f in reversed(features):
        if f.get("node") == node_name:
            return f
    return None
