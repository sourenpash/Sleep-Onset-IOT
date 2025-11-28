"""
env_map.py

Builds simple 2D heatmaps (temperature, humidity, light) over the room
using the latest readings from the ESP32 nodes as anchor points.

Usage:
  - As a module:
      from env_map import build_env_maps, compute_bed_env

  - As a script:
      python env_map.py
    This will:
      * Reload logs/sensor_log.csv every ~N seconds
      * Build env maps from the latest reading per node
      * Show live-updating heatmaps for temperature, humidity, and light.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Room + anchor configuration (EDIT THESE TO MATCH YOUR ROOM)
# ----------------------------------------------------------------------

# Rough room dimensions in meters
ROOM_W = 4.0  # width (x-direction)
ROOM_H = 3.0  # depth (y-direction)

# Approximate locations of nodes (x, y in meters in this room frame)
#  (0,0) is one corner of the room, x along width, y along depth.
NODE_POSITIONS: Dict[str, Tuple[float, float]] = {
    # example: window node on the left wall, mid-depth
    "window":  (0.3, ROOM_H / 2.0),
    # example: bedside node near the right wall, mid-depth
    "bedside": (ROOM_W - 0.3, ROOM_H / 2.0),
}

# Treat bed position as bedside node location for now
POS_BED: Tuple[float, float] = NODE_POSITIONS["bedside"]

# Grid resolution for the heatmap
GRID_NX = 20  # number of columns (x-direction)
GRID_NY = 12  # number of rows (y-direction)

# Small epsilon for distance to avoid division by zero
EPS = 1e-3

# Log file path (same as brain_server.py)
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "sensor_log.csv"


# ----------------------------------------------------------------------
# Public API for the brain (from recent_features in memory)
# ----------------------------------------------------------------------

def build_env_maps(
    recent_features: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build environment heatmaps from the latest readings per node.

    Args:
        recent_features: list of messages as used in brain_server, each like:
            {
              "node": "bedside" or "window",
              "ts": ...,
              "sensors": { ... }
            }

    Returns:
        grid_x: (GRID_NY, GRID_NX) array of x coordinates (meters)
        grid_y: (GRID_NY, GRID_NX) array of y coordinates (meters)
        temp_map: (GRID_NY, GRID_NX) array of temperature (°C) or NaN
        hum_map:  (GRID_NY, GRID_NX) array of humidity (%) or NaN
        light_map:(GRID_NY, GRID_NX) array of light (lux) or NaN

    Notes:
        - If we have no anchors with valid values for a given quantity,
          that map will be filled with NaN.
        - For now we assume:
            window node:  temp_win_c, hum_win_pct
            bedside node: temp_bed_c, hum_bed_pct, light_bed_lux
    """
    anchors = _collect_anchors(recent_features)

    # Build grid of (x,y) coordinates
    xs = np.linspace(0.0, ROOM_W, GRID_NX)
    ys = np.linspace(0.0, ROOM_H, GRID_NY)
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Prepare empty maps
    temp_map = np.full((GRID_NY, GRID_NX), np.nan, dtype=np.float32)
    hum_map = np.full((GRID_NY, GRID_NX), np.nan, dtype=np.float32)
    light_map = np.full((GRID_NY, GRID_NX), np.nan, dtype=np.float32)

    if not anchors:
        # No sensor data at all yet
        return grid_x, grid_y, temp_map, hum_map, light_map

    # Positions and values arrays, shape (n_anchors,)
    ax = np.array([a["pos"][0] for a in anchors], dtype=np.float32)
    ay = np.array([a["pos"][1] for a in anchors], dtype=np.float32)
    temps = np.array([a["temp"] for a in anchors], dtype=np.float32)
    hums = np.array([a["hum"] for a in anchors], dtype=np.float32)
    lights = np.array([a["light"] for a in anchors], dtype=np.float32)

    # Distance from each grid cell to each anchor:
    # grid_x, grid_y: (NY, NX)
    # we want dist: (NY, NX, n_anchors)
    gx = grid_x[..., None]  # (NY, NX, 1)
    gy = grid_y[..., None]  # (NY, NX, 1)

    dx = gx - ax  # broadcast over anchors
    dy = gy - ay
    dist = np.sqrt(dx * dx + dy * dy) + EPS  # avoid div by zero

    weights = 1.0 / dist  # inverse-distance weighting
    w_sum = np.sum(weights, axis=-1, keepdims=True)  # (NY, NX, 1)
    weights_norm = weights / w_sum  # normalized weights

    # For each quantity, only use anchors where that quantity is not NaN
    # Temperature
    valid_temp = ~np.isnan(temps)
    if np.any(valid_temp):
        w_temp = weights_norm[:, :, valid_temp]
        t_vals = temps[valid_temp]
        temp_map = np.sum(w_temp * t_vals, axis=-1)

    # Humidity
    valid_hum = ~np.isnan(hums)
    if np.any(valid_hum):
        w_hum = weights_norm[:, :, valid_hum]
        h_vals = hums[valid_hum]
        hum_map = np.sum(w_hum * h_vals, axis=-1)

    # Light
    valid_light = ~np.isnan(lights)
    if np.any(valid_light):
        w_light = weights_norm[:, :, valid_light]
        l_vals = lights[valid_light]
        light_map = np.sum(w_light * l_vals, axis=-1)

    return grid_x, grid_y, temp_map, hum_map, light_map


def sample_at_position(
    x: float,
    y: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    temp_map: np.ndarray,
    hum_map: np.ndarray,
    light_map: np.ndarray,
) -> Dict[str, float]:
    """
    Sample the environment maps at a given (x,y) position using bilinear interpolation.

    Args:
        x, y: position in meters within the room coordinate frame.
        grid_x, grid_y: as returned by build_env_maps(...)
        temp_map, hum_map, light_map: same shapes as grid_x/grid_y

    Returns:
        {
          "temp_c": float or NaN,
          "hum_pct": float or NaN,
          "light_lux": float or NaN,
        }
    """
    # Clamp x,y to room bounds
    x_clamped = np.clip(x, 0.0, ROOM_W)
    y_clamped = np.clip(y, 0.0, ROOM_H)

    ny, nx = temp_map.shape  # note: (rows, cols)

    # Compute fractional indices
    fx = x_clamped / ROOM_W * (nx - 1)
    fy = y_clamped / ROOM_H * (ny - 1)

    ix0 = int(np.floor(fx))
    iy0 = int(np.floor(fy))
    ix1 = min(ix0 + 1, nx - 1)
    iy1 = min(iy0 + 1, ny - 1)

    dx = fx - ix0
    dy = fy - iy0

    def bilinear(m: np.ndarray) -> float:
        # Handle all-NaN edge cases gracefully
        q11 = m[iy0, ix0]
        q21 = m[iy0, ix1]
        q12 = m[iy1, ix0]
        q22 = m[iy1, ix1]

        if np.all(np.isnan([q11, q21, q12, q22])):
            return float("nan")

        vals = np.array([q11, q21, q12, q22], dtype=np.float32)
        if np.any(np.isnan(vals)):
            # simple approach: return mean of non-NaNs
            return float(np.nanmean(vals))

        # Standard bilinear interpolation
        return float(
            q11 * (1 - dx) * (1 - dy)
            + q21 * dx * (1 - dy)
            + q12 * (1 - dx) * dy
            + q22 * dx * dy
        )

    temp = bilinear(temp_map)
    hum = bilinear(hum_map)
    light = bilinear(light_map)

    return {
        "temp_c": temp,
        "hum_pct": hum,
        "light_lux": light,
    }


def compute_bed_env(
    recent_features: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Convenience helper:
      - Builds env maps from recent_features
      - Samples them at the bed position POS_BED

    Returns:
        dict with keys: "temp_c", "hum_pct", "light_lux"
    """
    grid_x, grid_y, temp_map, hum_map, light_map = build_env_maps(recent_features)
    x_bed, y_bed = POS_BED
    return sample_at_position(x_bed, y_bed, grid_x, grid_y, temp_map, hum_map, light_map)


# ----------------------------------------------------------------------
# Helpers for in-memory anchor collection
# ----------------------------------------------------------------------

def _latest_by_node(
    features: List[Dict[str, Any]], node_name: str
) -> Optional[Dict[str, Any]]:
    """Return the most recent feature dict for the given node, or None."""
    for f in reversed(features):
        if f.get("node") == node_name:
            return f
    return None


def _collect_anchors(
    recent_features: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Collect anchor points from latest readings of each node.

    Each anchor dict has:
        {
          "node": "window" or "bedside",
          "pos": (x, y),
          "temp": float or NaN,
          "hum": float or NaN,
          "light": float or NaN,
        }
    """
    anchors: List[Dict[str, Any]] = []

    # Window node
    win_msg = _latest_by_node(recent_features, "window")
    if win_msg is not None and "window" in NODE_POSITIONS:
        sx, sy = NODE_POSITIONS["window"]
        s = win_msg.get("sensors", {})
        temp_win = _to_float_or_nan(s.get("temp_win_c"))
        hum_win = _to_float_or_nan(s.get("hum_win_pct"))
        # window node may not have a light sensor; leave as NaN
        light_win = _to_float_or_nan(s.get("light_win_lux", None))
        anchors.append(
            {
                "node": "window",
                "pos": (sx, sy),
                "temp": temp_win,
                "hum": hum_win,
                "light": light_win,
            }
        )

    # Bedside node
    bed_msg = _latest_by_node(recent_features, "bedside")
    if bed_msg is not None and "bedside" in NODE_POSITIONS:
        bx, by = NODE_POSITIONS["bedside"]
        s = bed_msg.get("sensors", {})
        temp_bed = _to_float_or_nan(s.get("temp_bed_c"))
        hum_bed = _to_float_or_nan(s.get("hum_bed_pct"))
        light_bed = _to_float_or_nan(s.get("light_bed_lux"))
        anchors.append(
            {
                "node": "bedside",
                "pos": (bx, by),
                "temp": temp_bed,
                "hum": hum_bed,
                "light": light_bed,
            }
        )

    return anchors


def _to_float_or_nan(val: Any) -> float:
    """Convert a value to float, or NaN if not valid."""
    try:
        if val is None:
            return float("nan")
        v = float(val)
        if np.isnan(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------
# CSV-based loader (for visualization mode)
# ----------------------------------------------------------------------

def load_latest_features_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Load sensor_log.csv and return a list of "feature" dicts,
    keeping only the latest row per node.

    Each returned dict is of the form:
        {
          "node": <str>,
          "ts": <float>,
          "sensors": { <name>: <value>, ... }
        }
    """
    if not csv_path.exists():
        return []

    latest_by_node: Dict[str, Dict[str, Any]] = {}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = row.get("node", "unknown")
            ts_str = row.get("ts", "")
            try:
                ts = float(ts_str)
            except Exception:
                ts = 0.0

            sensors: Dict[str, Any] = {}
            for k, v in row.items():
                if not k.startswith("s_"):
                    continue
                sensor_name = k[2:]  # drop "s_"
                sensors[sensor_name] = _to_float_or_nan(v)

            latest_by_node[node] = {
                "node": node,
                "ts": ts,
                "sensors": sensors,
            }

    return list(latest_by_node.values())


# ----------------------------------------------------------------------
# Visualization mode: live heatmaps from CSV
# ----------------------------------------------------------------------

def _run_live_view(update_interval_sec: float = 30.0) -> None:
    """
    Live heatmap visualization:
      - reload logs/sensor_log.csv every update_interval_sec
      - build env maps from latest readings per node
      - update matplotlib figure with temp/hum/light heatmaps
    """
    if not LOG_FILE.exists():
        print(f"[ENV] Log file not found: {LOG_FILE}")
        print("      Start brain_server.py and ESP32 nodes first.")
        return

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax_temp, ax_hum, ax_light = axes

    # Keep references to colorbar objects so we can update cleanly
    cbar_temp = cbar_hum = cbar_light = None

    while True:
        features = load_latest_features_from_csv(LOG_FILE)
        if not features:
            print("[ENV] No features found yet in CSV; waiting...")
            time.sleep(update_interval_sec)
            continue

        grid_x, grid_y, temp_map, hum_map, light_map = build_env_maps(features)

        # Clear axes
        ax_temp.clear()
        ax_hum.clear()
        ax_light.clear()

        extent = [0.0, ROOM_W, 0.0, ROOM_H]

        # Temperature heatmap
        im1 = ax_temp.imshow(temp_map, origin="lower", extent=extent, aspect="auto")
        ax_temp.set_title("Temperature (°C)")
        ax_temp.set_xlabel("x (m)")
        ax_temp.set_ylabel("y (m)")
        if cbar_temp:
            cbar_temp.remove()
        cbar_temp = fig.colorbar(im1, ax=ax_temp, shrink=0.8)

        # Humidity heatmap
        im2 = ax_hum.imshow(hum_map, origin="lower", extent=extent, aspect="auto")
        ax_hum.set_title("Humidity (%)")
        ax_hum.set_xlabel("x (m)")
        if cbar_hum:
            cbar_hum.remove()
        cbar_hum = fig.colorbar(im2, ax=ax_hum, shrink=0.8)

        # Light heatmap
        im3 = ax_light.imshow(light_map, origin="lower", extent=extent, aspect="auto")
        ax_light.set_title("Light (lux)")
        ax_light.set_xlabel("x (m)")
        if cbar_light:
            cbar_light.remove()
        cbar_light = fig.colorbar(im3, ax=ax_light, shrink=0.8)

        fig.suptitle("Environment Maps (latest readings per node)")
        fig.tight_layout()

        plt.draw()
        plt.pause(0.1)

        print("[ENV] Heatmaps updated from CSV. Sleeping...")
        time.sleep(update_interval_sec)


if __name__ == "__main__":
    try:
        _run_live_view(update_interval_sec=30.0)  # change to 60.0 if you prefer 1 min
    except KeyboardInterrupt:
        print("\n[ENV] Stopped by user.")
