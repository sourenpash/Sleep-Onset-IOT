"""
generate_fake_night.py

Generate a fake "night" of data with:
  - bedside sensors (temp/hum/light/noise/motion)
  - window sensors (temp/hum)
  - camera state (AWAKE / IN_BED / ASLEEP)

Output CSV: fake_night.csv  with columns:
  minute, cam_state,
  temp_bed_c, hum_bed_pct, light_bed_lux, noise_level, motion_index,
  temp_win_c, hum_win_pct
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Dict

import random

OUT_PATH = Path(__file__).with_name("fake_night.csv")


def simulate_fake_night() -> List[Dict[str, float]]:
    """
    Simulate about 3 hours (180 minutes) of activity:

      0–60   min: AWAKE (lights on, more motion, more noise)
      60–90  min: WARMUP / WANDERING (lights a bit dimmer, motion moderate)
      90–105 min: IN_BED (lights lower, motion slowing)
      105–180 min: ASLEEP (very low light, motion, noise)

    We also simulate bed/window temp & humidity drifting slowly.
    """
    random.seed(42)

    rows: List[Dict[str, float]] = []

    total_min = 180
    for minute in range(total_min):
        # Camera state
        if minute < 60:
            cam_state = "AWAKE"
        elif minute < 90:
            cam_state = "AWAKE"  # still not in bed, just wandering
        elif minute < 105:
            cam_state = "IN_BED"
        else:
            cam_state = "ASLEEP"

        # Base environment temps: let's say the window area is a bit cooler
        # over time if "night".
        # We'll make it slightly sinusoidal to look less flat.
        base_out_temp = 23.0 - 0.5 * math.sin(minute / 60.0)
        base_out_hum = 45.0 + 2.0 * math.sin(minute / 45.0)

        # Bed temp tends to follow outside but with small lag / offset
        if minute < 90:
            temp_bed = base_out_temp + 1.0  # warmer earlier in evening
        else:
            temp_bed = base_out_temp + 0.5

        hum_bed = base_out_hum + 3.0

        # Add small random noise
        temp_bed += random.gauss(0.0, 0.2)
        hum_bed += random.gauss(0.0, 1.0)

        temp_win = base_out_temp + random.gauss(0.0, 0.2)
        hum_win = base_out_hum + random.gauss(0.0, 1.0)

        # Light & motion & noise patterns based on camera state
        if cam_state == "AWAKE":
            light_bed = random.uniform(200.0, 400.0)
            motion_idx = random.uniform(0.5, 0.9)
            noise_level = random.uniform(0.4, 0.8)
        elif cam_state == "IN_BED":
            light_bed = random.uniform(50.0, 150.0)
            motion_idx = random.uniform(0.2, 0.5)
            noise_level = random.uniform(0.2, 0.5)
        else:  # ASLEEP
            light_bed = random.uniform(1.0, 20.0)
            motion_idx = random.uniform(0.0, 0.2)
            noise_level = random.uniform(0.05, 0.25)

        row = {
            "minute": minute,
            "cam_state": cam_state,
            "temp_bed_c": round(temp_bed, 2),
            "hum_bed_pct": round(hum_bed, 2),
            "light_bed_lux": round(light_bed, 2),
            "noise_level": round(noise_level, 3),
            "motion_index": round(motion_idx, 3),
            "temp_win_c": round(temp_win, 2),
            "hum_win_pct": round(hum_win, 2),
        }
        rows.append(row)

    return rows


def main() -> None:
    rows = simulate_fake_night()

    fieldnames = [
        "minute",
        "cam_state",
        "temp_bed_c",
        "hum_bed_pct",
        "light_bed_lux",
        "noise_level",
        "motion_index",
        "temp_win_c",
        "hum_win_pct",
    ]

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[FAKE] Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
