

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List

import sleep_model_stub  


CSV_PATH = Path(__file__).with_name("fake_night.csv")


def load_fake_data() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with CSV_PATH.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["minute"] = int(row["minute"])
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
    return rows


def row_to_recent_features(history: List[Dict[str, Any]], idx: int) -> List[Dict[str, Any]]:

    recent_features: List[Dict[str, Any]] = []

    for j in range(0, idx + 1):
        row = history[j]
        ts = float(row["minute"] * 60)  

        bedside_msg = {
            "node": "bedside",
            "ts": ts,
            "sensors": {
                "temp_bed_c": row["temp_bed_c"],
                "hum_bed_pct": row["hum_bed_pct"],
                "light_bed_lux": row["light_bed_lux"],
                "noise_level": row["noise_level"],
                "motion_index": row["motion_index"],
            },
        }
        recent_features.append(bedside_msg)

        window_msg = {
            "node": "window",
            "ts": ts,
            "sensors": {
                "temp_win_c": row["temp_win_c"],
                "hum_win_pct": row["hum_win_pct"],
            },
        }
        recent_features.append(window_msg)

    return recent_features


def main() -> None:
    if not CSV_PATH.exists():
        print(f"[TEST] {CSV_PATH} not found. Run generate_fake_night.py first.")
        return

    rows = load_fake_data()
    print(f"[TEST] Loaded {len(rows)} minutes from {CSV_PATH}")

    base_epoch = time.time()

    print(
        "minute, cam_state, model_state, p_sleep_30, "
        "minutes_until_predicted_sleep"
    )

    for i, row in enumerate(rows):
        recent_features = row_to_recent_features(rows, i)

        plan = sleep_model_stub.compute_sleep_plan(recent_features)
        debug = plan.get("debug", {})

        p_sleep_30 = debug.get("p_sleep_within_30_min", None)
        if p_sleep_30 is None:
            p_str = "  N/A"
        else:
            try:
                p_str = f"{float(p_sleep_30):5.3f}"
            except (TypeError, ValueError):
                p_str = "  N/A"

        now = base_epoch + row["minute"] * 60.0
        t_sleep_pred = plan["t_sleep_pred"]
        minutes_until_pred = (t_sleep_pred - now) / 60.0

        print(
            f"{row['minute']:3d}, "
            f"{row['cam_state']:<7}, "
            f"{plan['state']:<12}, "
            f"{p_str}, "
            f"{minutes_until_pred:6.1f}"
        )


    print("[TEST] Done.")


if __name__ == "__main__":
    main()
