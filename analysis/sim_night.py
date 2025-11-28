"""
Synthetic night simulator stub for testing sleep-state logic on fake signals.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List


def generate_timeline(start: datetime, end: datetime, step_minutes: int = 10) -> List[datetime]:
    """Return timestamps from start to end inclusive in fixed increments."""
    points: List[datetime] = []
    current = start
    delta = timedelta(minutes=step_minutes)
    while current <= end:
        points.append(current)
        current += delta
    return points


def build_fake_signals(length: int) -> Dict[str, List[float]]:
    """Create simple ramps/oscillations for sensor placeholders."""
    light_bed_lux = [max(0.0, 200 - i * 3.5) for i in range(length)]
    motion_index = [0.8 if i < length // 4 else 0.2 for i in range(length)]
    noise_level = [0.4 if i < length // 3 else 0.15 for i in range(length)]
    temp_bed_c = [23.0 - 0.01 * i for i in range(length)]
    temp_win_c = [22.0 - 0.015 * i for i in range(length)]
    return {
        "light_bed_lux": light_bed_lux,
        "motion_index": motion_index,
        "noise_level": noise_level,
        "temp_bed_c": temp_bed_c,
        "temp_win_c": temp_win_c,
    }


def main() -> None:
    start = datetime.strptime("20:00", "%H:%M")
    end = datetime.strptime("08:00", "%H:%M") + timedelta(days=1)
    timeline = generate_timeline(start, end, step_minutes=10)
    signals = build_fake_signals(len(timeline))

    print(f"Generated {len(timeline)} samples from {timeline[0]} to {timeline[-1]}.")
    print("Preview (first 3 entries):")
    for i in range(3):
        print({
            "time": timeline[i].strftime("%H:%M"),
            "light_bed_lux": signals["light_bed_lux"][i],
            "motion_index": signals["motion_index"][i],
            "noise_level": signals["noise_level"][i],
            "temp_bed_c": signals["temp_bed_c"][i],
            "temp_win_c": signals["temp_win_c"][i],
        })

    # TODO: implement heuristics to assign states AWAKE / WINDING_DOWN / ASLEEP.
    # TODO: plot signals vs. inferred state for quick inspection.


if __name__ == "__main__":
    main()