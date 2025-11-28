"""
Stub interface for controlling Kasa smart plugs.
"""

from typing import NoReturn


def set_device_state(name: str, state: str) -> None:
    """
    Stub for controlling Kasa smart plugs.
    name: logical device name, e.g. 'bed_lamp', 'humidifier', 'fan'.
    state: 'ON' or 'OFF'.
    """
    print(f"[SIM KASA] Setting {name} -> {state}")


if __name__ == "__main__":
    set_device_state("bed_lamp", "ON")