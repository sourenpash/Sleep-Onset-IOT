"""
kasa_control_stub.py

Control Kasa-based actuators for:
  - humidifier
  - dehumidifier
  - heater

You can:
  - import set_device_state / set_humidifier / set_dehumidifier / set_heater
  - or run this file directly from the command line to test.

Requires:
    pip install python-kasa

IMPORTANT:
  - Each device ID below can be an IP ("10.0.0.xx") or an alias you set in
    the Kasa app.
  - The humidifier/dehumidifier/heater themselves must be left in an
    "ON" state so that when the plug powers them they actually start running.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Dict

try:
    from kasa import SmartPlug  # python-kasa
except ImportError:
    SmartPlug = None
    print("[KASA] WARNING: python-kasa not installed. This stub will only print actions.")


# ---------------------------------------------------------------------
# CONFIG: set these to your actual devices
# ---------------------------------------------------------------------
# You can use either IPs or aliases here.
HUMIDIFIER_DEVICE_ID   = "10.0.0.164"
DEHUMIDIFIER_DEVICE_ID = "10.0.0.164"   # for now, same plug if you only have one
HEATER_DEVICE_ID       = "10.0.0.164"   # same here, just for testing
  # e.g. "10.0.0.62" or alias

# Map logical names (what the brain server uses) -> physical device IDs
DEVICE_MAP: Dict[str, str] = {
    "humidifier":   HUMIDIFIER_DEVICE_ID,
    "dehumidifier": DEHUMIDIFIER_DEVICE_ID,
    "heater":       HEATER_DEVICE_ID,
}


# ---------------------------------------------------------------------
# Core async helper
# ---------------------------------------------------------------------

async def _set_kasa_async(device_id: str, on: bool) -> None:
    """
    Async helper: connect to a Kasa plug and set on/off.
    device_id can be an IP or an alias discoverable on the LAN.
    """
    if SmartPlug is None:
        # python-kasa not available, just log
        print(f"[KASA] (DRY RUN) {device_id}: would set to {'ON' if on else 'OFF'}")
        return

    try:
        plug = SmartPlug(device_id)
        await plug.update()
        if on:
            await plug.turn_on()
        else:
            await plug.turn_off()
        print(f"[KASA] {device_id}: set to {'ON' if on else 'OFF'}")
    except Exception as e:
        print(f"[KASA] Error controlling {device_id}: {e}")


def _run_async(coro) -> None:
    """
    Run an async coroutine, handling the case where an event loop already exists.
    """
    try:
        asyncio.run(coro)
    except RuntimeError:
        # if we're already inside an event loop, create a new one
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()


# ---------------------------------------------------------------------
# Public API for brain_server and tests
# ---------------------------------------------------------------------

def set_device_state(name: str, on: bool) -> None:
    """
    Generic entry point. 'name' is a logical name:

        'humidifier', 'dehumidifier', 'heater'

    or a raw device ID (IP/alias).
    """
    # Map logical name to device_id, or treat name as device_id directly
    device_id = DEVICE_MAP.get(name, name)
    print(f"[KASA] Request: {name} -> {device_id} = {'ON' if on else 'OFF'}")
    _run_async(_set_kasa_async(device_id, on))


def set_humidifier(on: bool) -> None:
    set_device_state("humidifier", on)


def set_dehumidifier(on: bool) -> None:
    set_device_state("dehumidifier", on)


def set_heater(on: bool) -> None:
    set_device_state("heater", on)


# ---------------------------------------------------------------------
# CLI tester
# ---------------------------------------------------------------------

def _print_usage() -> None:
    print("Usage:")
    print("  python kasa_control_stub.py <device> <on|off>")
    print()
    print("Where <device> is one of:")
    print("  humidifier, dehumidifier, heater")
    print("or a raw Kasa IP/alias (bypassing DEVICE_MAP).")
    print()
    print("Examples:")
    print("  python kasa_control_stub.py humidifier on")
    print("  python kasa_control_stub.py heater off")
    print("  python kasa_control_stub.py 10.0.0.75 on  # direct IP")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        _print_usage()
        sys.exit(1)

    device_name = sys.argv[1]
    state_str = sys.argv[2].lower()

    if state_str not in ("on", "off"):
        print("[KASA] Invalid state; must be 'on' or 'off'.")
        _print_usage()
        sys.exit(1)

    on_flag = state_str == "on"
    set_device_state(device_name, on_flag)
