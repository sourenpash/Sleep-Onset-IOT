
from __future__ import annotations

import asyncio
import sys
from typing import Dict

try:
    from kasa import SmartPlug 
except ImportError:
    SmartPlug = None
    print("[KASA] WARNING: python-kasa not installed. This stub will only print actions.")



HUMIDIFIER_DEVICE_ID   = "10.0.0.164"
DEHUMIDIFIER_DEVICE_ID = "10.0.0.164"  
HEATER_DEVICE_ID       = "10.0.0.164"   

DEVICE_MAP: Dict[str, str] = {
    "humidifier":   HUMIDIFIER_DEVICE_ID,
    "dehumidifier": DEHUMIDIFIER_DEVICE_ID,
    "heater":       HEATER_DEVICE_ID,
}


async def _set_kasa_async(device_id: str, on: bool) -> None:
   
    if SmartPlug is None:
       
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
   
    try:
        asyncio.run(coro)
    except RuntimeError:
       
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()


def set_device_state(name: str, on: bool) -> None:
   
    device_id = DEVICE_MAP.get(name, name)
    print(f"[KASA] Request: {name} -> {device_id} = {'ON' if on else 'OFF'}")
    _run_async(_set_kasa_async(device_id, on))


def set_humidifier(on: bool) -> None:
    set_device_state("humidifier", on)


def set_dehumidifier(on: bool) -> None:
    set_device_state("dehumidifier", on)


def set_heater(on: bool) -> None:
    set_device_state("heater", on)


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
