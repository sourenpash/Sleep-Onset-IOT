# discover_kasa.py
import asyncio
from kasa import Discover

async def main():
    print("[DISCOVER] Scanning your local network for Kasa devices...")
    devices = await Discover.discover()

    if not devices:
        print("[DISCOVER] No Kasa devices found. Make sure:")
        print("  - You're on the same WiFi as the plugs.")
        print("  - The plugs are powered on and connected.")
        return

    for addr, dev in devices.items():
        await dev.update()
        print("--------------------------------------------------")
        print(f"IP:       {addr}")
        print(f"Alias:    {dev.alias}")
        print(f"Model:    {dev.model}")
        print(f"On/Off:   {'ON' if dev.is_on else 'OFF'}")

if __name__ == "__main__":
    asyncio.run(main())
