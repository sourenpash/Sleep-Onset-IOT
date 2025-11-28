# Pi Brain (stub)

This folder holds the Raspberry Pi brain stubs in offline simulation mode.

- `brain_server.py` - networked driver that starts the TCP server, consumes feature/actuation messages, computes dummy plans, and simulates applying actuation intents.
- `sleep_model_stub.py` - placeholder model returning a constant sleep plan.
- `kasa_control_stub.py` - stub interface to Kasa smart plugs (prints only).
- `network_server.py` - lightweight TCP server for newline-delimited JSON from ESP32 clients.

Networking and real device control will be added later.

## Manual test (stub mode)
- Run `python brain_server.py` on the Pi (or dev machine) to start the TCP server.
- Flash/compile the dummy `esp32-bedside/main.cpp` and `esp32-window/main.cpp`, then open their Serial monitors.
- Verify that each ESP32 connects over Wi-Fi, sends the `hello` JSON, and the server logs the registration.
- Watch the server log incoming feature JSON; every ~10s it prints and broadcasts a sleep plan.
- ESP32 logs should show the received plan state; when ESP32s send desired actions, the server prints `[SIM KASA] Setting ...`.