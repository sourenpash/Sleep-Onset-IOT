# Sleep Environment Project (stubs)

Three-device system in stub form:
- Raspberry Pi "Pi Brain" for sleep-state prediction and Kasa control.
- ESP32 "Bedside Node" sensing bed-adjacent environment.
- ESP32 "Window Node" sensing window/outdoor environment.

Folders:
- `docs/` — JSON protocol and architecture notes.
- `pi-brain/` — central brain stubs (model + Kasa shim + offline driver).
- `esp32-bedside/` and `esp32-window/` — firmware skeletons emitting fake data.
- `analysis/` — simulation and early modelling sketches.

Everything currently runs on fake data and prints only; real sensors, networking, and Kasa control will be added later.