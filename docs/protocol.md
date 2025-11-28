# Sleep Environment Protocol

## 1. Feature vectors: ESP32 ? Raspberry Pi
Two JSON formats capture sensor features from each node.

### Bedside node example
```json
{
  "node": "bedside",
  "ts": 1732579200,
  "sensors": {
    "temp_bed_c": 23.4,
    "hum_bed_pct": 41.2,
    "light_bed_lux": 120.0,
    "noise_level": 0.37,
    "motion_index": 0.8
  }
}
```

### Window node example
```json
{
  "node": "window",
  "ts": 1732579200,
  "sensors": {
    "temp_win_c": 24.0,
    "hum_win_pct": 38.0,
    "light_win_lux": 200.0,
    "temp_out_c": 19.0,
    "hum_out_pct": 65.0
  }
}
```

**Field notes**
- `ts`: Unix timestamp (seconds).
- Units: `temp_*_c` in °C, `hum_*_pct` in %RH, `light_*_lux` in lux, `noise_level` as normalized [0,1], `motion_index` as normalized [0,1].
- Required: `node`, `ts`, `sensors` with indoor fields (`temp_bed_c`, `hum_bed_pct`, `light_bed_lux`, `noise_level`, `motion_index` for bedside; `temp_win_c`, `hum_win_pct`, `light_win_lux` for window). Optional: outdoor fields `temp_out_c`, `hum_out_pct` may be omitted when unavailable.

## 2. Sleep plan: Raspberry Pi ? ESP32s
Defines the predicted sleep state and environment targets.

```json
{
  "state": "WINDING_DOWN",
  "t_sleep_pred": 1732580400,
  "confidence": 0.78,
  "targets": {
    "temp_c": 22.0,
    "humidity_pct": 45.0,
    "max_light_lux": 30.0
  }
}
```

- `state` ? { "AWAKE", "WINDING_DOWN", "ASLEEP", "OUT_OF_ROOM" }.
- `t_sleep_pred`: Unix timestamp of predicted sleep onset.
- `confidence`: model certainty in [0, 1].
- `targets`: desired environmental setpoints near the bed (temperature °C, relative humidity %, maximum light level lux).

## 3. Actuation intent: ESP32 ? Raspberry Pi (for Kasa control)
Nodes declare desired plug states; Pi maps logical names to specific Kasa plugs.

### Bedside node
```json
{
  "node": "bedside",
  "ts": 1732579205,
  "desired": {
    "bed_lamp": "ON",
    "humidifier": "OFF"
  }
}
```

### Window node
```json
{
  "node": "window",
  "ts": 1732579205,
  "desired": {
    "fan": "ON"
  }
}
```

- Values in `desired` are "ON" or "OFF".
- The Pi resolves logical names (e.g., `bed_lamp`, `humidifier`, `fan`) to concrete Kasa plugs.