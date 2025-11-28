# Networking Protocol

Transport: TCP over Wi-Fi using newline-delimited JSON messages.

Topology:
- Raspberry Pi: TCP server listening on `0.0.0.0:5000`.
- ESP32-Bedside and ESP32-Window: TCP clients that connect to the Pi.

Message format:
- Each message is one JSON object, UTF-8 encoded, terminated by `\n`.
- Payload structures match the feature, sleep plan, and actuation intent examples in `docs/protocol.md`.

Handshake:
- On connect, each ESP32 immediately sends:
  ```json
  { "type": "hello", "node": "bedside" }
  ```
  or
  ```json
  { "type": "hello", "node": "window" }
  ```
- The server remembers which socket belongs to which node for targeted sends.
