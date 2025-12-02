import json
import socket
import time
import random

SERVER_HOST = "127.0.0.1"   
SERVER_PORT = 5000

HELLO_MSG = {
    "type": "hello",
    "node": "bedside"
}


def make_random_feature() -> dict:
    now = int(time.time())

    temp = random.uniform(20.0, 25.0)          
    hum = random.uniform(30.0, 60.0)          
    light = random.uniform(0.0, 200.0)        
    noise = random.uniform(0.0, 1.0)          
    motion = random.uniform(0.0, 1.0)         

    return {
        "node": "bedside",
        "ts": now,
        "sensors": {
            "temp_bed_c": temp,
            "hum_bed_pct": hum,
            "light_bed_lux": light,
            "noise_level": noise,
            "motion_index": motion,
        },
    }


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)  

    try:
        print("[CLIENT] Connecting to server...")
        sock.connect((SERVER_HOST, SERVER_PORT))
        print("[CLIENT] Connected to server")

        hello_line = json.dumps(HELLO_MSG) + "\n"
        print(f"[CLIENT] Sending hello: {HELLO_MSG}")
        sock.sendall(hello_line.encode("utf-8"))

        print("[CLIENT] Starting random feature loop + listening for plans (Ctrl+C to stop)...")

        sock.settimeout(2.0)

        buffer = b""
        last_feature_time = 0.0
        FEATURE_PERIOD = 1.0  

        while True:
            now = time.time()

            if now - last_feature_time >= FEATURE_PERIOD:
                feature_msg = make_random_feature()
                feature_line = json.dumps(feature_msg) + "\n"
                print(f"[CLIENT] Sending feature: {feature_msg}")
                sock.sendall(feature_line.encode("utf-8"))
                last_feature_time = now

            try:
                chunk = sock.recv(4096)
                if not chunk:
                    print("[CLIENT] Server closed connection")
                    break
                buffer += chunk

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                        print(f"[CLIENT] Received: {msg}")
                    except json.JSONDecodeError as e:
                        print(f"[CLIENT] JSON decode error: {e} on line: {line!r}")

            except socket.timeout:
                pass

            time.sleep(0.1)

    except Exception as e:
        print(f"[CLIENT] Error: {e}")
    finally:
        sock.close()
        print("[CLIENT] Closed connection")


if __name__ == "__main__":
    main()
