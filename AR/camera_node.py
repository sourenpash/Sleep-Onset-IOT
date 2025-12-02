import json
import socket
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# ==== CONFIGURE THESE ====
SERVER_IP = "10.0.0.32"  # SERVER/PC IP ADDRESS
SERVER_PORT = 5000

CAMERA_INDEX = 0           
UPDATE_INTERVAL_SEC = 10.0 # how often to send an activity message
# ==========================

HELLO_MSG = {"type": "hello", "node": "camera"}

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def classify_activity(
    landmarks: Optional[landmark_pb2.NormalizedLandmarkList],
    frame_width: int,
    frame_height: int,
    no_person_secs: float,
) -> Tuple[str, float]:

    # If no landmarks for some time, we treat as AWAY
    if landmarks is None:
        if no_person_secs > 10.0:
            return "AWAY", 0.9
        else:
            # short gaps, keep last state or treat as low-confidence AWAY
            return "AWAY", 0.3

    lm = landmarks.landmark

    # MediaPipe Pose landmark indices:
    # LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12
    # LEFT_HIP = 23, RIGHT_HIP = 24
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    try:
        ls = lm[LEFT_SHOULDER]
        rs = lm[RIGHT_SHOULDER]
        lh = lm[LEFT_HIP]
        rh = lm[RIGHT_HIP]
    except IndexError:
        return "AWAY", 0.1

    # Body orientation: vector from hips to shoulders
    hip_center_x = (lh.x + rh.x) / 2.0
    hip_center_y = (lh.y + rh.y) / 2.0
    shoulder_center_x = (ls.x + rs.x) / 2.0
    shoulder_center_y = (ls.y + rs.y) / 2.0

    dx = shoulder_center_x - hip_center_x
    dy = shoulder_center_y - hip_center_y

    # Compare vertical vs horizontal component.
    # If |dy| > |dx| -> mostly vertical in image coordinates.
    body_vertical = abs(dy) > abs(dx)

    if body_vertical:
        # Torso more vertical in the image → upright-ish (sitting/standing)
        return "AWAKE_IN_ROOM", 0.8
    else:
        # Torso more horizontal → lying down
        return "IN_BED_LYING", 0.8


def main():
    # --- Connect to brain server ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)

    print("[CAM] Connecting to server...")
    sock.connect((SERVER_IP, SERVER_PORT))
    print("[CAM] Connected to server")

    hello_line = json.dumps(HELLO_MSG) + "\n"
    sock.sendall(hello_line.encode("utf-8"))
    print("[CAM] Sent hello:", HELLO_MSG)

    # --- Camera & MediaPipe setup ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[CAM] ERROR: Could not open camera")
        sock.close()
        return

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    last_send_time = 0.0
    last_detection_time = 0.0
    last_state = "AWAY"
    last_conf = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAM] WARNING: Failed to grab frame")
                time.sleep(0.1)
                continue

            frame_height, frame_width, _ = frame.shape

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks
                last_detection_time = time.time()
            else:
                landmarks = None

            # How long since we last saw a person?
            no_person_secs = time.time() - last_detection_time

            activity_state, activity_conf = classify_activity(
                landmarks, frame_width, frame_height, no_person_secs
            )

            # Keep a short memory so state doesn't flicker
            if activity_state == "AWAY" and activity_conf < 0.5:
                activity_state = last_state
                activity_conf = last_conf * 0.9

            # Debug view
            debug_frame = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    debug_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

            cv2.putText(
                debug_frame,
                f"Activity: {activity_state} ({activity_conf:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Camera Node (press q to quit)", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Send an update every UPDATE_INTERVAL_SEC
            now = time.time()
            if now - last_send_time >= UPDATE_INTERVAL_SEC:
                msg = {
                    "node": "camera",
                    "ts": now,
                    "sensors": {
                        "activity_state": activity_state,
                        "activity_conf": activity_conf,
                    },
                }
                line = json.dumps(msg) + "\n"
                try:
                    sock.sendall(line.encode("utf-8"))
                    print("[CAM] Sent activity:", msg)
                except Exception as e:
                    print(f"[CAM] Error sending to server: {e}")
                    break

                last_send_time = now
                last_state = activity_state
                last_conf = activity_conf

    except KeyboardInterrupt:
        print("[CAM] KeyboardInterrupt, exiting")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        print("[CAM] Closed connection")


if __name__ == "__main__":
    main()
