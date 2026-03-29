from collections import deque
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hand_tracker_mplconfig")

import cv2
import mediapipe as mp
import mido
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


MIDI_PORT_NAME = "IAC Driver PythonToLogic"  # Match your IAC port name in MIDI Studio
SMOOTHING_WINDOW = 5  # Number of frames to average to reduce jitter
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def send_cc(outport, value, cc_num):
    value = max(0, min(127, int(value)))
    msg = mido.Message("control_change", control=cc_num, value=value)
    outport.send(msg)


def smooth_value(queue, new_value):
    queue.append(new_value)
    if len(queue) > SMOOTHING_WINDOW:
        queue.popleft()
    return sum(queue) / len(queue)


def normalized_to_pixel_coordinates(x, y, width, height):
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return min(int(x * width), width - 1), min(int(y * height), height - 1)


def draw_hand_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape
    pixel_coordinates = {}

    for index, landmark in enumerate(hand_landmarks):
        point = normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
        if point is None:
            continue
        pixel_coordinates[index] = point

    for connection in HAND_CONNECTIONS:
        start = pixel_coordinates.get(connection.start)
        end = pixel_coordinates.get(connection.end)
        if start and end:
            cv2.line(frame, start, end, (0, 255, 0), 2)

    for point in pixel_coordinates.values():
        cv2.circle(frame, point, 4, (255, 255, 255), -1)
        cv2.circle(frame, point, 2, (0, 0, 255), -1)


def create_hand_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing hand landmarker model at {MODEL_PATH}. "
            "Place hand_landmarker.task in the models folder."
        )

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def process_hand(frame, result, hand_index, outport, left_hand_smoothers, right_hand_smoothers):
    handedness = result.handedness[hand_index][0].category_name.lower()
    hand_landmarks = result.hand_landmarks[hand_index]
    draw_hand_landmarks(frame=frame, hand_landmarks=hand_landmarks)

    wrist = hand_landmarks[0]
    smoothers = left_hand_smoothers if handedness == "left" else right_hand_smoothers
    x = smooth_value(smoothers["x"], wrist.x)
    y = smooth_value(smoothers["y"], wrist.y)
    z = smooth_value(smoothers["z"], wrist.z)

    x_midi = int(x * 127)
    y_midi = int((1 - y) * 127)
    z_midi = int((1 + z) * 64)

    if handedness == "left":
        send_cc(outport, x_midi, 1)
        send_cc(outport, y_midi, 2)
        send_cc(outport, z_midi, 3)
    elif handedness == "right":
        send_cc(outport, x_midi, 4)
        send_cc(outport, y_midi, 5)
        send_cc(outport, z_midi, 6)


def main():
    left_hand_smoothers = {"x": deque(), "y": deque(), "z": deque()}
    right_hand_smoothers = {"x": deque(), "y": deque(), "z": deque()}

    with create_hand_landmarker() as hand_landmarker:
        outport = mido.open_output(MIDI_PORT_NAME)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Failed to open webcam")
            return

        print("🎥 Webcam active — Press 'q' to quit")

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("❌ Failed to read frame")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.monotonic() * 1000)
                result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

                for hand_index in range(len(result.hand_landmarks)):
                    process_hand(
                        frame,
                        result,
                        hand_index,
                        outport,
                        left_hand_smoothers,
                        right_hand_smoothers,
                    )

                cv2.imshow("Hand Tracker MIDI", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            outport.close()

    print("👋 Exiting. MIDI + Webcam closed.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        print(f"❌ {error}")
