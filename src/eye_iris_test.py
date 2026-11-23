
import cv2
import mediapipe as mp
import time
from pathlib import Path

# Locate the model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

# MediaPipe configuration
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

try:
    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                break

            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = time.monotonic_ns() // 1_000_000

            results = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            if results.face_landmarks:

                landmarks = results.face_landmarks[0]

                # Iris landmark ranges
                iris_groups = {
                    "Right Iris": range(468, 473),
                    "Left Iris": range(473, 478)
                }

                for eye_name, indices in iris_groups.items():

                    for index in indices:

                        landmark = landmarks[index]

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            3,
                            (0, 255, 0),
                            -1
                        )

                # Draw eye contours using selected points
                eye_groups = {
                    "Right Eye": [33, 160, 158, 133, 153, 144],
                    "Left Eye": [362, 385, 387, 263, 373, 380]
                }

                for eye_name, indices in eye_groups.items():

                    points = []

                    for index in indices:
                        landmark = landmarks[index]

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        points.append((x, y))

                    for point in points:
                        cv2.circle(
                            frame,
                            point,
                            2,
                            (255, 0, 0),
                            -1
                        )

                cv2.putText(
                    frame,
                    "Eye and Iris Detection",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("IrisMind - Eye Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
