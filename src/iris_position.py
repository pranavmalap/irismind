
import cv2
import mediapipe as mp
import time
from pathlib import Path

# Find model file
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

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


def calculate_iris_position(landmarks, iris_index,
                            eye_left_index, eye_right_index,
                            eye_top_index, eye_bottom_index):

    # Iris centre
    iris = landmarks[iris_index]

    # Eye corners
    corner1 = landmarks[eye_left_index]
    corner2 = landmarks[eye_right_index]

    # Eye vertical boundaries
    top = landmarks[eye_top_index]
    bottom = landmarks[eye_bottom_index]

    # Horizontal boundaries
    x_min = min(corner1.x, corner2.x)
    x_max = max(corner1.x, corner2.x)

    # Vertical boundaries
    y_min = min(top.y, bottom.y)
    y_max = max(top.y, bottom.y)

    eye_width = x_max - x_min
    eye_height = y_max - y_min

    if eye_width == 0 or eye_height == 0:
        return None

    # Calculate relative iris position
    relative_x = (iris.x - x_min) / eye_width
    relative_y = (iris.y - y_min) / eye_height

    return relative_x, relative_y


frame_count = 0

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

                # MediaPipe iris and eye landmark indices
                eye_config = {
                    "Right Eye": (468, 33, 133, 159, 145),
                    "Left Eye": (473, 362, 263, 386, 374)
                }

                for eye_name, indices in eye_config.items():

                    iris_index = indices[0]
                    eye_left = indices[1]
                    eye_right = indices[2]
                    eye_top = indices[3]
                    eye_bottom = indices[4]

                    position = calculate_iris_position(
                        landmarks,
                        iris_index,
                        eye_left,
                        eye_right,
                        eye_top,
                        eye_bottom
                    )

                    if position is not None:

                        iris_x, iris_y = position

                        text = (
                            f"{eye_name}: "
                            f"X={iris_x:.2f}, Y={iris_y:.2f}"
                        )

                        y_position = (
                            40 if eye_name == "Right Eye" else 80
                        )

                        cv2.putText(
                            frame,
                            text,
                            (20, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2
                        )

                        # Draw iris centre
                        iris = landmarks[iris_index]

                        px = int(iris.x * width)
                        py = int(iris.y * height)

                        cv2.circle(
                            frame,
                            (px, py),
                            5,
                            (0, 0, 255),
                            -1
                        )

                        # Print measurements occasionally
                        if frame_count % 30 == 0:
                            print(
                                f"{eye_name}: "
                                f"X={iris_x:.3f}, "
                                f"Y={iris_y:.3f}"
                            )

            cv2.imshow("IrisMind - Iris Position", frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
