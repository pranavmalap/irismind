
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        break

    # Print frame information
    print("Frame shape:", frame.shape)
    print("Data type:", frame.dtype)

    cv2.imshow("IrisMind", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
