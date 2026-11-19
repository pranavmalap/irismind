
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        break

    # Original image
    cv2.imshow("Original", frame)

    # Increase brightness
    bright = cv2.convertScaleAbs(frame, alpha=1, beta=500)
    cv2.imshow("Bright", bright)

    # Increase contrast
    contrast = cv2.convertScaleAbs(frame, alpha=2, beta=0)
    cv2.imshow("Contrast", contrast)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
