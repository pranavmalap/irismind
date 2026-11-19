
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    exit()

while True:

    success, frame = camera.read()

    if not success:
        break



    # Select a region of interest
    roi = frame[100:300, 150:450]

    # Display original frame
    cv2.imshow("Original Frame", frame)

    # Display cropped region
    cv2.imshow("Region of Interest", roi)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()