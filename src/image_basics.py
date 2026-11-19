
import cv2

# Read an image
image = cv2.imread("Images/sample.jpg")

# Check whether image was loaded
if image is None:
    print("Image not found!")
    exit()
print(image[100, 100])

# Display image information
print("Shape:", image.shape)
print("Dimensions:", image.ndim)
print("Data type:", image.dtype)

# Display the image
cv2.imshow("IrisMind - Original Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
