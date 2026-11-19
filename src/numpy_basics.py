
import numpy as np

# Create a simple image with 3 rows and 3 columns
image = np.array([
    [0, 100, 600],
    [50, 2250, 200],
    [25, 75, 125]
])
#we are using a 2D array to represent a grayscale image where each value corresponds to the intensity of a pixel. The values range from 0 (black) to 255 (white) for an 8-bit image, but in this case, we have values that exceed that range, which could represent a higher bit depth or simply be arbitrary values for testing purposes. 
print("Image:")
print(image)

print("Shape:", image.shape)
print("Dimensions:", image.ndim)
print("Data type:", image.dtype)

print("Brightest pixel:", np.max(image))
print("Darkest pixel:", np.min(image))
