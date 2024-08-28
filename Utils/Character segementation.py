import cv2
import numpy as np

# Load the image
image = cv2.imread(r'C:\Users\HHR6\PycharmProjects\AcksessionIntegration\Image_processing\adjusted_Iraq\a-268-b_JPG_jpg.rf.2ec3c82709608f87c3cc182b502aa18b.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours from left to right
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

# Loop through each contour to segment characters
for i, contour in enumerate(contours):
    # Get the bounding box for each contour
    x, y, w, h = cv2.boundingRect(contour)

    # Extract the character using the bounding box
    char = image[y:y + h, x:x + w]

    # Save the segmented character image
    char_image_path = f'segmented_char_{i}.png'
    cv2.imwrite(char_image_path, char)

    # Optionally, draw the bounding box on the original image for visualization
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

# Show the final image with bounding boxes (optional)
cv2.imshow('Segmented Characters', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
