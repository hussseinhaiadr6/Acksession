import os
import time
from PIL import Image
from ultralytics import YOLO
import torch

# Load your custom model
model = torch.hub.load(r'C:\Users\HHR6\PycharmProjects\ALPR\yolov5', 'custom', path='../Models/district_ocr.pt', source='local')
model.conf = 0.35  # NMS confidence threshold
iou = 0.45  # NMS IoU threshold
model.agnostic = False

# Directory paths
input_dir = r'C:\Users\HHR6\PycharmProjects\AcksessionIntegration\LP_Recognition\Arabic_OCR\Iraq_Districts\images\content\arabic_dataset-3\test\images/'


# Create the resized_images directory if it doesn't exist


# Function to resize image
def resize_image(input_path, size=(320, 320)):
    with Image.open(input_path) as img:
        resized_img = img.resize(size, Image.LANCZOS)
        #resized_img.save(output_path)
        return resized_img


# Save the formatted strings to a text file
output_file_path = 'arabic_ocr-results.txt'
with open(output_file_path, 'w') as f:
    for filename in os.listdir(input_dir):
        print(os.path.join(input_dir,filename))
        resized_image=resize_image(os.path.join(input_dir,filename))
        # Image file path


        # Measure inference time
        start_time = time.time()

        # Run inference on the resized image
        results = model(resized_image)

        end_time = time.time()
        inference_time = end_time - start_time

        # Convert results to pandas DataFrame
        df = results.pandas().xyxy[0]

        # Sort detections from right to left (based on xmax in descending order)
        df_sorted = df.sort_values(by='xmax', ascending=True)

        # Get the class names in sorted order
        sorted_classes = df_sorted['name'].tolist()
        print(sorted_classes)
        # Create the formatted string
        output_string = f"{filename} {' '.join(sorted_classes)} Inference time: {inference_time:.2f} seconds"

        # Write the formatted string to the file
        f.write(output_string + '\n')
