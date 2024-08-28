import os
import time
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch

# Load your custom model
model = torch.hub.load(r'C:\Users\HHR6\PycharmProjects\ALPR\yolov5', 'custom', path='../Models/ocr_2.pt',
                       source='local')
model.conf = 0.28  # NMS confidence threshold
iou = 0.75  # NMS IoU threshold
model.agnostic = False

# Directory paths
input_dir = r'C:\Users\HHR6\PycharmProjects\AcksessionIntegration\LP_Detection\Yolo\output_Train\OLD_Iraq'
output_dir = r'./output8/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


# Function to resize image
def resize_image(input_path, size=(320, 320)):
    with Image.open(input_path) as img:
        resized_img = img.resize(size, Image.LANCZOS)
        return resized_img


# Function to draw text on image
def draw_text_on_image(image, text, font_size=60):  # Increase font_size for bigger text
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Use TrueType font with specified size
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if the specified font is not found

    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    image_width, image_height = image.size
    text_position = ((image_width - text_width) // 2, image_height - text_height - 10)
    draw.text(text_position, text, fill="red", font=font)

# Save the formatted strings to a text file
output_file_path = os.path.join(output_dir, 'arabic_ocr-results.txt')
with open(output_file_path, 'w') as f:
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        print(input_path)

        # Resize the image
        resized_image = resize_image(input_path)

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
        sorted_confidence = df_sorted['confidence'].tolist()
        print(sorted_confidence)
        sorted_classes = df_sorted['name'].tolist()
        print(sorted_classes)

        # Create the formatted string
        output_string = f"{filename} {' '.join(sorted_classes)} Inference time: {inference_time:.2f} seconds"

        # Draw text on the resized image
        draw_text_on_image(resized_image, ' '.join(sorted_classes))

        # Save the image with the text
        output_image_path = os.path.join(output_dir, f"text_{filename}")
        resized_image.save(output_image_path)

        # Write the formatted string to the file
        f.write(output_string + '\n')
