import os
from PIL import Image
import torch

# Load your custom model
model = torch.hub.load(r'C:\Users\HHR6\PycharmProjects\Testing_Pipeline\yolov5', 'custom',force_reload=True, path='./Models/IranLPDetection3.pt',source="local")
model.conf = 0.45  # NMS confidence threshold
iou = 0.45  # NMS IoU threshold
model.agnostic = True

# Load the second model (assuming a second model is required)
second_model = torch.hub.load(r'C:\Users\HHR6\PycharmProjects\Testing_Pipeline\yolov5', 'custom',force_reload=True, path='./Models/IRAN_Iraq.pt',source="local")
second_model.conf = 0.45
second_model.agnostic = True

# Directory paths
input_dir = r'C:\Users\HHR6\PycharmProjects\Testing_Pipeline\Acksession-Project/yolov8_compiled_dataset/test/images'
output_dir = r'./output_Iraq_dataset'
old_iraq_dir = os.path.join(output_dir, 'OLD_Iraq')
new_iraq_dir = os.path.join(output_dir, 'New_Iraq')
iran_dir = os.path.join(output_dir, 'Iran')

# Create necessary directories if they don't exist
os.makedirs(old_iraq_dir, exist_ok=True)
os.makedirs(new_iraq_dir,exist_ok=True)
os.makedirs(iran_dir, exist_ok=True)

# Function to resize image
def resize_image(input_path, size=(640, 640)):
    with Image.open(input_path) as img:
        resized_img = img.resize(size, Image.LANCZOS)
        return resized_img

# Function to extend the bounding box
def extend_bbox(bbox, image_size, factor=4):
    x1, y1, x2, y2 = bbox
    width, height = image_size

    w = x2 - x1
    h = y2 - y1

    x1_new = max(x1 - w * (factor - 1) // 2, 0)
    y1_new = max(y1 - h * (factor - 1) // 2, 0)
    x2_new = min(x2 + w * (factor - 1) // 2, width)
    y2_new = min(y2 + h * (factor - 1) // 2, height)

    return (x1_new, y1_new, x2_new, y2_new)

# Function for inference
def inference_image(input_path):
    resized_image = resize_image(input_path)
    results = model(resized_image)
    df = results.pandas().xyxy[0]
    print(df)
    for index, row in df.iterrows():
        class_id = int(row['class'])
        bbox = [0.95*int(row['xmin']), 0.95*int(row['ymin']), 1.05*int(row['xmax']), 1.05*int(row['ymax'])]

        if class_id == 0 or class_id==1:  # Class ID 2 for extending the bounding box
            # Extend the bounding box by a factor of 4
            extended_bbox = extend_bbox(bbox, resized_image.size, factor=4)

            # Crop the extended image
            extended_cropped_image = resized_image.crop(extended_bbox)

            # Save the extended cropped image (optional)



            # Use the second model for further processing
            second_results = second_model(extended_cropped_image)
            second_df = second_results.pandas().xyxy[0]

            # Print the results of the second model

            for index, row in second_df.iterrows():
                class_id = int(row['class'])
                bbox = [0.95*int(row['xmin']), 0.95*int(row['ymin']), 1.05*int(row['xmax']), 1.05*int(row['ymax'])]
                if class_id == 0:
                    cropped_image=extended_cropped_image.crop(bbox)
                    save_path=os.path.join(new_iraq_dir,os.path.basename(input_path))
                    cropped_image.save(save_path)
                else:
                    cropped_image = extended_cropped_image.crop(bbox)
                    save_path = os.path.join(iran_dir, os.path.basename(input_path))
                    cropped_image.save(save_path)

        else:
            # Save the bounding box image if it's not class 2
            cropped_image = resized_image.crop(bbox)
            save_path = os.path.join(old_iraq_dir, os.path.basename(input_path))
            cropped_image.save(save_path)


def inference_directory(input_dir):
    paths=os.listdir(input_dir)
    for path in paths:
        inference_image(os.path.join(input_dir,path))

# Test the function
#inference_image(r"C:\Users\HHR6\PycharmProjects\Testing_Pipeline\Acksession-Project\yolov8_compiled_dataset\test\images\1_jpg.rf.e0d71bfac749526f7f57a7274f7f8f1e.jpg")
inference_directory(input_dir)