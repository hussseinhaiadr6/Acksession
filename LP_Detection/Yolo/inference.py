import os
import time

from PIL import Image
import torch
import cv2
import numpy as np
from LP_Recognition.Farsi_OCR.OCR import inference_image_farsi
from LP_Recognition.Farsi_OCR.OCR import inference_image_farsi
# Load your custom model
model = torch.hub.load(r'C:\Users\HHR6\PycharmProjects\Testing_Pipeline\yolov5', 'custom',force_reload=True, path='./Models/IranLPDetection3.onnx',source="local")
model.conf = 0.45  # NMS confidence threshold
iou = 0.45  # NMS IoU threshold
model.agnostic = True

# Load the second model (assuming a second model is required)
second_model = torch.hub.load(r'C:\Users\HHR6\PycharmProjects\Testing_Pipeline\yolov5', 'custom',force_reload=True, path='./Models/IRAN_Iraq.pt',source="local")
second_model.conf = 0.45
second_model.agnostic = True

# Directory paths
input_dir = r'C:\Users\HHR6\PycharmProjects\AcksessionIntegration\Datasets\no_benchmark\valid\images'
output_dir = r'./output_Iraq_dataset_valid'
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
    start_time = time.time()
    resized_image = resize_image(input_path)
    results = model(resized_image)
    df = results.pandas().xyxy[0]
    for index, row in df.iterrows():
        class_id = int(row['class'])
        bbox_1 = [0.95*int(row['xmin']), 0.95*int(row['ymin']), 1.05*int(row['xmax']), 1.05*int(row['ymax'])]

        if class_id == 0 or class_id==1:  # Class ID 2 for extending the bounding box
            # Extend the bounding box by a factor of 4
            extended_bbox = extend_bbox(bbox_1, resized_image.size, factor=4)

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
                    type="New_Iraq"
                    end = time.time()
                    print("Time taken ", end - start_time)
                    return bbox_1, type,resized_image
                else:
                    cropped_image = extended_cropped_image.crop(bbox)
                    save_path = os.path.join(iran_dir, os.path.basename(input_path))
                    cropped_image.save(save_path)
                    type="Iran"
                    end = time.time()
                    print("Time taken ", end - start_time)
                    return bbox_1, type,resized_image

        else:
            # Save the bounding box image if it's not class 2
            cropped_image = resized_image.crop(bbox_1)
            save_path = os.path.join(old_iraq_dir, os.path.basename(input_path))
            cropped_image.save(save_path)
            type="Old_Iraq"
            end=time.time()
            print("Time taken ",end-start_time)
            return bbox_1,type,resized_image

    return [0,0,1,1],"None",resized_image
def inference_directory(input_dir):
    paths=os.listdir(input_dir)
    for path in paths:
        start=time.time()
        inference_image(os.path.join(input_dir,path))
        end=time.time()


def inference_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_counter = 0
    while True:

        ret, frame = cap.read()
        if not ret:
            break
        if frame_counter<550:
            frame_counter+=1
            continue
        if frame_counter % 2 != 0:
            print(frame_counter)
            frame_counter += 1
            continue

        # Convert the frame to PIL image for processing
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame.save('./frame.png')

        bbox,class_1,resize_image=inference_image("./frame.png")

        resize_image = np.array(resize_image)

        cv2.rectangle(resize_image, (int(bbox[0]/0.95), int(bbox[1]/0.95)),(int(bbox[2]/1.05), int(bbox[3]/1.05)), (0, 255, 0), 2)
        # Convert back to OpenCV format
        # Display the processed frame

        label = f"Class {class_1}"
        cv2.putText(resize_image, label, (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Press 'q' to quit the video display
        resize_image_1=resize_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        if class_1=="Iran":
            im = Image.fromarray(resize_image_1)
            im.save("./frame_temp.jpg")
            txt=inference_image_farsi("./frame_temp.jpg")
            cv2.putText(resize_image, txt, (int(bbox[0]), int(bbox[1] - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)
        cv2.imshow('Processed Frame', resize_image)
        frame_counter+=1
        print(frame_counter)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Example usage
input_video_path = r'C:\Users\HHR6\PycharmProjects\Testing_Pipeline\videoplayback.mp4'
inference_video(input_video_path)

# Test the function
#inference_image(r"C:/Users\HHR6\PycharmProjects\Testing_Pipeline\Acksession-Project\yolov8_compiled_dataset/test\Benchmark_Dataset/1_jpg.rf.e0d71bfac749526f7f57a7274f7f8f1e.jpg")"""
#inference_directory(input_dir)