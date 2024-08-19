import tensorflow as tf
import cv2
import numpy as np
import os
import time  # Import time module
import tensorflow as tf
print(tf.__version__)
# Load the TensorFlow SavedModel
model_path = r"C:\Users\HHR6\PycharmProjects\AcksessionIntegration\LP_Detection\EfficientNEt\Efficientdet\content\saved_model"  # Path to the directory containing saved_model.pb
model = tf.saved_model.load(model_path)
infer = model.signatures['serving_default']

# Tensor names based on the provided information
input_tensor_name = 'images'
output_boxes_tensor_name = 'output_0'
output_classes_tensor_name = 'output_3'
output_scores_tensor_name = 'output_2'
output_num_detections_tensor_name = 'output_1'


# Load labels
def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


labels = load_labels("./labels.txt")  # Path to your labels file



def inference(image_path):
    input_image = cv2.imread(image_path)  # Path to your image
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Convert to tensor and normalize
    input_data = np.expand_dims(input_image, axis=0).astype(np.uint8)

    # Run inference
    output = infer(tf.constant(input_data))
    # Get output tensor names and process the output
    output_boxes = output["output_0"].numpy()
    output_classes = output["output_2"].numpy()
    output_scores = output["output_1"].numpy()

    # Interpret the model output
    predicted_boxes = output_boxes[0]  # Assuming the first batch item
    predicted_scores = output_scores[0]  # Assuming the first detection

    # Check if output_classes is an array or a scalar
    predicted_class_indices = output_classes[0]

    # Get the index of the highest score
    list_outputs = []
    for i in range(len(predicted_scores)):
        if predicted_scores[i] > 0.25:
            list_outputs.append(i)
    highest_predicted_scores = [predicted_scores[i] for i in list_outputs]
    highest_score_boxes = [predicted_boxes[i] for i in list_outputs]
    highest_score_class_indexes = [predicted_class_indices[i] for i in list_outputs]  # Ensure index is an integer
    print(highest_score_class_indexes)
    print(labels)
    # Check if the index is within the range of labels
    highest_score_labels = [labels[int(i) - 1] for i in highest_score_class_indexes]
    return highest_score_boxes, highest_predicted_scores, highest_score_labels


# Directory containing images
image_dir = "C:/Users/HHR6/PycharmProjects/Testing_Pipeline/Acksession-Project/yolov8_compiled_dataset/test/images"  # Path to the directory containing images

# Create the results directory if it doesn't exist
results_dir = "./results_efficient"
os.makedirs(results_dir, exist_ok=True)

# Loop through all images in the directory
for image_filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_filename)
    print(image_path)

    # Skip if it's not an image
    if not (image_path.endswith(".jpg") or image_path.endswith(".jpeg") or image_path.endswith(".png")):
        continue
    start_time = time.time()





    res_boxes, res_scores, res_labels = inference(image_path)
    end_time = time.time()
    inference_time = end_time - start_time
    print("Inference tile =", inference_time)
    input_image = cv2.imread(image_path)

    for i, (highest_score_box, label) in enumerate(zip(res_boxes, res_labels)):
        ymin, xmin, ymax, xmax = highest_score_box
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)

        # Draw the bounding box on the image
        cv2.rectangle(input_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        text = label
        cv2.putText(input_image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Create a directory for the label if it doesn't exist
        label_dir = os.path.join(results_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        # Save the cropped image in the label directory
        cropped_image = input_image[ymin:ymax, xmin:xmax]
        cropped_image_path = os.path.join(label_dir, f"{os.path.splitext(image_filename)[0]}_{label}_{i}.jpg")
        cv2.imwrite(cropped_image_path, cropped_image)

    # Save the annotated image
    # Display the annotated image using matplotlib
