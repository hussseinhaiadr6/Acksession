import os
import time
import torch
import torchvision.transforms as T

from PIL import Image, ImageDraw, ImageFont
# Define transforms
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Function to convert bounding box format
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


# Function to rescale bounding boxes
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Function to filter and process bounding boxes
def filter_bboxes_from_outputs(outputs, threshold=0.7):
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    probas_to_keep = probas[keep]
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas_to_keep, bboxes_scaled


# Function to draw rectangles and save the image
def draw_and_save_results(pil_img, prob=None, boxes=None, output_path=None):
    draw = ImageDraw.Draw(pil_img)
    colors = COLORS * 100
    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            draw.rectangle([xmin, ymin, xmax, ymax], outline=tuple(int(x * 255) for x in c), width=3)
            cl = p.argmax()
            text = f'{finetuned_classes[cl]}: {p[cl]:0.4f}'
            font = ImageFont.truetype("arial.ttf", 24)
            draw.text((xmin, ymin), text,font=font, fill="red")
    if output_path:
        pil_img.save(output_path)


# Function to run workflow on an image
def run_workflow(my_image, my_model, threshold=0.7):
    img = transform(my_image).unsqueeze(0)
    outputs = my_model(img)
    probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs, threshold=threshold)
    return probas_to_keep, bboxes_scaled


# Define model and load weights
model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=4)

checkpoint = torch.load('./DETR_weights/checkpoint_drive.pth',
                        map_location='cpu')

model.load_state_dict(checkpoint['model'],
                      strict=False)


model.eval()

# Define classes and colors
finetuned_classes = ['LP', 'Background', 'English_plate', 'Iraq_plate']
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556]]

# Directory containing Benchmark_Dataset
img_dir = 'C:/Users/HHR6/PycharmProjects/Testing_Pipeline/Acksession-Project/yolov8_compiled_dataset/test/images/'

# Directory to save Benchmark_Dataset with rectangles
output_dir = 'C:/Users/HHR6/PycharmProjects/Testing_Pipeline/results_detr'

# Ensure output directory exists, create it if necessary
os.makedirs(output_dir, exist_ok=True)

# Loop over each image in the directory
for img_name in os.listdir(img_dir):
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        img_path = os.path.join(img_dir, img_name)
        im = Image.open(img_path)

        # Measure inference time
        start_time = time.time()

        # Run workflow on the image
        probas_to_keep, bboxes_scaled = run_workflow(im, model, threshold=0.4)


        # Measure inference time
        inference_time = time.time() - start_time
        print(f"Inference time for {img_name}: {inference_time:.2f} seconds")

        # Draw rectangles and save the image
        output_path = os.path.join(output_dir, f'result_{img_name}')
        draw_and_save_results(im, probas_to_keep, bboxes_scaled, output_path)
