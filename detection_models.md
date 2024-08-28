# License Plate Detection for Iraqi and Iranian Vehicles

This section focuses on detecting and recognizing license plates from Iraqi and Iranian vehicles using three distinct AI models: DETR, EfficientNet, and YOLO. Each model was trained specifically for this task, leveraging their unique strengths to achieve optimal performance.

## Table of Contents

- [Overview](#overview)
- [Models](#models)
  - [DETR](#detr)
  - [EfficientNet](#efficientnet)
  - [YOLO](#yolo)
    - [Performance](#performance)

## Overview

This project is designed to detect and recognize license plates from images of Iraqi and Iranian vehicles. We trained three different models to understand and identify license plates in varying conditions, leveraging the strengths of different deep learning architectures.

## Models

### DETR

**DETR (DEtection TRansformers)** is a transformer-based model that handles object detection as a direct set prediction problem. It's known for its ability to manage complex relationships in image features and for its end-to-end approach, eliminating the need for anchor boxes. This model is particularly effective in scenarios where the objects of interest (in this case, license plates) may be partially obscured or located in diverse positions within the image.

- **Strengths**: Handles complex scenes well, robust to occlusion, end-to-end object detection.
- **Weaknesses**: May require more computational resources, slower inference times compared to other models.

### EfficientNet

**EfficientNet** is a family of convolutional neural networks that scales efficiently in terms of depth, width, and resolution. It achieves state-of-the-art accuracy with fewer parameters and computations. In the context of license plate detection, EfficientNet is utilized for its balance between accuracy and efficiency, making it suitable for deployment in resource-constrained environments.

- **Strengths**: High accuracy with fewer parameters, efficient for edge devices.
- **Weaknesses**: May not perform as well on highly complex or cluttered images.

### YOLO

**YOLO (You Only Look Once)** is a real-time object detection system that divides images into a grid and predicts bounding boxes and probabilities for each grid cell. YOLO is renowned for its speed, making it ideal for applications requiring real-time processing. For license plate detection, YOLO's ability to quickly and accurately detect objects makes it a strong candidate.

- **Strengths**: Fast inference time, suitable for real-time detection.
- **Weaknesses**: May struggle with small objects or objects in dense environments.

#### **Performance**

To improve classification performance, two YOLO models are used for license plate detection. The first classifies between 3 types of plates: New (English) Iraqi plates, Old (Arabic) Iraqi plates and Iranian plates. The second model is used only if the plate is detected as New Iraqi or Iranian; it performs a second inference to confirm or correct the classification.

**First model:**
* Precision: 96.6%
* Recall: 92.9%
* Mean average precision @0.5 IoU: 96.1%
* Mean average precision @0.5-0.95 IoU: 64.8%

**Second model:**
* Precision: 98.2%
* Recall: 98.3%
* Mean average precision @0.5 IoU: 99.1%
* Mean average precision @0.5-0.95 IoU: 71.5%
