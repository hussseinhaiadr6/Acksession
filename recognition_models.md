# License Plate Recognition for Iraqi and Iranian Vehicles

This section of the project focuses on recognizing license plate characters for Iraqi and Iranian vehicles. We developed two YOLO-based OCR models for Arabic and Farsi text recognition and leveraged the PaddleOCR English model for detecting and recognizing English numbers.

## Table of Contents

- [Overview](#overview)
- [Models](#models)
  - [YOLO Arabic OCR](#yolo-arabic-ocr)
    - [Dataset](#dataset)
  - [YOLO Farsi OCR](#yolo-farsi-ocr)
    - [Dataset](#dataset-1)
  - [PaddleOCR for English Numbers](#paddleocr-for-english-numbers)
- [Model Performance](#model-performance)
  - [YOLO Arabic OCR Performance](#yolo-arabic-ocr-performance)
  - [YOLO Farsi OCR Performance](#yolo-farsi-ocr-performance)
  - [PaddleOCR Performance](#paddleocr-performance)

## Overview

This section is dedicated to recognizing the characters on license plates from Iraqi and Iranian vehicles. Given the multilingual nature of these plates, we developed specialized OCR models for Arabic and Farsi scripts, alongside utilizing an existing PaddleOCR model for recognizing English numbers.

## Models

### YOLO Arabic OCR

The **YOLO Arabic OCR model** is designed to recognize Arabic characters on Iraqi license plates. Given the distinctiveness and complexity of Arabic script, this model was trained to accurately detect and interpret Arabic letters and numerals, even in challenging conditions such as varying font sizes and lighting.

- **Strengths**: Accurate recognition of Arabic script, robust to variations in text appearance.
- **Weaknesses**: May require fine-tuning for very specific regional variations or uncommon fonts.

#### Dataset

Dataset under open-source licensing from Roboflow. Used the datasets from the following projects:
* **workspace**: yakand, **project**: lpr-cckha, **version**: 1
* **workspace**: msc-9pkbx, **projects**: new-zfexd, **version**: 1

Dataset size:
* **train**: 7,933 images
* **test**: 370 images
* **valid**: 785 images

Class distribution in train dataset:
* 0: 4,123 instances
* 1: 3,535 instances
* 2: 4,130 instances
* 3: 3,382 instances
* 4: 3,405 instances
* 5: 3,793 instances
* 6: 3,579 instances
* 7: 3,787 instances
* 8: 3,247 instances
* 9: 3,076 instances
* a: 705 instances
* b: 103 instances
* d: 145 instances
* i: 169 instances
* f: 143 instances
* h: 164 instances
* j: 214 instances
* k: 90 instances
* l: 78 instances
* m: 253 instances
* n: 169 instances
* qaf: 84 instances
* r: 302 instances
* s: 240 instances
* t: 164 instances
* o: 197 instances
* z: 34 instances


### YOLO Farsi OCR

The **YOLO Farsi OCR model** focuses on recognizing Farsi characters, which are commonly found on Iranian license plates. Farsi, like Arabic, uses a complex script that can vary significantly in appearance. This model was trained to handle these variations effectively, ensuring accurate recognition across different plate designs.

- **Strengths**: High accuracy for Farsi text, handles variations in script style.
- **Weaknesses**: Performance may vary with heavily stylized or worn-out text.

#### Dataset

Dataset under open-source licensing from Roboflow. Used the datasets from the following projects:
* **workspace**: mohammadrezas-space, **project**: plate-detection-v2-zzg0d, **version**: 3
* **workspace**: my-space-kt30n, **projects**: ms-plate, **version**: 3
* **workspace**: m-ms, **projects**: plate-9vtsz, **version**: 1

Dataset size:
* **train**: 4,439 images
* **test**: 463 images
* **valid**: 1,279 images

Class distribution in train dataset:
* 0: 460 instances
* 1: 4,352 instances
* 2: 3,873 instances
* 3: 3,490 instances
* 4: 3,806 instances
* 5: 3,031 instances
* 6: 2,956 instances
* 7: 3,042 instances
* 8: 2,879 instances
* 9: 3,058 instances
* be: 343 instances
* dal: 430 instances
* ein: 191 instances
* he: 294 instances
* jim: 231 instances
* lam: 252 instances
* mim: 222 instances
* nun: 581 instances
* qaf: 348 instances
* sad: 353 instances
* sin: 237 instances
* ta: 180 instances
* te: 190 instances
* vav: 237 instances
* ye: 275 instances
* zhe: 63 instances

### PaddleOCR for English Numbers

For recognizing English numbers on license plates, we utilized the **PaddleOCR English model**. This pre-trained model is known for its efficiency and accuracy in detecting and interpreting English digits. It complements the Arabic and Farsi OCR models by handling the numeric portion of the license plates.

- **Strengths**: High accuracy for numeric characters, integrates well with multilingual OCR.
- **Weaknesses**: Limited to recognizing only numeric characters, relies on the other models for non-numeric text.

## Model Performance

### YOLO Arabic OCR Performance

* Precision: 90.2%
* Recall: 99.6%
* Mean average precision @0.5 IoU: 99%
* Mean average precision @0.5-0.95 IoU: 72.5% 

### YOLO Farsi OCR Performance

* Precision: 98.7%
* Recall: 99%
* Mean average precision @0.5 IoU: 99%
* Mean average precision @0.5-0.95 IoU: 68.4%

### PaddleOCR Performance

*Performance metrics such as accuracy, precision, recall, and F1-score will be detailed here.*
