# District Identification from Iraqi License Plates

## Overview

This project aims to identify the district name on Iraqi license plates using two AI/ML methods. These methods are designed to enhance the accuracy of district identification by leveraging both object detection and optical character recognition (OCR) techniques. The two methods developed are:

1. **YOLO Model**: A classic YOLO (You Only Look Once) object detection model tailored for recognizing district names directly from license plates.
2. **PaddleOCR with Fuzzy Matching**: An OCR-based approach using PaddleOCR to read the text from the license plate, followed by fuzzy matching techniques to identify the closest possible district name.

---

## Method 1: YOLO Model

### Description
The YOLO model is a real-time object detection system that has been adapted to identify district names on Iraqi license plates. This method relies on detecting the location and content of the district name directly from the image of the license plate.

### Dataset Used
Dataset under open-source licensing from Roboflow. Used the datasets from the following projects:
* **workspace**: yakand, **project**: lpr-cckha, **version**: 1
* **workspace**: msc-9pkbx, **projects**: new-zfexd, **version**: 1

Dataset size:
* **train**: 6,892 images
* **test**: 324 images
* **valid**: 659 images

Class distribution in train dataset:
* al-anbar: 91 instances
* babel: 329 instances
* baghdad: 1,251 instances
* al-basra: 405 instances
* zi-qar: 308 instances
* diyali: 307 instances
* dahuk: 517 instances
* erbil: 1,343 instances
* karbala: 171 instances
* karkouk: 145 instances
* misan: 113 instances
* al-masna: 357 instances
* al-najaf: 243 instances
* ninawa: 144 instances
* private: 2,896 instances
* al-qadissiyah: 254 instances
* salah-eddine:116 instances
* suleimaniyah: 577 instances
* waset: 203 instances
* taxi: 1,033 instances
* truck: 489 instances

## Method 2: PaddleOCR with Fuzzy Matching

### Description
This method involves two main steps:

1. **Text Extraction**: PaddleOCR is employed to extract text from the license plate image. PaddleOCR is a lightweight, high-performance OCR system that is well-suited for recognizing text in various languages and scripts, including Arabic, which is used on Iraqi license plates.
  
2. **Fuzzy Matching**: Once the text is extracted, it is compared against a predefined list of district names using fuzzy matching techniques. Fuzzy matching helps in identifying the closest possible match even if the OCR output is not perfect or contains slight variations in spelling.

