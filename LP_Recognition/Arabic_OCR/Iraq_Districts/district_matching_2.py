from paddleocr import PaddleOCR
from rapidfuzz import process, distance

import os
import cv2

def match_districts(image_path, ocr, old_path):
    result = ocr.ocr(image_path, det=False)

    districts = ["الأنبار", "بابل", "بغداد", "البصرة", "ذي قار", "ديالى", "دهوك", "اربيل", "كربلاء", "كركوك", "ميسان", "المثنى", "النجف", "نينوى", "خصوصي", "القادسية", "صلاح الدين", "سليمانية", "واسط", "اجرة", "حمل", "العراق"]

    for word in result[0][0][0].split(" "):
        best_match, score, _ = process.extractOne(word[::-1], districts, scorer=distance.JaroWinkler.normalized_distance)
        if score < 0.8 and best_match != "العراق":
            print("OCR output: ", word)
            print("Fuzzy matching: ", best_match[::], score)
        best_match, score, _ = process.extractOne(word[::-1], districts, scorer=distance.DamerauLevenshtein.normalized_distance)
        if score < 0.8 and best_match != "العراق":
            print("Fuzzy matching 2: ", best_match[::], score)
            cv2.imshow("image", cv2.imread(image_path))
            cv2.waitKey()
            cv2.destroyAllWindows()

def crop_and_match_image(image_path, ocr):
    image = cv2.imread(image_path)
    match_districts(image_path, ocr, image_path)


def main():
    ocr = PaddleOCR(lang="ar")

    #crop_and_match_image("./not_cropped.jpg", ocr)
    root_dir = "C:/Users/HHR6/PycharmProjects/ALPR/yolov5/runs/detect/exp5/crops/"
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file)

                print(file_path)
                crop_and_match_image(file_path, ocr)

if __name__ == "__main__":
    main()