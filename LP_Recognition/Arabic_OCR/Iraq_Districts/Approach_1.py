from paddleocr import PaddleOCR
from rapidfuzz import process, distance

import os
import cv2

def match_districts(image_path, ocr, old_path):
    result = ocr.ocr(image_path, det=False)

    districts = ["الأنبار", "بابل", "بغداد", "البصرة", "ذي قار", "ديالى", "دهوك", "اربيل", "كربلاء", "كركوك", "ميسان", "المثنى", "النجف", "نينوى", "خصوصي", "القادسية", "صلاح الدين", "سليمانية", "واسط", "اجرة", "حمل", "العراق"]

    for word in result[0][0][0].split(" "):
        best_match, score, _ = process.extractOne(word[::-1], districts, scorer=distance.Indel.normalized_distance)
        if score < 0.8 and best_match != "العراق":
            print("OCR output: ", word)
            print("Fuzzy matching Indel : ", best_match[::], score)
        best_match, score, _ = process.extractOne(word[::-1], districts, scorer=distance.JaroWinkler.normalized_distance)
        if score < 0.8 and best_match != "العراق":

            print("Fuzzy matching JaroWinkler: ", best_match[::], score)
        best_match, score, _ = process.extractOne(word[::-1], districts, scorer=distance.DamerauLevenshtein.normalized_distance)
        if score < 0.8 and best_match != "العراق":
            print("Fuzzy matching DamerauLevenshtein:", best_match[::], score)

            cv2.imshow("image", cv2.imread(image_path))
            cv2.waitKey()
            cv2.destroyAllWindows()

def crop_and_match_image(image_path, ocr):
    image = cv2.imread(image_path)
    dh, dw, _ = image.shape

    result = ocr.ocr(image_path, rec=False)
    cv2.imshow("image", cv2.imread(image_path))
    cv2.waitKey()
    cv2.destroyAllWindows()
    i = 0
    if result[0] is None:
        return
    for box in result[0]:
        x1, y1 = box[0]
        x2, y2 = box[2]

        file_name = str(i) + "_tmp.jpg"
        cv2.imwrite(file_name, image[max(0, int(y1 - 0.05*dh)) : min(dh, int(y2 + 0.05*dh)), max(0, int(x1 - 0.05*dw)): min(dw, int(x2+0.05*dw))])


        i += 1

        print(image_path)
        match_districts(file_name, ocr, image_path)

def main():
    ocr = PaddleOCR(lang="ar", det_db_box_thresh=0.2, det_db_thresh=0.1)

    #crop_and_match_image("./not_cropped.jpg", ocr)
    root_dir = r"C:\Users\HHR6\PycharmProjects\AcksessionIntegration\LP_Detection\Yolo\output_Iraq\OLD_Iraq/"
    for image in os.listdir(root_dir):
        crop_and_match_image(root_dir + image, ocr)

if __name__ == "__main__":
    main()