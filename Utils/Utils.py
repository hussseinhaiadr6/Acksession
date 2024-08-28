import cv2
import numpy as np
import csv
import os
import skimage

def keep_first_two_files(directory):
    # Get a list of files in the directory
    files = os.listdir(directory)
    files = [file for file in files if file.endswith('.jpg') and file[:-4].isdigit()]

    # Sort files based on the numeric value of the filename (without the .jpg extension)
    files.sort(key=lambda x: int(x[:-4]))

    # Check if there are more than two files
    if len(files) > 2:
        # Iterate over the files starting from the third one
        for file in files[2:]:
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")



def visualize_canny_edges(image_path, low_threshold=50, high_threshold=100):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detector
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)

    return edges

def adjust_contour(image, x, y, w, h,type="English Plate"):

    stat_file = "./bboxes_data.csv"

    # Extract average LP H/W ratio
    with open(stat_file, "r") as f:
        reader = csv.reader(f)
        line = None
        for row in reader:
            if row[0] == type:
                line = row

    _, harmonic_mean, mean, stdev = line

    assert h != 0 and w != 0, "Invalid box dimensions"
    image_height, image_width, _ = image.shape
    print(image_height, image_width)


    # LP width should be at least 85% of bounding box width
    if type=="English Plate":
        w = round(max(w, 0.95 * image_width))
    else:

        w = round(max(w, 0.85 * image_width))

    print(" wdith adjusted is :", w)
    harmonic_mean = float(harmonic_mean)
    mean = float(mean)
    stdev = float(stdev)
    # print(mean, stdev)

    # Ensure consistency with average ratio
    hw_ratio = h / w
    # print(hw_ratio)
    if hw_ratio < mean - stdev or hw_ratio > mean + 2 * stdev:
        h = round((h * (mean / hw_ratio))*1.3)

    image_center_x, image_center_y = image_width // 2, image_height // 2
    # print(image_center_x, image_center_y)
    center_x, center_y = x + w // 2, y + h // 2
    print(center_x, center_y)
    # print(center_x, center_y)
    if center_x < image_center_x - 0.05 * image_width or center_x > image_center_x + 0.05 * image_width:
        # If LP is not centered, move it to center of image
        x = image_center_x - w // 2
    if center_y < image_center_y - 0.1 * image_height or center_y > image_center_y + 0.1 * image_height:
        y = image_center_y - h // 2
    print("new cnters ", x,y)
    #handling edge cases
    y=max(0,y)
    x=max(0,x)

    return x, y, w, h


def find_license_plate_boxes(edge_image, image_path,apply_adjustment=False,type="English Plate"):
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to hold the largest contours by area and perimeter
    largest_contour_area = None
    largest_contour_perimeter = None
    largest_area = 0
    largest_perimeter = 0

    # Loop through all detected contours
    for contour in contours:
        image = cv2.imread(image_path)
        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(contour, True)
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if 2 <= len(approx) <= 17:
            area = cv2.contourArea(contour)
            if perimeter > largest_perimeter:
                largest_contour_perimeter = contour
                largest_perimeter = perimeter
            # Update if the current contour has the largest area
            if area > largest_area:
                largest_contour_area = contour
                largest_area = area

    # Check if any contours were found
    if largest_contour_area is not None and largest_contour_perimeter is not None:
        # Get bounding box coordinates for the largest area
        x_area, y_area, w_area, h_area = cv2.boundingRect(largest_contour_area)

        # Get bounding box coordinates for the largest perimeter
        x_perimeter, y_perimeter, w_perimeter, h_perimeter = cv2.boundingRect(largest_contour_perimeter)
        # Adjust box to optimize fitting
        if apply_adjustment:

            if w_perimeter * h_perimeter < 0.5 * edge_image.shape[0] * edge_image.shape[1] and w_area * h_area < 0.5 * \
                    edge_image.shape[0] * edge_image.shape[1]:
                print(image_path, " the area of the bbox is small adjusting ")
                x_perimeter, y_perimeter, w_perimeter, h_perimeter = adjust_contour(image, x_perimeter, y_perimeter,
                                                                                    w_perimeter, h_perimeter,type)
                x_area, y_area, w_area, h_area = adjust_contour(image, x_area, y_area, w_area, h_area,type)

        # Determine which box has the largest area between the two
        box_area1 = w_area * h_area
        box_area2 = w_perimeter * h_perimeter
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        image_center_x, image_center_y = image_width // 2, image_height // 2

        if box_area1 > box_area2:
            if apply_adjustment:
                centerX, centerY = x_area + (w_area // 2), y_area + (h_area // 2)
                if centerX < image_center_x - 0.1 * image_width or centerX > image_center_x + 0.1 * image_width or centerY < image_center_y - 0.1 * image_height or centerY > image_center_y + 0.1 * image_height:
                    print(image_path, " the bbox is not centered  of the bbox is small adjusting ")
                    # If LP is not centered, move it to center of image
                    x_area, y_area, w_area, h_area = adjust_contour(image, x_area, y_area, w_area, h_area,type)
            print(image_path,"area is bigger")
            return (x_area, y_area, w_area, h_area), largest_contour_area

        else:
            if apply_adjustment:
                centerX, centerY = x_perimeter + (w_perimeter // 2), y_perimeter + (h_perimeter // 2)
                if centerX < image_center_x - 0.1 * image_width or centerX > image_center_x + 0.1 * image_width or centerY < image_center_y - 0.1 * image_height or centerY > image_center_y + 0.1 * image_height:
                    print(image_path, " the bbox is not centered  of the bbox is small adjusting ")
                    # If LP is not centered, move it to center of image
                    x_perimeter, y_perimeter, w_perimeter, h_perimeter = adjust_contour(image, x_perimeter, y_perimeter,w_perimeter, h_perimeter,type)
            print(image_path,"perimetere is bigger")
            return (x_perimeter, y_perimeter, w_perimeter, h_perimeter), largest_contour_perimeter
    else:
        raise ValueError("No suitable license plate contours detected.")


def visualize_license_plate_boxes(image_path, edge_image,save_dir,apply_adjustments=False,type="English Plate"):
    # Load the original image
    image = cv2.imread(image_path)

    # Find the bounding boxes of the license plate
    bounding_box, contour = find_license_plate_boxes(edge_image, image_path,apply_adjustments,type)
    x, y, w, h = bounding_box

    # Draw the bounding box on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cropped_image = image[y:y + h, x:x + w]

    # Save the cropped image
    cropped_image_path = os.path.join(save_dir,image_path.split('\\')[-1])

    cv2.imwrite(cropped_image_path, cropped_image)


def rotate_image_mine(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
def rotate_image(image, angle):
    return skimage.transform.rotate(image, angle)
def find_best_two_rotation_angles(image, angle_increment=1, max_angle=60):
    best_angle11 = 0
    best_angle22 = 0
    highest_max_value1 = -np.inf
    highest_max_value2 = -np.inf

    angles = range(-max_angle, max_angle, angle_increment)

    for angle in angles:
        rotated_image = rotate_image(image, angle)
        vertical_hist = np.sum(rotated_image, axis=1, keepdims=True) / 100
        max_value = np.max(vertical_hist)
        # get 2 best angle because the image could be upside down and have a max value
        if max_value > highest_max_value1:
            highest_max_value2 = highest_max_value1
            best_angle22 = best_angle11
            highest_max_value1 = max_value
            best_angle11 = angle
        elif max_value > highest_max_value2:
            highest_max_value2 = max_value
            best_angle22 = angle



    return (best_angle11, highest_max_value1), (best_angle22, highest_max_value2)

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding
    thresh11 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 5, 5)

    return thresh11

def fix_angle(dir, save_dir):

    paths=os.listdir(dir)
    for path in paths:

      if path.endswith(".jpg"):

        save_path=f"{save_dir}/{path}"
        image_path=f"{dir}/{path}"

        image_read=cv2.imread(image_path)
        thresh= preprocess_image(image_path)
        (best_angle1, highest_max_value1), (best_angle2, highest_max_value2) = find_best_two_rotation_angles(thresh)

        rotate1=rotate_image_mine(image_read, min(best_angle1, best_angle2))
        cv2.imwrite(save_path, rotate1)



def Optimise_LP(dir,save_dir, apply_adjustments=False, type="English Plate"):
    os.makedirs(save_dir, exist_ok=True)
    fix_angle(dir,save_dir)
    paths = os.listdir(save_dir)
    for path in paths:
      image_path= os.path.join(save_dir, path)
      edge=visualize_canny_edges(image_path)
      visualize_license_plate_boxes(image_path, edge,save_dir, apply_adjustments,type)

dir=r"C:\Users\HHR6\PycharmProjects\AcksessionIntegration\LP_Detection\Yolo\output_Iraq_dataset_test\New_Iraq/"

save_dir_1=r"../Image_processing/no_adjustment_New_Iraq"

Optimise_LP(dir,save_dir_1,apply_adjustments=False,type="English Plate")