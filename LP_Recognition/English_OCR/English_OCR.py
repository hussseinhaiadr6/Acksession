from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont
import os

# Initialize OCR
ocr = PaddleOCR(lang="ch", ocr_version="PP-OCRv4", use_angle_cls=True)

# Create output directory if it doesn't exist
output_dir = r".\output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define font (adjust the path to the font file as needed)
font_path = "arial.ttf"  # Change this to your font file path if needed
font_size = 25

for filename in os.listdir(
        r"C:\Users\HHR6\PycharmProjects\AcksessionIntegration\Image_processing\adjustment_New_Iraq/"):
    img_file = os.path.join(r"C:\Users\HHR6\PycharmProjects\AcksessionIntegration\Image_processing\adjustment_New_Iraq",
                            filename)

    # Perform OCR on the image
    result = ocr.ocr(img_file)
    print(f"Processing file: {img_file}")

    # Load the image
    image = Image.open(img_file)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    for res in result:
        if res:
            for line in res:
                if line:
                    txt = line[1][0]
                    # Draw text on the image
                    position = (int(line[0][0][0]), int(line[0][0][1])+10)  # Use the OCR detected position
                    draw.text(position, txt, fill="red", font=font)
                else:
                    print("No detection")

    # Save the modified image in the output directory
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)
    print(f"Saved output to {output_path}")
