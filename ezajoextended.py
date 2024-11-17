import cv2
import numpy as np
from easyocr import Reader
from ultralytics import YOLO
import re

# Load YOLO model
model = YOLO('best.pt')  # Replace 'best.pt' with the path to your trained model

# Initialize EasyOCR reader for the desired language
reader = Reader(['hu'])

# Load the image
#image_path = 'kep_mienk (1).jpg'  # Replace with your image path
image_path = 'szürke bounding boxxal/uj_02 (1).jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Use YOLO model to detect license plates
results = model.predict(source=image, save=False, conf=0.25)


# Helper function to format license plate text
def format_license_plate(text):
    # Remove any non-alphanumeric characters except for dash
    text = re.sub(r'[^A-Za-z0-9-]', '', text)

    # Try to match known patterns and reformat if needed
    if len(text) == 7 and re.match(r'^[A-Z]{4}\d{3}$', text):
        # If format is like "AAAA123" (new style), insert dash
        text = text[:3] + "-" + text[3:]
    elif len(text) == 8 and re.match(r'^[A-Z]{4}-\d{3}$', text):
        # If format is already correct "AAA-123" (new style)
        text = text[:3] + "-" + text[4:]
    elif len(text) == 6 and re.match(r'^[A-Z]{3}\d{3}$', text):
        # Old style license plate, e.g., "ABC123"
        text = text[:3] + "-" + text[3:]
    return text


# Process each detected bounding box
for result in results:
    boxes = result.boxes.xyxy  # Get bounding boxes
    for box in boxes:
        # Extract bounding box coordinates and convert to integers
        x1, y1, x2, y2 = map(int, box[:4])

        # Crop the detected license plate region
        license_plate_region = image[y1:y2, x1:x2]

        # Convert to RGB for EasyOCR compatibility
        license_plate_rgb = cv2.cvtColor(license_plate_region, cv2.COLOR_BGR2RGB)

        # Use EasyOCR to read text from the license plate region
        ocr_results = reader.readtext(license_plate_rgb)

        # Extract and clean the recognized text
        for detection in ocr_results:
            raw_text = detection[1]  # The detected text is in the second position
            cleaned_text = format_license_plate(raw_text)
            print("Cleaned License Plate Text:", cleaned_text)

            # Determine if it's an old or new license plate format
            plate_type = ""
            if re.match(r'^[A-Z]{3}-\d{3}$', cleaned_text):
                plate_type = "Old Hungarian License Plate"
            elif re.match(r'^[A-Z]{3}-\d{2}[A-Z]$', cleaned_text):
                plate_type = "New Hungarian License Plate"

            # Draw bounding box, detected text, and plate type on the original image if matched
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, cleaned_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            if plate_type:
                cv2.putText(image, plate_type, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Save the final image with annotations
output_path = 'kiolvasott_kepek/thick6.jpg'
cv2.imwrite(output_path, image)
#print(f"Annotated image saved as {output_path}")

# Optionally display the final image with bounding boxes and text
cv2.imshow("License Plate Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
