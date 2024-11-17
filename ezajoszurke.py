import cv2
import numpy as np
from easyocr import Reader
from ultralytics import YOLO

# Load YOLO model
model = YOLO('best.pt')  # Replace 'best.pt' with the path to your trained model

# Initialize EasyOCR reader for the desired language
reader = Reader(['hu'])

# Load the image
image_path = 'szürke bounding boxxal/new_plate (1).jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Use YOLO model to detect license plates
results = model.predict(source=image, save=False, conf=0.25)

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

        # Extract and display the recognized text
        for detection in ocr_results:
            text = detection[1]  # The detected text is in the second position

            # Check if the detected text is longer than 6 characters
            if len(text) > 6:
                print("Detected New License Plate Text:", text)

                # Draw the bounding box on the original image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Write "NEW LICENSE PLATE" label and the detected text on the image
                label = f"New Type of license plate: {text}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

output_path = 'kiolvasott_kepek/thick6.jpg'
cv2.imwrite(output_path, image)
# Display the final image with bounding boxes and text
cv2.imshow("License Plate Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
