from PIL import Image
import pytesseract
import pandas as pd
import os
import cv2
import numpy as np
from difflib import get_close_matches

# Load the image
image_path = r"C:\Users\lenovo\OneDrive\Desktop\PS2-Samples-HackRX5\Sample7.png"

# Open the image using OpenCV
image = cv2.imread(image_path)

# Resize the image to enhance OCR accuracy
scaled_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Convert the image to grayscale
gray_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply adaptive thresholding to improve text clarity
thresh_image = cv2.adaptiveThreshold(blurred_image, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

# Experiment with morphological transformations (dilation, erosion)
kernel = np.ones((2, 2), np.uint8)
dilated_image = cv2.dilate(thresh_image, kernel, iterations=1)

# Save the processed image for manual review (optional)
processed_image_path = r"C:\Users\lenovo\OneDrive\Desktop\PS2-Samples-HackRX5\processed_sample5.png"
cv2.imwrite(processed_image_path, dilated_image)

# Function to perform OCR with retries
def perform_ocr(image, config):
    return pytesseract.image_to_string(image, config=config)

# OCR multiple attempts with different configs
ocr_attempts = [
    r'--oem 3 --psm 6',  # Standard config for uniform block of text
    r'--oem 3 --psm 4',  # Treat as a single column of text
    r'--oem 3 --psm 11', # Sparse text
]

# Try OCR with different configurations and accumulate text results
extracted_text = ""
for attempt in ocr_attempts:
    extracted_text = perform_ocr(dilated_image, attempt)
    if extracted_text.strip():
        break  # Stop if some text is found

# Split the OCR output into lines for easier processing
lines = extracted_text.split('\n')

# Function to search for the "Provisional diagnosis" text using both exact and fuzzy matching
def find_diagnosis_line(lines):
    keywords = ["provisional diagnosis", "diagnosis", "diag", "provisional"]
    
    # Try exact match first
    for line in lines:
        if any(keyword in line.lower() for keyword in keywords):
            return line  # Return the matched line if found
    
    # Fallback: Fuzzy matching
    for line in lines:
        closest_matches = get_close_matches(line.lower(), keywords, cutoff=0.5)
        if closest_matches:
            return line  # Return the matched line if found
    
    return ""

# Extract the line with "Provisional diagnosis"
diagnosis_line = find_diagnosis_line(lines)

# Extract the actual diagnosis from the line
provisional_diagnosis = ""
if diagnosis_line:
    parts = diagnosis_line.split(':')
    if len(parts) > 1:
        provisional_diagnosis = parts[1].strip()

# If no clear diagnosis is found, keep the text for manual review, but don't assume it's unreadable
if not provisional_diagnosis:
    print("Diagnosis not clearly identified. Text for review:")
    print(extracted_text)
    provisional_diagnosis = extracted_text  # Use full OCR output as fallback for review

# Create a DataFrame with the file name and the extracted diagnosis
data = {
    'file_name': [os.path.basename(image_path)],
    'provisional diagnosis': [provisional_diagnosis]
}

df = pd.DataFrame(data)

# Define the output path
output_path = r"C:\Users\lenovo\OneDrive\Desktop\PS2-Samples-HackRX5\answer\diagnosis_output.xlsx"

# Ensure the directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the DataFrame to an Excel file
df.to_excel(output_path, index=False)

print(f"Diagnosis saved to {output_path}")
