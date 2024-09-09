import os
import cv2
import pytesseract
import numpy as np
import pandas as pd
from difflib import get_close_matches

# Define the folder path directly in the code
folder_path = r"C:\Users\lenovo\OneDrive\Desktop\PS2-Samples-HackRX5"  # Change this to your desired folder path

def process_folder(folder_path):
    '''
    This function processes all images in the provided folder
    and saves the extracted information into one spreadsheet.
    '''
    all_data = []  # List to accumulate all results
    
    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):  # Process only PNG files
            image_path = os.path.join(folder_path, file_name)

            # Process each image
            provisional_diagnosis = process_image(image_path)
            
            # Append results to the list
            all_data.append({
                'file_name': file_name,
                'provisional diagnosis': provisional_diagnosis
            })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_data)

    # Save the DataFrame to an Excel file
    output_path = os.path.join(folder_path, 'diagnosis_output.xlsx')
    df.to_excel(output_path, index=False)

    print(f"All diagnoses saved to {output_path}")

def process_image(image_path):
    '''
    This function processes a single image and extracts provisional diagnosis.
    '''
    
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

    # Perform OCR with multiple attempts
    ocr_attempts = [
        r'--oem 3 --psm 6',  # Standard config for uniform block of text
        r'--oem 3 --psm 4',  # Treat as a single column of text
        r'--oem 3 --psm 11', # Sparse text
    ]

    extracted_text = ""
    for attempt in ocr_attempts:
        extracted_text = pytesseract.image_to_string(dilated_image, config=attempt)
        if extracted_text.strip():
            break  # Stop if some text is found

    # Split the OCR output into lines for easier processing
    lines = extracted_text.split('\n')

    # Find the line containing the "Provisional diagnosis"
    diagnosis_line = find_diagnosis_line(lines)

    # Extract the actual diagnosis from the line
    provisional_diagnosis = ""
    if diagnosis_line:
        parts = diagnosis_line.split(':')
        if len(parts) > 1:
            provisional_diagnosis = parts[1].strip()

    # Fallback if no clear diagnosis is found
    if not provisional_diagnosis:
        provisional_diagnosis = extracted_text  # Use full OCR output as fallback

    return provisional_diagnosis

def find_diagnosis_line(lines):
    '''
    Searches for the "Provisional diagnosis" in the text lines using exact and fuzzy matching.
    '''
    keywords = ["provisional diagnosis", "diagnosis", "diag", "provisional"]
    
    for line in lines:
        if any(keyword in line.lower() for keyword in keywords):
            return line  # Return the matched line if found
    
    for line in lines:
        closest_matches = get_close_matches(line.lower(), keywords, cutoff=0.5)
        if closest_matches:
            return line  # Return the matched line if found
    
    return ""

# Call the function directly with the hardcoded folder path
process_folder(folder_path)
