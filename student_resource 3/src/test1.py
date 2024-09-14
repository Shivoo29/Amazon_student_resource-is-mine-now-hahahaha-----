import cv2
import easyocr
import os
import csv
import pandas as pd
from tqdm import tqdm
import warnings
from PIL import Image
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def is_valid_image(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False

def process_images(image_directory, output_csv):
    # Initialize EasyOCR reader with English language
    reader = easyocr.Reader(['en'], gpu=True)

    # Prepare CSV file
    fieldnames = ['image_name', 'detected_text', 'confidence']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Check if the directory exists
        if not os.path.exists(image_directory):
            print(f"Error: Directory not found: {image_directory}")
            return

        # Iterate through all images in the directory
        for image_name in tqdm(os.listdir(image_directory)):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_directory, image_name)
                
                # Check if the image is valid
                if not is_valid_image(image_path):
                    print(f"Skipping corrupted or invalid image: {image_name}")
                    continue

                try:
                    # Perform OCR on the image
                    results = reader.readtext(image_path)
                    
                    # Write results to CSV
                    for (bbox, text, prob) in results:
                        writer.writerow({
                            'image_name': image_name,
                            'detected_text': text,
                            'confidence': prob
                        })
                except Exception as e:
                    print(f"Error processing image {image_name}: {str(e)}")

    print(f"OCR results saved to {output_csv}")

    # Read the CSV file into a pandas DataFrame for easy viewing
    df = pd.read_csv(output_csv)
    print(df.head())

# Define the path to your image directory
image_directory = '/home/codemaster29/Documents/Coding_Stuff/mlChallenge/AmazonML/66e31d6ee96cd_student_resource_3/student_resource 3/images/train'

# Define the output CSV file path
output_csv = 'ocr_results.csv'

# Process the images
process_images(image_directory, output_csv)