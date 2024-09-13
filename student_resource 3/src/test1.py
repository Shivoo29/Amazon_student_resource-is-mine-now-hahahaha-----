import cv2
import easyocr
import os
import csv
import pandas as pd
from tqdm import tqdm

def process_images(image_directory, output_csv):
    # Initialize EasyOCR reader with English language
    reader = easyocr.Reader(['en'], gpu=True)

    # Prepare CSV file
    fieldnames = ['image_name', 'detected_text', 'confidence']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through all images in the directory
        for image_name in tqdm(os.listdir(image_directory)):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_directory, image_name)
                
                # Perform OCR on the image
                results = reader.readtext(image_path)
                
                # Write results to CSV
                for (bbox, text, prob) in results:
                    writer.writerow({
                        'image_name': image_name,
                        'detected_text': text,
                        'confidence': prob
                    })

    print(f"OCR results saved to {output_csv}")

    # Read the CSV file into a pandas DataFrame for easy viewing
    df = pd.read_csv(output_csv)
    print(df.head())

# Define the path to your image directory
image_directory = '/home/codemaster29/Documents/Coding_Stuff/mlChallenge/AmazonML/66e31d6ee96cd_student_resource_3/student_resource 3/images/sampletest'

# Define the output CSV file path
output_csv = 'ocr_results.csv'

# Process the images
process_images(image_directory, output_csv)