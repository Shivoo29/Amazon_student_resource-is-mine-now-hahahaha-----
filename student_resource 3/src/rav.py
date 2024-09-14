import cv2
import easyocr
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import os

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Unit mappings provided
unit_mappings = {
    'mg': 'milligram', 'MG': 'milligram', 'Mg': 'milligram', 'mG': 'milligram',
    'g': 'gram', 'G': 'gram',
    'kg': 'kilogram', 'KG': 'kilogram', 'Kg': 'kilogram', 'kG': 'kilogram',
    'cm': 'centimetre', 'CM': 'centimetre', 'Cm': 'centimetre', 'cM': 'centimetre',
    'mm': 'millimetre', 'MM': 'millimetre', 'Mm': 'millimetre', 'mM': 'millimetre',
    'inch': 'inch', 'Inch': 'inch', 'INCH': 'inch',
    'yard': 'yard', 'Yard': 'yard', 'YARD': 'yard',
    'v': 'volt', 'V': 'volt', 'Volt': 'volt', 'VOLT': 'volt',
    'w': 'watt', 'W': 'watt', 'Watt': 'watt', 'WATT': 'watt',
    'kv': 'kilovolt', 'kV': 'kilovolt', 'Kv': 'kilovolt', 'KV': 'kilovolt', 'KiloVolt': 'kilovolt',
    'mv': 'millivolt', 'mV': 'millivolt', 'Mv': 'millivolt', 'MV': 'millivolt', 'MilliVolt': 'millivolt',
    'kw': 'kilowatt', 'kW': 'kilowatt', 'Kw': 'kilowatt', 'KW': 'kilowatt', 'KiloWatt': 'kilowatt',
    'ml': 'millilitre', 'ML': 'millilitre', 'Ml': 'millilitre', 'mL': 'millilitre',
    'l': 'litre', 'L': 'litre', 'Litre': 'litre', 'LITRE': 'litre',
    'ft': 'foot', 'FT': 'foot', 'Ft': 'foot', 'fT': 'foot',
    'yd': 'yard', 'YD': 'yard', 'Yd': 'yard', 'yD': 'yard',
    'lb': 'pound', 'lbs': 'pound', 'LB': 'pound', 'Lbs': 'pound', 'LBS': 'pound', 'IBS': 'pound', 'Ibs': 'pound',
    'm': 'meter', 'M': 'meter', 'Meter': 'meter', 'METER': 'meter',
    'oz': 'ounce', 'OZ': 'ounce', 'Oz': 'ounce', 'oZ': 'ounce',
    'cu': 'cubic', 'CU': 'cubic', 'Cu': 'cubic', 'cU': 'cubic',
    'cubic foot': 'cubic foot', 'Cubic Foot': 'cubic foot', 'CUBIC FOOT': 'cubic foot',
    'cubic inch': 'cubic inch', 'Cubic Inch': 'cubic inch', 'CUBIC INCH': 'cubic inch',
    'ton': 'ton', 'Ton': 'ton', 'TON': 'ton',
    'tonne': 'ton', 'Tonne': 'ton', 'TONNE': 'ton'
}

# Entity Unit Map
entity_unit_map = {
    "width": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "depth": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "height": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
    "item_weight": {"milligram", "kilogram", "gram", "ounce", "ton", "pound"},
    "maximum_weight_recommendation": {"milligram", "kilogram", "gram", "ounce", "ton", "pound"},
    "voltage": {"volt", "kilovolt", "millivolt"},
    "wattage": {"watt", "kilowatt"},
    "item_volume": {"millilitre", "litre", "gallon", "cubic foot", "cubic inch"}
}

# Improved regex pattern to capture more numerical data and units
unit_pattern = r'(\d+\.?\d*)\s?(lb|lbs|IBS|Ibs|pound|meter|m|mg|g|kg|cm|mm|inch|ft|yard|v|volt|w|watt|kv|kV|mv|mV|kw|kW|ml|l|litre|cu|cubic foot|cubic inch|ton|tonne|oz)'


# Function to perform OCR with retries
def perform_ocr_with_retries(image_path, max_retries=2):
    for attempt in range(max_retries):
        result = reader.readtext(image_path)
        if result:
            return result
    return []

# Function to map extracted text to appropriate entity and units
def map_to_entity_and_unit(extracted_text):
    matches = re.findall(unit_pattern, extracted_text, re.IGNORECASE)
    entity_data = defaultdict(set)  # Use set to avoid duplicates

    # Map extracted values to their appropriate units with full names
    for match in matches:
        value, unit = match
        full_unit = unit_mappings.get(unit.lower(), unit)  # Map unit to full name
        for entity, units in entity_unit_map.items():
            if full_unit.lower() in units:
                entity_data[entity].add(f"{value} {full_unit}")  # Add to set to avoid duplicates
                break

    # Convert sets back to lists
    entity_data = {k: list(v) for k, v in entity_data.items()}
    return entity_data

# Function to display the image with OCR result
def display_image_with_ocr(image_path, ocr_text):
    img = cv2.imread(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    print(f"OCR Result: {ocr_text}")

# Function to save data to a CSV file
def save_to_csv(csv_file, data, column_names):
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        if not file_exists:
            writer.writeheader()  # Write header only if file does not exist
        
        writer.writerow(data)

# Function to split dimensions into width, depth, and height
def split_dimensions(dimensions_value):
    dimensions_split = dimensions_value.split()  # Assuming the dimensions are space-separated
    if len(dimensions_split) >= 3:
        return dimensions_split[0], dimensions_split[1], dimensions_split[2]
    return 0, 0, 0

# Process a folder of images and extract relevant information
def process_images(image_folder, csv_file):
    column_names = ["image number", "width", "depth", "height", "item_weight", 
                    "maximum_weight_recommendation", "voltage", "wattage", "item_volume"]
    
    for image_num, image_file in enumerate(os.listdir(image_folder), start=1):
        image_path = os.path.join(image_folder, image_file)

        # Perform OCR with retries
        ocr_result = perform_ocr_with_retries(image_path)

        # Initialize data row with empty arrays (all columns are arrays)
        data_row = {col: [] for col in column_names}
        data_row["image number"] = [image_num]  # Keep image number as a single value list

        if ocr_result:
            extracted_text = ' '.join([res[1] for res in ocr_result])

            # Extract entity data using regex
            entity_data = map_to_entity_and_unit(extracted_text)

            # Display image and OCR results
            display_image_with_ocr(image_path, extracted_text)

            # Print extracted entity data with unit mappings
            if entity_data:
                print(f"Extracted Data for {image_file}: {dict(entity_data)}")

                # Update the data_row with extracted entity values
                for entity, values in entity_data.items():
                    if entity == "dimensions":  # Special case for dimensions
                        if values:
                            width, depth, height = split_dimensions(values[0])
                            data_row["width"] = [width]
                            data_row["depth"] = [depth]
                            data_row["height"] = [height]
                    else:
                        # Store all values as arrays, even if it's a single value
                        data_row[entity] = list(values) if values else []

            else:
                print(f"No relevant data found for {image_file}.")
        else:
            print(f"No text found in {image_file} after retries.")
        
        # If no values are extracted for a column, fill it with [0]
        for col in column_names:
            if not data_row[col]:
                data_row[col] = [0]

        # Save the row to the CSV file
        save_to_csv(csv_file, data_row, column_names)

# Example usage
image_folder = '66e31d6ee96cd_student_resource_3/student_resource 3/images/test'  # Replace with your folder path
csv_file = 'rav_results.csv'
process_images(image_folder, csv_file)