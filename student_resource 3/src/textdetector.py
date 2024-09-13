import cv2
import easyocr
import matplotlib.pyplot as plt
import os

# Define the path to your image
image_path = r'/home/codemaster29/Documents/Coding_Stuff/mlChallenge/AmazonML/66e31d6ee96cd_student_resource_3/student_resource 3/images/test/11gHj8dhhrL.jpg'

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"File not found: {image_path}")
else:
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to read image from: {image_path}")
    else:
        # Initialize EasyOCR reader with English language
        reader = easyocr.Reader(['en'], gpu=True)

        # Perform OCR on the image
        text = reader.readtext(image_path)  # Pass the image path instead of the numpy array

        # Print the detected text
        print(text)

        # Optionally, display the image with detected text using matplotlib
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
