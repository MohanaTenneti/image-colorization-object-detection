# AI-Based Image Colorization and Object Detection

A deep learning-based computer vision project that colorizes images using DDColor and detects objects using YOLOv8x.

## Overview

Image colorization is the process of adding realistic colors to grayscale or monochrome images. Object detection identifies and locates objects within an image.

This project combines both tasks into a single application. The input image is first processed using the pretrained DDColor model to generate a colorized image. The colorized output is then passed to YOLOv8x for object detection.

The complete system is deployed using Streamlit, allowing users to upload an image, view the colorized result, and detect objects with bounding boxes and confidence scores.

## Technologies Used

- Python
- PyTorch
- NumPy
- OpenCV
- Pillow
- DDColor
- YOLOv8x
- Ultralytics
- Hugging Face Hub
- Streamlit

## Project Workflow

1. Uploaded an image through the Streamlit interface.
2. Validated and loaded the input image.
3. Converted the image into the required format for processing.
4. Loaded the pretrained DDColor model.
5. Generated a colorized version of the input image using DDColor.
6. Displayed the original and colorized images.
7. Loaded the pretrained YOLOv8x object detection model when detection was requested.
8. Passed the colorized image to YOLOv8x.
9. Detected objects and generated bounding boxes.
10. Displayed detected object classes, confidence scores, and total object count.

## Models

### DDColor

DDColor is used for image colorization. A pretrained DDColor model is loaded through Hugging Face and used to generate the colorized output.

### YOLOv8x

YOLOv8x is used for object detection. It identifies objects present in the colorized image and provides:

- Object class
- Confidence score
- Bounding box

## Application

The project is implemented as a Streamlit application.

The application allows users to:

- Upload an image
- View the original image
- Generate and view the colorized image
- Run object detection
- View bounding boxes around detected objects
- View object names and confidence scores
- View the total number of detected objects


## How to Run

Clone the repository:

```bash
git clone https://github.com/MohanaTenneti/image-colorization-object-detection.git
cd image-colorization-object-detection

pip install -r requirements.txt

streamlit run app.py
