import os
import sys

# adding DDColor folder so Python can find it
ddcolor_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "DDColor"
)

sys.path.insert(0, ddcolor_path)

import streamlit as st
from PIL import Image
import cv2
import numpy as np
import torch

from ddcolor import DDColor, ColorizationPipeline
from huggingface_hub import PyTorchModelHubMixin
from ultralytics import YOLO


# loading the DDColor model
class DDColorHF(DDColor, PyTorchModelHubMixin):

    def __init__(self, config=None, **kwargs):

        if isinstance(config, dict):
            kwargs = {**config, **kwargs}

        super().__init__(**kwargs)


@st.cache_resource
def load_ddcolor():

    # checking if GPU is available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # loading pretrained DDColor model
    model = DDColorHF.from_pretrained(
        "piddnad/ddcolor_modelscope"
    )

    model = model.to(device)
    model.eval()

    # creating the colorization pipeline
    colorizer = ColorizationPipeline(
        model,
        input_size=512,
        device=device
    )

    return colorizer


@st.cache_resource
def load_yolo():

    # loading YOLOv8x model
    model = YOLO("yolov8x.pt")

    return model


# app title
st.title("AI-Based Image Colorization and Object Recognition")

st.write(
    "Upload an image to colorize it and detect objects."
)


# uploading image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp"]
)


if uploaded_file is not None:

    # opening the uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True
    )

    # converting image into numpy array
    image_np = np.array(image)

    # converting RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    # loading DDColor
    colorizer = load_ddcolor()

    # colorizing the image
    with st.spinner("Colorizing image..."):

        colorized_bgr = colorizer.process(image_bgr)

    # converting BGR back to RGB
    colorized_rgb = cv2.cvtColor(
        colorized_bgr,
        cv2.COLOR_BGR2RGB
    )

    st.subheader("Colorized Image")

    st.image(
        colorized_rgb,
        use_container_width=True
    )

    # object detection section
    st.subheader("Object Detection")

    if st.button("Detect Objects"):

        # loading YOLO
        yolo_model = load_yolo()

        # running object detection
        with st.spinner("Detecting objects..."):

            results = yolo_model(colorized_bgr)

        result = results[0]

        # showing detected image
        result_image = result.plot()

        result_image = cv2.cvtColor(
            result_image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            result_image,
            caption="Detected Objects",
            use_container_width=True
        )

        # getting detected objects
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:

            st.subheader("Detected Objects")

            for box in boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                object_name = result.names[class_id]

                st.write(
                    f"{object_name}: {confidence:.2f}"
                )

            st.write(
                f"Total objects detected: {len(boxes)}"
            )

        else:

            st.write("No objects detected.")