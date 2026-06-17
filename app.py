import os
import urllib.request
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Set up page layout to wide mode
st.set_page_config(layout="wide", page_title="Dual Face Filter App")
st.title("🎭 Dual Image Effects Processor")
st.write("Upload an image to see the split effects side by side.")

# -------------------------------------------------------------------------
# Robust Face Detector Initializer
# -------------------------------------------------------------------------
XML_NAME = "haarcascade_frontalface_default.xml"


@st.cache_resource
def load_face_cascade():
    """Locates or downloads the Haar Cascade file safely on the server."""
    # Attempt 1: Standard OpenCV directory
    cascade_path = os.path.join(cv2.data.haarcascades, XML_NAME)

    if not os.path.exists(cascade_path):
        # Attempt 2: Download directly from OpenCV's official repo
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{XML_NAME}"
        try:
            urllib.request.urlretrieve(url, XML_NAME)
            cascade_path = XML_NAME
        except Exception:
            return None

    return cv2.CascadeClassifier(cascade_path)


face_cascade = load_face_cascade()

# -------------------------------------------------------------------------
# Image Processing Functions
# -------------------------------------------------------------------------


def pixelate_face_only(image, blocks=12):
    """Detects faces and pixelates only the detected face bounding boxes."""
    cv_img = np.array(image.convert("RGB"))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    if face_cascade is None or face_cascade.empty():
        st.error(
            "Face detector model could not be loaded. Showing original image."
        )
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        st.warning("No faces detected on the left image.")
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    for x, y, w, h in faces:
        face_roi = cv_img[y : y + h, x : x + w]

        # Shrink and expand the face region to pixelate it
        bw = max(1, w // blocks)
        bh = max(1, h // blocks)
        small = cv2.resize(face_roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Merge back into the image
        cv_img[y : y + h, x : x + w] = pixelated_roi

    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


def color_pencil_sketch(image):
    """Converts image to a vibrant color pencil sketch."""
    cv_img = np.array(image.convert("RGB"))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    # Built-in OpenCV pencil sketch filter
    _, color_sketch = cv2.pencilSketch(
        cv_img, sigma_s=50, sigma_r=0.05, shade_factor=0.04
    )

    return cv2.cvtColor(color_sketch, cv2.COLOR_BGR2RGB)


# -------------------------------------------------------------------------
# UI Layout
# -------------------------------------------------------------------------

# The file uploader is fixed at the top
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

# Side-by-side columns are declared early to stay open together
col1, col2 = st.columns(2)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    with col1:
        st.subheader("Left: Face Pixelated Only")
        pixelated_result = pixelate_face_only(img, blocks=12)
        st.image(pixelated_result, width="stretch")

    with col2:
        st.subheader("Right: Color-Penciled Sketch")
        sketch_result = color_pencil_sketch(img)
        st.image(sketch_result, width="stretch")
else:
    with col1:
        st.info("Awaiting image upload for the Left filter...")
    with col2:
        st.info("Awaiting image upload for the Right filter...")
