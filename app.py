import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Set up page layout
st.set_page_config(layout="wide", page_title="Dual Face Filter App")
st.title("🎭 Dual Image Effects Processor")
st.write("Upload an image to see the split effects.")

# Load the built-in OpenCV face detector
# (Haar Cascade is lightweight and comes pre-packaged with cv2)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------------------------------------------------
# Image Processing Functions
# -------------------------------------------------------------------------


def pixelate_face_only(image, blocks=8):
    """Detects faces and pixelates only the detected face bounding boxes."""
    # Convert PIL to OpenCV BGR format
    cv_img = np.array(image.convert("RGB"))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # If no faces are found, return the original image
    if len(faces) == 0:
        st.warning("No faces detected on the left image. Showing original.")
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # Loop through each detected face and pixelate that region
    for x, y, w, h in faces:
        # Extract the face Region of Interest (ROI)
        face_roi = cv_img[y : y + h, x : x + w]

        # Downsample and upsample the face to create the blocky pixel effect
        # We ensure blocks width/height don't drop to 0
        bw = max(1, w // blocks)
        bh = max(1, h // blocks)

        small = cv2.resize(face_roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Paste the pixelated face back into the original image
        cv_img[y : y + h, x : x + w] = pixelated_roi

    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


def color_pencil_sketch(image):
    """Converts image to a vibrant color pencil sketch while maintaining high recognizability."""
    cv_img = np.array(image.convert("RGB"))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    # Pencil sketch filter
    _, color_sketch = cv2.pencilSketch(
        cv_img, sigma_s=50, sigma_r=0.05, shade_factor=0.04
    )

    return cv2.cvtColor(color_sketch, cv2.COLOR_BGR2RGB)


# -------------------------------------------------------------------------
# UI Components
# -------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Left: Face Pixelated Only")
        # Blocks parameter controls intensity. Lower = more pixelated/unrecognizable.
        pixelated_result = pixelate_face_only(img, blocks=12)
        st.image(pixelated_result, use_container_width=True)

    with col2:
        st.subheader("Right: Color-Penciled (Whole Image)")
        sketch_result = color_pencil_sketch(img)
        st.image(sketch_result, use_container_width=True)
else:
    st.info("Please upload an image file to activate the filters.")
