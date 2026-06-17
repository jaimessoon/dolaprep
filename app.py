import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Set up page layout
st.set_page_config(layout="wide", page_title="Dual Face Filter App")
st.title("🎭 Dual Image Effects Processor")
st.write("Upload an image to see the split effects side by side.")

# Load the built-in OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------------------------------------------------
# Image Processing Functions
# -------------------------------------------------------------------------


def pixelate_face_only(image, blocks=8):
    """Detects faces and pixelates only the detected face bounding boxes."""
    cv_img = np.array(image.convert("RGB"))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
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

        bw = max(1, w // blocks)
        bh = max(1, h // blocks)

        small = cv2.resize(face_roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        cv_img[y : y + h, x : x + w] = pixelated_roi

    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


def color_pencil_sketch(image):
    """Converts image to a vibrant color pencil sketch."""
    cv_img = np.array(image.convert("RGB"))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    _, color_sketch = cv2.pencilSketch(
        cv_img, sigma_s=50, sigma_r=0.05, shade_factor=0.04
    )

    return cv2.cvtColor(color_sketch, cv2.COLOR_BGR2RGB)


# -------------------------------------------------------------------------
# UI Layout
# -------------------------------------------------------------------------

# Place the upload field clearly at the top
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

# Create the left and right layout columns so they are always ready
col1, col2 = st.columns(2)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Process and display Left side immediately
    with col1:
        st.subheader("Left: Face Pixelated Only")
        pixelated_result = pixelate_face_only(img, blocks=12)
        # Fixed: Updated parameter to match the new Streamlit API specification
        st.image(pixelated_result, width="stretch")

    # Process and display Right side concurrently
    with col2:
        st.subheader("Right: Color-Penciled Sketch")
        sketch_result = color_pencil_sketch(img)
        # Fixed: Updated parameter to match the new Streamlit API specification
        st.image(sketch_result, width="stretch")
else:
    # Optional placeholders or messages while waiting for an upload
    with col1:
        st.info("Awaiting image upload for the Left filter...")
    with col2:
        st.info("Awaiting image upload for the Right filter...")
