import os
import urllib.request
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Set up page layout to wide mode
st.set_page_config(layout="wide", page_title="Dual Face Filter App")
st.title("🎭 Dual Image Effects Processor")
st.write("Upload different images to see the distinct effects side by side.")

# -------------------------------------------------------------------------
# Robust Face Detector Initializer
# -------------------------------------------------------------------------
XML_NAME = "haarcascade_frontalface_default.xml"


@st.cache_resource
def load_face_cascade():
    """Locates or downloads the Haar Cascade file safely on the server."""
    cascade_path = os.path.join(cv2.data.haarcascades, XML_NAME)

    if not os.path.exists(cascade_path):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{XML_NAME}"
        try:
            urllib.request.urlretrieve(url, XML_NAME)
            cascade_path = XML_NAME
        except Exception:
            return None

    return cv2.CascadeClassifier(cascade_path)


face_cascade = load_face_cascade()

# -------------------------------------------------------------------------
# Processing Functions
# -------------------------------------------------------------------------


def pixelate_face_only(image, blocks=10):
    """Detects faces and pixelates ONLY the bounding box areas securely."""
    rgb_img = image.convert("RGB")
    cv_img = np.array(rgb_img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    h, w = cv_img.shape[:2]

    if face_cascade is None or face_cascade.empty():
        st.error("Face detector model could not be loaded on the server.")
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        st.warning(
            "No distinct faces detected on the left image. Showing original background."
        )
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    for x, y, fw, fh in faces:
        # Standardize coordinates within image bounds to prevent out-of-index crashes
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + fw)
        y2 = min(h, y + fh)

        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        face_roi = cv_img[y1:y2, x1:x2]

        # Calculate block downsampling steps
        bw = max(1, (x2 - x1) // blocks)
        bh = max(1, (y2 - y1) // blocks)

        # Apply block rendering
        small = cv2.resize(face_roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(
            small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
        )

        # Paste pixelated region back into the background frame
        cv_img[y1:y2, x1:x2] = pixelated_roi

    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


def color_pencil_sketch(image):
    """Creates a beautiful color pencil effect that guarantees high face recognizability

    by blending sketch outlines over the original vivid image layers.
    """
    rgb_img = image.convert("RGB")
    cv_img = np.array(rgb_img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    try:
        # 1. Generate a clean grayscale pencil sketch (isolating the lines)
        gray_sketch, _ = cv2.pencilSketch(
            cv_img, sigma_s=30, sigma_r=0.07, shade_factor=0.03
        )

        # 2. Convert the single-channel grayscale sketch back to 3-channel BGR
        sketch_bgr = cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR)

        # 3. Blend the outlines directly with the original image
        # cv2.multiply mixes the dark pencil lines on top of the original colors
        blended = cv2.multiply(cv_img, sketch_bgr, scale=1.0 / 255.0)

        # 4. Mix 70% blended sketch + 30% original crisp image color to ensure high recognition
        final_result = cv2.addWeighted(blended, 0.7, cv_img, 0.3, 0)

        return cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

    except Exception:
        # Fallback if processing encounters an unexpected matrix issue
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


# -------------------------------------------------------------------------
# Side-by-Side User Interface Layout
# -------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Left Side Filter")
    left_file = st.file_uploader(
        "Upload Image to Pixelate Face",
        type=["jpg", "jpeg", "png"],
        key="left_up",
    )

    if left_file is not None:
        try:
            left_img = Image.open(left_file)
            # blocks=8 ensures heavy, unidentifiable blocks over the face coordinates
            pixelated_result = pixelate_face_only(left_img, blocks=8)
            st.image(pixelated_result, use_container_width=True)
        except Exception as e:
            st.error(f"Error handling left image: {e}")
    else:
        st.info("Awaiting left side picture choice...")

with col2:
    st.subheader("Right Side Filter")
    right_file = st.file_uploader(
        "Upload Image for Sketch Filter",
        type=["jpg", "jpeg", "png"],
        key="right_up",
    )

    if right_file is not None:
        try:
            right_img = Image.open(right_file)
            sketch_result = color_pencil_sketch(right_img)
            st.image(sketch_result, use_container_width=True)
        except Exception as e:
            st.error(f"Error handling right image: {e}")
    else:
        st.info("Awaiting right side picture choice...")
