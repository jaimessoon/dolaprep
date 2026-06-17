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
# Core Helper Pixelation Function
# -------------------------------------------------------------------------


def apply_pixelation_to_box(cv_img, x1, y1, x2, y2, blocks):
    """Helper to heavily pixelate a specific box region safely."""
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return cv_img

    face_roi = cv_img[y1:y2, x1:x2]
    bw = max(1, (x2 - x1) // blocks)
    bh = max(1, (y2 - y1) // blocks)

    small = cv2.resize(face_roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
    pixelated_roi = cv2.resize(
        small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
    )

    cv_img[y1:y2, x1:x2] = pixelated_roi
    return cv_img


# -------------------------------------------------------------------------
# Side-by-Side User Interface Layout
# -------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Left Side: Face Pixelation")
    left_file = st.file_uploader(
        "Upload Image to Pixelate Face",
        type=["jpg", "jpeg", "png"],
        key="left_up",
    )

    if left_file is not None:
        try:
            left_img = Image.open(left_file).convert("RGB")
            cv_img = np.array(left_img)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            h, w = cv_img.shape[:2]

            # 1. Run AI Detection
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = []
            if face_cascade is not None and not face_cascade.empty():
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30)
                )

            # 2. Check if AI found a face
            ai_found = len(faces) > 0

            # 3. Add control toggles in a clean expander box below the uploader
            with st.expander("Adjustment Settings", expanded=not ai_found):
                use_manual = st.checkbox(
                    "Enable Manual Override Box", value=not ai_found
                )
                pixel_strength = st.slider(
                    "Pixel Size (Lower = More Blurry)", 4, 30, 8
                )

                if use_manual:
                    st.info(
                        "Adjust sliders below to move the box directly over the face if the AI missed it."
                    )
                    # Create sliders proportional to the uploaded photo size
                    box_size = st.slider("Box Size", 10, min(w, h), int(min(w, h) * 0.3))
                    center_x = st.slider("Move Horizontal (X)", 0, w, int(w * 0.5))
                    center_y = st.slider("Move Vertical (Y)", 0, h, int(h * 0.4))

                    # Calculate box coordinates based on manual sliders
                    mx1 = max(0, center_x - box_size // 2)
                    my1 = max(0, center_y - box_size // 2)
                    mx2 = min(w, center_x + box_size // 2)
                    my2 = min(h, center_y + box_size // 2)

            # 4. Process image based on chosen method
            if use_manual:
                cv_img = apply_pixelation_to_box(
                    cv_img, mx1, my1, mx2, my2, blocks=pixel_strength
                )
            elif ai_found:
                for x, y, fw, fh in faces:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w, x + fw), min(h, y + fh)
                    cv_img = apply_pixelation_to_box(
                        cv_img, x1, y1, x2, y2, blocks=pixel_strength
                    )
            else:
                st.warning(
                    "AI could not detect your face structure. Please check 'Enable Manual Override Box' above to position it manually!"
                )

            # 5. Render Final Output
            final_left = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            st.image(final_left, use_container_width=True)

        except Exception as e:
            st.error(f"Error handling left image: {e}")
    else:
        st.info("Awaiting left side picture choice...")

with col2:
    st.subheader("Right Side: Sketch Filter")
    right_file = st.file_uploader(
        "Upload Image for Sketch Filter",
        type=["jpg", "jpeg", "png"],
        key="right_up",
    )

    if right_file is not None:
        try:
            right_img = Image.open(right_file).convert("RGB")
            cv_img_r = np.array(right_img)
            cv_img_r = cv2.cvtColor(cv_img_r, cv2.COLOR_RGB2BGR)

            # Beautiful blended sketch effect that preserves distinct color recognition
            gray_sketch, _ = cv2.pencilSketch(
                cv_img_r, sigma_s=30, sigma_r=0.07, shade_factor=0.03
            )
            sketch_bgr = cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR)
            blended = cv2.multiply(cv_img_r, sketch_bgr, scale=1.0 / 255.0)
            final_right_cv = cv2.addWeighted(blended, 0.7, cv_img_r, 0.3, 0)

            final_right = cv2.cvtColor(final_right_cv, cv2.COLOR_BGR2RGB)
            st.image(final_right, use_container_width=True)
        except Exception as e:
            st.error(f"Error handling right image: {e}")
    else:
        st.info("Awaiting right side picture choice...")
