import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Set up page layout to wide mode
st.set_page_config(layout="wide", page_title="Dual Face Filter App")
st.title("🎭 Dual Image Effects Processor")
st.write("Upload different images to see the distinct effects side by side.")

# -------------------------------------------------------------------------
# Processing Functions
# -------------------------------------------------------------------------


def pixelate_entire_image(image, blocks=14):
    """Guaranteed crash-proof pixelation by processing the matrix grid directly."""
    # Convert image strictly to 3-channel RGB
    rgb_img = image.convert("RGB")
    cv_img = np.array(rgb_img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    h, w = cv_img.shape[:2]

    # Downsample and upsample to blockify the image completely
    bw = max(1, w // blocks)
    bh = max(1, h // blocks)

    small = cv2.resize(cv_img, (bw, bh), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return cv2.cvtColor(pixelated, cv2.COLOR_BGR2RGB)


def color_pencil_sketch(image):
    """Converts image to a pencil sketch safely without channel mismatch crashes."""
    rgb_img = image.convert("RGB")
    cv_img = np.array(rgb_img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    try:
        _, color_sketch = cv2.pencilSketch(
            cv_img, sigma_s=50, sigma_r=0.05, shade_factor=0.04
        )
        return cv2.cvtColor(color_sketch, cv2.COLOR_BGR2RGB)
    except Exception:
        # Failsafe: if pencilSketch fails, fall back to an edgy cartoon filter
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(cv_img, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)


# -------------------------------------------------------------------------
# Side-by-Side User Interface
# -------------------------------------------------------------------------

# Create two permanent column splits across the viewport
col1, col2 = st.columns(2)

with col1:
    st.subheader("Left Side Filter")
    left_file = st.file_uploader(
        "Upload Image to Pixelate", type=["jpg", "jpeg", "png"], key="left_up"
    )

    if left_file is not None:
        left_img = Image.open(left_file)
        # Apply heavy pixelation filter to ensure it is unrecognizable
        pixelated_result = pixelate_entire_image(left_img, blocks=25)
        st.image(pixelated_result, use_container_width=True)
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
        right_img = Image.open(right_file)
        # Apply color pencil filter
        sketch_result = color_pencil_sketch(right_img)
        st.image(sketch_result, use_container_width=True)
    else:
        st.info("Awaiting right side picture choice...")
