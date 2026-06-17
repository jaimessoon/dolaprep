import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Set up page layout
st.set_page_config(layout="wide", page_title="Dual Face Filter App")
st.title("🎭 Dual Image Effects Processor")
st.write("Upload an image to see the split effects.")

# -------------------------------------------------------------------------
# Image Processing Functions
# -------------------------------------------------------------------------


def pixelate_image(image, blocks=10):
    """Pixelates the entire image heavily by downsizing and upsizing."""
    cv_img = np.array(image.convert("RGB"))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    # Get original size
    h, w = cv_img.shape[:2]

    # Resize down to a very small size, then scale back up to create blocks
    # Using 'blocks' to control intensity. Lower blocks = more pixelated.
    small = cv2.resize(cv_img, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return cv2.cvtColor(pixelated, cv2.COLOR_BGR2RGB)


def color_pencil_sketch(image):
    """Converts image to a vibrant color pencil sketch while maintaining high recognizability."""
    cv_img = np.array(image.convert("RGB"))
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    # Pencil sketch filter gives two outputs: grayscale sketch and color sketch
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
    # Load the uploaded image
    img = Image.open(uploaded_file)

    # Create two columns for the side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Left: Unrecognizable (Pixelated)")
        # Run pixelation function (blocks=12 ensures heavy blurring)
        pixelated_result = pixelate_image(img, blocks=12)
        st.image(pixelated_result, use_container_width=True)

    with col2:
        st.subheader("Right: Color-Penciled (Recognizable)")
        # Run color pencil function
        sketch_result = color_pencil_sketch(img)
        st.image(sketch_result, use_container_width=True)
else:
    st.info("Please upload an image file to activate the filters.")
