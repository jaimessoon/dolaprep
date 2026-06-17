import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Set up page layout to wide mode
st.set_page_config(layout="wide", page_title="Dual Face Filter App")
st.title("🎭 Advanced Image Effects Processor")
st.write("Upload different images to see high-quality split effects side by side.")

# -------------------------------------------------------------------------
# Processing Helper Functions
# -------------------------------------------------------------------------


def apply_pixelation_to_box(cv_img, x1, y1, x2, y2, blocks=10):
    """Safely pixelates a targeted bounding box region."""
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


def estimate_face_centroid(cv_img):
    """A pure-python, crash-proof facial center estimator based on skin-tone thresholding.

    Works flawlessly across all Python versions without needing external XML models.
    """
    h, w = cv_img.shape[:2]
    # Convert to HSV color space to easily isolate human skin tones
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    moments = cv2.moments(mask)

    if moments["m00"] > 500:
        # Calculate coordinate center of mass
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy
    return int(w * 0.5), int(h * 0.4)


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

            # Estimate face location automatically on upload
            default_cx, default_cy = estimate_face_centroid(cv_img)
            default_size = int(min(w, h) * 0.35)

            with st.expander("🎯 Fine-Tune Face Bounds Location", expanded=True):
                pixel_strength = st.slider(
                    "Block Blur Size (Lower = More Blurry)", 4, 30, 12
                )
                box_size = st.slider("Box Size", 10, min(w, h), default_size)
                center_x = st.slider("Move Horizontal (X)", 0, w, default_cx)
                center_y = st.slider("Move Vertical (Y)", 0, h, default_cy)

                # Clamp values to image bounds safely
                mx1 = max(0, center_x - box_size // 2)
                my1 = max(0, center_y - box_size // 2)
                mx2 = min(w, center_x + box_size // 2)
                my2 = min(h, center_y + box_size // 2)

            # Process pixelation area
            cv_img = apply_pixelation_to_box(
                cv_img, mx1, my1, mx2, my2, blocks=pixel_strength
            )

            final_left = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            st.image(final_left, use_container_width=True)

        except Exception as e:
            st.error(f"Error handling left image: {e}")
    else:
        st.info("Awaiting left side picture choice...")

with col2:
    st.subheader("Right Side: High-Quality Sketch Filter")
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

            # High-fidelity artistic hand-drawn sketch execution
            gray_img = cv2.cvtColor(cv_img_r, cv2.COLOR_BGR2GRAY)
            inverted_img = 255 - gray_img

            # Massive kernel size gives smooth charcoal drawing textures
            blurred = cv2.GaussianBlur(inverted_img, (31, 31), 0)
            inverted_blurred = 255 - blurred

            # Color-dodge calculation layer
            pencil_sketch = cv2.divide(gray_img, inverted_blurred, scale=256.0)
            sketch_bgr = cv2.cvtColor(pencil_sketch, cv2.COLOR_GRAY2BGR)

            # Soft multiply blend overlay preserves rich portrait data
            blended = cv2.multiply(cv_img_r, sketch_bgr, scale=1.0 / 255.0)

            # Blend 85% sketch contours + 15% original color to perfectly match image2 style
            final_right_cv = cv2.addWeighted(blended, 0.85, cv_img_r, 0.15, 0)

            final_right = cv2.cvtColor(final_right_cv, cv2.COLOR_BGR2RGB)
            st.image(final_right, use_container_width=True)

        except Exception as e:
            st.error(f"Error handling right image: {e}")
    else:
        st.info("Awaiting right side picture choice...")
