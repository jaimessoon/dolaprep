import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

# Set up page layout to wide mode
st.set_page_config(layout="wide", page_title="Dual Face Filter App")
st.title("🎭 Advanced Image Effects Processor")
st.write(
    "Upload different images to see high-quality split effects side by side."
)

# -------------------------------------------------------------------------
# Core Helper Pixelation Function (Using Pure PIL)
# -------------------------------------------------------------------------


def apply_pixelation_to_box(img, x1, y1, x2, y2, blocks=10):
    """Safely pixelates a targeted bounding box region using pure PIL."""
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return img

    # Crop out the target face box region
    face_box = (x1, y1, x2, y2)
    face_roi = img.crop(face_box)

    # Calculate tiny downsample size based on block strength
    tw = max(1, (x2 - x1) // blocks)
    th = max(1, (y2 - y1) // blocks)

    # Scale down, then scale back up using NEAREST block resampling
    small = face_roi.resize((tw, th), resample=Image.Resampling.BILINEAR)
    pixelated_roi = small.resize(
        (x2 - x1, y2 - y1), resample=Image.Resampling.NEAREST
    )

    # Paste the blocky face region back onto the sharp original image copy
    result_img = img.copy()
    result_img.paste(pixelated_roi, face_box)
    return result_img


def estimate_face_centroid(img):
    """Estimates the middle-center of the picture coordinates as a baseline placement."""
    w, h = img.size
    return int(w * 0.5), int(h * 0.4)


# -------------------------------------------------------------------------
# Pure PIL Pencil Sketch Function (Matches Target Quality)
# -------------------------------------------------------------------------


def pil_artistic_sketch(img):
    """Generates an authentic, high-quality sketch filter with fine shading

    by calculating a soft color-dodge blend entirely using PIL.
    """
    # 1. Convert source image to Grayscale
    gray_img = img.convert("L")

    # 2. Create an Inverted copy of the grayscale frame
    inverted_img = ImageOps.invert(gray_img)

    # 3. Apply a massive Gaussian Blur to the inverted image to isolate shading gradients
    blurred_img = inverted_img.filter(ImageFilter.GaussianBlur(radius=15))

    # 4. Perform a high-precision Color Dodge blend between gray and blurred maps
    # This matches the identical pencil-shading look of your image2 example
    gray_arr = np.array(gray_img, dtype=np.float32)
    blur_arr = np.array(blurred_img, dtype=np.float32)

    # Avoid zero division errors safely
    blur_arr[blur_arr == 255] = 254
    sketch_arr = (gray_arr * 255) / (255 - blur_arr)
    sketch_arr = np.clip(sketch_arr, 0, 255).astype(np.uint8)

    sketch_mask = Image.fromarray(sketch_arr).convert("L")

    # 5. Blend the dark pencil lines gracefully back onto the original color layers
    # Mixes roughly 80% sketch framework lines + 20% original tone details
    final_sketch = Image.blend(img, sketch_mask.convert("RGB"), alpha=0.8)
    return final_sketch


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
            w, h = left_img.size

            # Dynamic automated slider presets on fresh file upload
            default_cx, default_cy = estimate_face_centroid(left_img)
            default_size = int(min(w, h) * 0.35)

            with st.expander("🎯 Target Your Face Location", expanded=True):
                pixel_strength = st.slider(
                    "Block Blur Size (Lower = More Blurry)", 4, 40, 15
                )
                box_size = st.slider("Box Size", 10, min(w, h), default_size)
                center_x = st.slider("Move Horizontal (X)", 0, w, default_cx)
                center_y = st.slider("Move Vertical (Y)", 0, h, default_cy)

                # Clamp box coordinates inside image dimensions safely
                mx1 = max(0, center_x - box_size // 2)
                my1 = max(0, center_y - box_size // 2)
                mx2 = min(w, center_x + box_size // 2)
                my2 = min(h, center_y + box_size // 2)

            # Process image array
            processed_left = apply_pixelation_to_box(
                left_img, mx1, my1, mx2, my2, blocks=pixel_strength
            )
            st.image(processed_left, use_container_width=True)

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

            with st.spinner("Rendering pencil portrait..."):
                processed_right = pil_artistic_sketch(right_img)

            st.image(processed_right, use_container_width=True)

        except Exception as e:
            st.error(f"Error handling right image: {e}")
    else:
        st.info("Awaiting right side picture choice...")
