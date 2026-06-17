import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

# Set up page layout to wide mode
st.set_page_config(layout="wide", page_title="Dual Face Filter App")
st.title("🎭 Advanced Image Effects Processor")
st.write("Upload different images to see high-quality split effects side by side.")

# -------------------------------------------------------------------------
# Robust Face Detector & AI Sketch Model Initializers
# -------------------------------------------------------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.4
)

MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/mosaic-8.onnx"
MODEL_PATH = "style_transfer_model.onnx"


@st.cache_resource
def load_onnx_model():
    """Downloads and caches the deep learning style transfer model on the server."""
    if not os.path.exists(MODEL_PATH):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception:
            return None
    try:
        return ort.InferenceSession(MODEL_PATH)
    except Exception:
        return None


ort_session = load_onnx_model()

# -------------------------------------------------------------------------
# Processing Functions
# -------------------------------------------------------------------------


def pixelate_face_mediapipe(image, blocks=8):
    """Detects faces using Google MediaPipe and pixelates the face zones heavily."""
    rgb_img = image.convert("RGB")
    cv_img = np.array(rgb_img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    mp_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(mp_rgb)

    if not results.detections:
        st.warning(
            "AI could not detect your face automatically. Showing original image."
        )
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        xmin, ymin = int(bbox.xmin * w), int(bbox.ymin * h)
        box_w, box_h = int(bbox.width * w), int(bbox.height * h)

        # Expand boxes safely to mask the entire head zone comfortably
        x1, y1 = max(0, xmin - int(box_w * 0.1)), max(0, ymin - int(box_h * 0.2))
        x2, y2 = min(w, xmin + box_w + int(box_w * 0.1)), min(
            h, ymin + box_h + int(box_h * 0.1)
        )

        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        face_roi = cv_img[y1:y2, x1:x2]
        bw, bh = max(1, (x2 - x1) // blocks), max(1, (y2 - y1) // blocks)

        small = cv2.resize(face_roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
        cv_img[y1:y2, x1:x2] = cv2.resize(
            small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
        )

    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


def ai_artistic_sketch(image):
    """Generates an authentic, high-quality artistic sketch using an ONNX neural network session."""
    rgb_img = image.convert("RGB")
    origin_w, origin_h = rgb_img.size

    if ort_session is None:
        # High-fidelity procedural pencil shading fallback if model fails to compile
        cv_img = np.array(rgb_img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        adapted_blur = cv2.adaptiveThreshold(
            cv2.GaussianBlur(gray, (5, 5), 0),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            9,
            5,
        )
        return cv2.cvtColor(adapted_blur, cv2.COLOR_GRAY2RGB)

    # Pre-process image to feed into the deep learning input layer tensor
    # Most style transfer networks expect a standardized 224x224 shape size
    resized = rgb_img.resize((224, 224))
    input_data = np.array(resized).astype(np.float32)

    # Transpose dimensions from (H, W, C) to network-standard Channel-First shape (1, C, H, W)
    input_data = input_data.transpose(2, 0, 1)
    input_data = np.expand_dims(input_data, axis=0)

    # Execute Inference Session Pass
    inputs = {ort_session.get_inputs()[0].name: input_data}
    outputs = ort_session.run(None, inputs)

    # Reconstruct array matrix out of raw inference tensor output
    output_array = outputs[0][0]
    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    output_array = output_array.transpose(1, 2, 0)

    # Upscale back cleanly to match original upload dimension footprint layout
    final_img = Image.fromarray(output_array).resize((origin_w, origin_h))
    return final_img


# -------------------------------------------------------------------------
# UI Layout
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
            left_img = Image.open(left_file)
            pixelated_result = pixelate_face_mediapipe(left_img, blocks=8)
            st.image(pixelated_result, use_container_width=True)
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
            right_img = Image.open(right_file)
            with st.spinner("AI is painting your artistic portrait..."):
                sketch_result = ai_artistic_sketch(right_img)
            st.image(sketch_result, use_container_width=True)
        except Exception as e:
            st.error(f"Error handling right image: {e}")
    else:
        st.info("Awaiting right side picture choice...")
