"""
Mini Photoshop — CV Assignment Scaffold
=========================================
This file provides the app shell (UI, pipeline management, caching, undo).
Students implement the actual image-processing functions in the
"STUDENT IMPLEMENTATION" section below using OpenCV.

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io


# =============================================================================
# STUDENT IMPLEMENTATION — this is the only section students edit.
# Each function takes a BGR np.ndarray (OpenCV convention) + params,
# and returns a new BGR np.ndarray. Do not mutate the input in place.
# =============================================================================

def op_brightness_contrast(img: np.ndarray, brightness: int, contrast: float) -> np.ndarray:
    """
    brightness: -100 to 100 (additive)
    contrast:   0.5 to 3.0  (multiplicative)
    """
    # TODO: implement with OpenCV
    #raise NotImplementedError
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def op_blur(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    kernel_size: odd integer, e.g. 3-31
    sigma: gaussian sigma
    """
    # TODO: implement with OpenCV
    #raise NotImplementedError
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(1, kernel_size)
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma)


def op_sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    """
    amount: strength of sharpening, e.g. 0.0-3.0
    """
    # TODO: implement with OpenCV (e.g. unsharp mask: original + amount*(original - blurred))
    #raise NotImplementedError
    if amount == 0:
        return img.copy()  # no sharpening requested; still return a new array
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def op_rotate(img: np.ndarray, angle_degrees: float, scale: float) -> np.ndarray:
    """
    angle_degrees: rotation angle
    scale: uniform scale factor applied during rotation
    """
    # TODO: implement using cv2.getRotationMatrix2D + cv2.warpAffine
    #raise NotImplementedError
    h, w = img.shape[:2]
    center = (w / 2, h / 2)  # rotate about the image center
    M = cv2.getRotationMatrix2D(center, angle_degrees, scale)
    return cv2.warpAffine(img, M, (w, h))



def op_translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    # TODO: implement using a translation matrix + cv2.warpAffine
    #raise NotImplementedError
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h))


def op_perspective_warp(img: np.ndarray,
                        dx0: int = 0, dy0: int = 0,  # top-left corner offset
                        dx1: int = 0, dy1: int = 0,  # top-right corner offset
                        dx2: int = 0, dy2: int = 0,  # bottom-right corner offset
                        dx3: int = 0, dy3: int = 0   # bottom-left corner offset
                        ) -> np.ndarray:
    h, w = img.shape[:2]
    # Source: the 4 corners of the original image
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Destination: each corner shifted by its slider values
    dst = np.float32([[0 + dx0, 0 + dy0],
                      [w + dx1, 0 + dy1],
                      [w + dx2, h + dy2],
                      [0 + dx3, h + dy3]])
    # Solve for the perspective matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp into an output the same size as the input
    return cv2.warpPerspective(img, M, (w, h))



# Registry: maps a human-readable op name to (function, param spec).
# Param spec drives the auto-generated sliders below — students should
# NOT need to touch the UI code to add params, just edit this dict entry
# if they add a new op (optional extension, not required for base assignment).
OPS = {
    "Brightness/Contrast": {
        "fn": op_brightness_contrast,
        "params": {
            "brightness": {"type": "slider", "min": -100, "max": 100, "default": 0},
            "contrast":   {"type": "slider", "min": 0.5, "max": 3.0, "default": 1.0, "step": 0.05},
        },
    },
    "Gaussian Blur": {
        "fn": op_blur,
        "params": {
            "kernel_size": {"type": "slider", "min": 1, "max": 31, "default": 1, "step": 2},
            "sigma":       {"type": "slider", "min": 0.0, "max": 10.0, "default": 0.0, "step": 0.1},
        },
    },
    "Sharpen": {
        "fn": op_sharpen,
        "params": {
            "amount": {"type": "slider", "min": 0.0, "max": 3.0, "default": 0.0, "step": 0.1},
        },
    },
    "Rotate/Scale": {
        "fn": op_rotate,
        "params": {
            "angle_degrees": {"type": "slider", "min": -180, "max": 180, "default": 0},
            "scale":         {"type": "slider", "min": 0.1, "max": 3.0, "default": 1.0, "step": 0.05},
        },
    },
    "Translate": {
        "fn": op_translate,
        "params": {
            "dx": {"type": "slider", "min": -300, "max": 300, "default": 0},
            "dy": {"type": "slider", "min": -300, "max": 300, "default": 0},
        },
    },

    "Perspective Warp": {
        "fn": op_perspective_warp,
        "params": {
            "dx0": {"type": "slider", "min": -200, "max": 200, "default": 0},
            "dy0": {"type": "slider", "min": -200, "max": 200, "default": 0},
            "dx1": {"type": "slider", "min": -200, "max": 200, "default": 0},
            "dy1": {"type": "slider", "min": -200, "max": 200, "default": 0},
            "dx2": {"type": "slider", "min": -200, "max": 200, "default": 0},
            "dy2": {"type": "slider", "min": -200, "max": 200, "default": 0},
            "dx3": {"type": "slider", "min": -200, "max": 200, "default": 0},
            "dy3": {"type": "slider", "min": -200, "max": 200, "default": 0},
        },
    },

}

# =============================================================================
# APP SHELL — provided. Handles image I/O, pipeline (op stack), undo,
# caching, and rendering. Students should not need to edit below this line.
# =============================================================================

st.set_page_config(page_title="Mini Photoshop", layout="wide")


def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


@st.cache_data(show_spinner=False)
def run_op(op_name: str, img_bytes: bytes, params_tuple: tuple) -> bytes:
    """
    Cache key = (op_name, input image bytes, param values).
    This means re-running the SAME op with the SAME upstream image and
    SAME params is free — only genuinely new (op, input, params)
    combinations trigger real computation. This is what keeps the
    pipeline fast as students add more steps.
    """
    img = pil_to_cv(Image.open(io.BytesIO(img_bytes)))
    params = dict(params_tuple)
    fn = OPS[op_name]["fn"]
    result = fn(img, **params)
    out_img = cv_to_pil(result)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return buf.getvalue()


def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- Session state: pipeline is a list of (op_name, params) applied in order ---
if "pipeline" not in st.session_state:
    st.session_state.pipeline = []  # list of dicts: {"op": str, "params": dict}
if "original_bytes" not in st.session_state:
    st.session_state.original_bytes = None


st.title("Mini Photoshop")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    st.session_state.original_bytes = uploaded.getvalue()

if st.session_state.original_bytes is None:
    st.info("Upload an image to begin.")
    st.stop()

col_controls, col_before, col_after = st.columns([1, 1, 1])

# --- Sidebar-ish controls: add a new op to the pipeline ---
with col_controls:
    st.subheader("Pipeline")

    new_op = st.selectbox("Add operation", list(OPS.keys()))
    if st.button("Add to pipeline"):
        default_params = {k: v["default"] for k, v in OPS[new_op]["params"].items()}
        st.session_state.pipeline.append({"op": new_op, "params": default_params})

    st.divider()

    # Render sliders for each op currently in the pipeline, in order.
    # Also provide remove/reorder controls (undo = remove last step).
    for i, step in enumerate(st.session_state.pipeline):
        with st.expander(f"{i+1}. {step['op']}", expanded=True):
            spec = OPS[step["op"]]["params"]
            for pname, pspec in spec.items():
                slider_kwargs = dict(
                    label=pname,
                    min_value=pspec["min"],
                    max_value=pspec["max"],
                    value=step["params"][pname],
                    key=f"{i}_{pname}",
                )
                if "step" in pspec:
                    slider_kwargs["step"] = pspec["step"]
                step["params"][pname] = st.slider(**slider_kwargs)

            if st.button("Remove this step", key=f"remove_{i}"):
                st.session_state.pipeline.pop(i)
                st.rerun()

    if st.button("Undo last step", disabled=len(st.session_state.pipeline) == 0):
        st.session_state.pipeline.pop()
        st.rerun()

    if st.button("Clear all"):
        st.session_state.pipeline = []
        st.rerun()

# --- Run the pipeline: each step's cached result feeds the next step's input ---
current_bytes = st.session_state.original_bytes
errors = []
for step in st.session_state.pipeline:
    params_tuple = tuple(sorted(step["params"].items()))
    try:
        current_bytes = run_op(step["op"], current_bytes, params_tuple)
    except NotImplementedError:
        errors.append(step["op"])
        break
    except Exception as e:
        st.error(f"Error in '{step['op']}': {e}")
        break

with col_before:
    st.subheader("Original")
    st.image(st.session_state.original_bytes, width='stretch')

with col_after:
    st.subheader("Result")
    if errors:
        st.warning(f"'{errors[0]}' is not implemented yet — showing pipeline up to that point.")
    st.image(current_bytes, width='stretch')
    st.download_button("Download result", data=current_bytes, file_name="result.png")
