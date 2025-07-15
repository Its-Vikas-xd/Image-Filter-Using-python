import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from io import BytesIO

# --------------------------
# Streamlit Page Settings
# --------------------------
st.set_page_config(page_title="🎨 Image Filter App", layout="centered", page_icon="📸")
st.title("📸 Image Filter App")
st.markdown("Upload your image, apply a filter, adjust brightness/contrast, and download the final result!")

# --------------------------
# Helper Functions
# --------------------------

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def apply_sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def apply_filters(img, filter_name):
    if filter_name == "Grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif filter_name == "Sepia":
        img = apply_sepia(img)
    elif filter_name == "Blur":
        img = cv2.GaussianBlur(img, (15, 15), 0)
    elif filter_name == "Canny Edge":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_name == "Pencil Sketch":
        _, sketch_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        img = sketch_color
    return img

def adjust_brightness_contrast(img_pil, brightness=1.0, contrast=1.0):
    enhancer_b = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer_b.enhance(brightness)
    enhancer_c = ImageEnhance.Contrast(img_pil)
    return enhancer_c.enhance(contrast)

def convert_to_downloadable(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def auto_resize(img_pil, max_dim=600):
    width, height = img_pil.size
    if max(width, height) > max_dim:
        ratio = max_dim / float(max(width, height))
        new_size = (int(width * ratio), int(height * ratio))
        return img_pil.resize(new_size)
    return img_pil

# --------------------------
# Main App Logic
# --------------------------

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image = auto_resize(image)

    # Show original image
    st.subheader("🖼️ Original Image")
    st.image(image, use_container_width=True)

    # Filter Selection (Single Only)
    filter_selected = st.selectbox("🧪 Select One Filter", 
                                   ["None", "Grayscale", "Sepia", "Blur", "Canny Edge", "Pencil Sketch"])

    # Brightness / Contrast controls
    st.markdown("### 🔧 Adjust Image Settings")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)

    # Apply brightness/contrast
    adjusted_pil = adjust_brightness_contrast(image, brightness, contrast)
    adjusted_cv2 = pil_to_cv2(adjusted_pil)

    # Apply selected filter
    if filter_selected != "None":
        filtered_cv2 = apply_filters(adjusted_cv2, filter_selected)
    else:
        filtered_cv2 = adjusted_cv2

    final_pil_image = cv2_to_pil(filtered_cv2)

    # Display final image
    st.markdown("### ✨ Processed Image")
    st.image(final_pil_image, use_container_width=True)

    # Download option
    st.markdown("### 📥 Download Your Image")
    img_bytes = convert_to_downloadable(final_pil_image)
    st.download_button("⬇️ Download as PNG", data=img_bytes,
                       file_name="filtered_image.png", mime="image/png")

else:
    st.info("👆 Please upload an image to begin.")
