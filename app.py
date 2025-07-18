import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw, ImageFont
from io import BytesIO
import os

# --------------------------
# Streamlit Page Settings
# --------------------------
st.set_page_config(
    page_title="🎨 Advanced Image Filter App",
    layout="centered",
    page_icon="🖼️",
    initial_sidebar_state="expanded"
)
st.title("🖼️ Advanced Image Filter App")
st.markdown("Upload your image, apply filters, adjust settings, and download the result!")

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
        img = cv2.GaussianBlur(img, (25, 25), 0)
    elif filter_name == "Canny Edge":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_name == "Pencil Sketch":
        gray, sketch = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        img = sketch
    elif filter_name == "Invert Colors":
        img = cv2.bitwise_not(img)
    elif filter_name == "Oil Painting":
        img = cv2.xphoto.oilPainting(img, 7, 1)
    elif filter_name == "Emboss":
        kernel = np.array([[0, -1, -1], 
                           [1, 0, -1], 
                           [1, 1, 0]])
        img = cv2.filter2D(img, -1, kernel)

    elif filter_name == "Sharpen":
        kernel = np.array([[-1, -1, -1], 
                           [-1, 9, -1], 
                           [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
    return img

def adjust_brightness_contrast(img_pil, brightness=1.0, contrast=1.0, saturation=1.0):
    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer.enhance(brightness)
    
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(contrast)
    
    enhancer = ImageEnhance.Color(img_pil)
    return enhancer.enhance(saturation)

def convert_to_downloadable(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def auto_resize(img_pil, max_dim=800):
    width, height = img_pil.size
    if max(width, height) > max_dim:
        ratio = max_dim / float(max(width, height))
        new_size = (int(width * ratio), int(height * ratio))
        return img_pil.resize(new_size, Image.LANCZOS)
    return img_pil

def add_watermark(image, text, position, opacity=0.5):
    """Add text watermark to image"""
    watermark = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    
    # Use default font (size based on image dimensions)
    font_size = min(image.size) // 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text bounding box using textbbox instead of textsize
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    
    # Position handling
    margin = 20
    if position == "Bottom Right":
        x = image.width - text_width - margin
        y = image.height - text_height - margin
    elif position == "Top Left":
        x = margin
        y = margin
    elif position == "Top Right":
        x = image.width - text_width - margin
        y = margin
    elif position == "Bottom Left":
        x = margin
        y = image.height - text_height - margin
    else:  # Center
        x = (image.width - text_width) // 2
        y = (image.height - text_height) // 2
    
    # Draw text with opacity
    draw.text((x, y), text, font=font, fill=(255, 255, 255, int(255 * opacity)))
    
    # Composite with original image
    return Image.alpha_composite(image.convert("RGBA"), watermark).convert("RGB")

def apply_rotation(image, rotation):
    """Rotate image based on selection"""
    if rotation == 90:
        return image.transpose(Image.ROTATE_90)
    elif rotation == 180:
        return image.transpose(Image.ROTATE_180)
    elif rotation == 270:
        return image.transpose(Image.ROTATE_270)
    return image

# --------------------------
# Main App Logic
# --------------------------

# Filter options with categories
FILTER_OPTIONS = {
    "Basic": ["None", "Grayscale", "Sepia", "Blur", "Sharpen"],
    "Artistic": ["Pencil Sketch", "Oil Painting", "Emboss", "Vignette"],
    "Special Effects": ["Canny Edge", "Invert Colors"]
}

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# File uploader
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and process image
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image = auto_resize(image)
        
        # Display original image
        with st.expander("🖼️ Original Image", expanded=True):
            st.image(image, use_container_width=True)
        
        # Sidebar controls
        with st.sidebar:
            st.header("⚙️ Filter Settings")
            
            # Filter selection
            filter_category = st.selectbox("Filter Category", list(FILTER_OPTIONS.keys()))
            filter_selected = st.selectbox("Select Filter", FILTER_OPTIONS[filter_category])
            
            # Rotation
            rotation = st.selectbox("Rotate Image", [0, 90, 180, 270])
            image = apply_rotation(image, rotation)
            
            # Adjustments
            st.subheader("🎚 Adjustments")
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
            saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
            
            # Watermark
            st.subheader("💧 Watermark")
            watermark_text = st.text_input("Watermark Text", "")
            watermark_position = st.selectbox("Position", 
                                            ["Bottom Right", "Top Left", "Top Right", "Bottom Left", "Center"])
            watermark_opacity = st.slider("Opacity", 0.0, 1.0, 0.5, 0.1) if watermark_text else 0.0
            
            # Reset button
            if st.button("Reset All Settings"):
                st.session_state.clear()
                st.experimental_rerun()
        
        # Apply adjustments
        enhancer = ImageEnhance.Brightness(image)
        adjusted_pil = enhancer.enhance(brightness)
        
        enhancer = ImageEnhance.Contrast(adjusted_pil)
        adjusted_pil = enhancer.enhance(contrast)
        
        enhancer = ImageEnhance.Color(adjusted_pil)
        adjusted_pil = enhancer.enhance(saturation)
        
        # Apply filter
        adjusted_cv2 = pil_to_cv2(adjusted_pil)
        
        if filter_selected != "None":
            filtered_cv2 = apply_filters(adjusted_cv2, filter_selected)
        else:
            filtered_cv2 = adjusted_cv2
            
        final_pil_image = cv2_to_pil(filtered_cv2)
        
        # Apply watermark if text exists
        if watermark_text:
            final_pil_image = add_watermark(final_pil_image, watermark_text, 
                                           watermark_position, watermark_opacity)
        
        # Store in session state
        st.session_state.processed_image = final_pil_image
        
        # Display final image
        with st.expander("✨ Processed Image", expanded=True):
            st.image(final_pil_image, use_container_width=True)
            
            # Download option
            st.markdown("### 📥 Download Your Image")
            img_bytes = convert_to_downloadable(final_pil_image)
            st.download_button(
                "⬇️ Download as PNG", 
                data=img_bytes,
                file_name="filtered_image.png", 
                mime="image/png"
            )
            
            # Save to session state
            st.session_state.processed_image = final_pil_image
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.stop()

else:
    st.info("👆 Please upload an image to get started")
    st.session_state.processed_image = None

# Show sample images if no upload
if not uploaded_file and st.checkbox("Show Sample Images"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://picsum.photos/300/200?nature", caption="Nature Sample")
    with col2:
        st.image("https://picsum.photos/300/200?architecture", caption="Architecture Sample")
    with col3:
        st.image("https://picsum.photos/300/200?people", caption="People Sample")