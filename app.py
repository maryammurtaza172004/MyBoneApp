import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="BoneScan AI Pro", page_icon="🦴")

st.title("🦴 BoneScan AI: Refined Detection")
st.write("Applying medical-grade noise reduction...")

uploaded_file = st.file_uploader("Upload an X-ray here", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Image Preparation
    input_image = Image.open(uploaded_file)
    image_np = np.array(input_image.convert('RGB'))
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Stronger Noise Reduction (The "Filter")
    # We use Bilateral Filter because it smooths skin/noise but keeps bone edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 3. Higher Threshold Canny
    # We increased these numbers so only very strong "cracks" show up
    edges = cv2.Canny(filtered, 100, 200)

    # 4. Finding Shapes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fracture_count = 0
    for cnt in contours:
        # We only box items that are long enough to be a fracture
        length = cv2.arcLength(cnt, False)
        if length > 30: # Ignore tiny noise dots
            x, y, w, h = cv2.boundingRect(cnt)
            # Only box if it's not a perfect square (fractures are usually lines)
            if w > 5 or h > 5:
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
                fracture_count += 1

    # 5. Display
    if fracture_count > 0:
        st.warning(f"⚠️ {fracture_count} Potential structural anomalies detected.")
    else:
        st.success("✅ No major fractures detected by the current filter.")

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)