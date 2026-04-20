import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="BoneScan AI Pro", page_icon="🦴")

st.title("🦴 BoneScan AI: Fracture Detection")
st.write("Detecting structural anomalies in X-ray imaging...")

uploaded_file = st.file_uploader("Upload an X-ray here", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    image_np = np.array(input_image.convert('RGB'))
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # FIXED LINE BELOW: Changed find_contours to findContours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fracture_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 400: 
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
            fracture_count += 1

    if fracture_count > 15:
        st.error(f"🚨 ANALYSIS: HIGH SEVERITY ({fracture_count} anomalies)")
    elif 0 < fracture_count <= 15:
        st.warning(f"⚠️ ANALYSIS: MODERATE/MINOR ({fracture_count} anomalies)")
    else:
        st.success("✅ ANALYSIS: NO MAJOR ANOMALIES DETECTED")

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Detection View (Red Boxes = Suspected Areas)")
    st.info("Note: This is a software simulation. All results must be verified by a Radiologist.")