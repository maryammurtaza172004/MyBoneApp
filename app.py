import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="BoneScan AI Pro", page_icon="🦴")

st.title("🦴 BoneScan AI: Fracture Detection")
st.write("Detecting structural anomalies in X-ray imaging...")

uploaded_file = st.file_uploader("Upload an X-ray here", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Prepare the Image
    input_image = Image.open(uploaded_file)
    image_np = np.array(input_image.convert('RGB'))
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Advanced Filtering (Cleaning the "noise" to find tiny cracks)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # 3. Finding the "Shapes" (Contours)
    contours, _ = cv2.find_contours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Draw Red Boxes around the most "suspicious" areas
    fracture_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 500: # This range looks for "tiny" to "medium" lines
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
            fracture_count += 1

    # 5. Severity Logic
    if fracture_count > 10:
        status = "HIGH SEVERITY"
        color = "red"
    elif 0 < fracture_count <= 10:
        status = "MODERATE/MINOR"
        color = "orange"
    else:
        status = "NO MAJOR ANOMALIES DETECTED"
        color = "green"

    # 6. Display Results
    st.subheader(f"Analysis Result: :{color}[{status}]")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Detection View (Red Boxes = Suspected Areas)")
    
    st.write(f"**Anomaly Count:** {fracture_count} areas flagged for review.")
    st.info("Note: This is a software simulation. All results must be verified by a Radiologist.")
    