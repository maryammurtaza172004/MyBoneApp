import streamlit as st
import cv2
import numpy as np

# This makes the app look clean
st.set_page_config(page_title="BoneScan AI", page_icon="🦴")

st.title("🦴 My Bone Scanner App")
st.write("Professional Biomedical Edge Detection System")

uploaded_file = st.file_uploader("Upload an X-ray here", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Processing the file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # 2. Creating two columns (Side by side)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Scan")
        st.image(image, use_column_width=True)

    # 3. The Medical Brain
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150) # Tweaked for better detail

    with col2:
        st.subheader("Scanner View")
        st.image(edges, use_column_width=True)

    # 4. The Result Message
    st.success("✅ Scan completed successfully!")
    st.info("Note: This tool is for educational purposes. Consult a doctor for medical advice.")