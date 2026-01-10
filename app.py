import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="UAV Detection", layout="centered")

st.title("🛡️ UAV / Aerial Object Detection")
st.write("Upload an image to detect aerial objects using YOLOv8.")

# Load model (cached so it loads once)
@st.cache_resource
def load_model():
    return YOLO("model/dron.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Running YOLO detection..."):
            results = model(image, conf=0.25)

            # YOLO auto-draws boxes + labels
            result_img = results[0].plot()

            st.image(
                result_img,
                caption="Detection Result",
                use_column_width=True
            )

        st.success("Detection completed")
