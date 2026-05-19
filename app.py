import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.title("AI Based Pothole Detection System")

if not os.path.exists("best (2).pt"):
    st.error("best (2).pt model file not found")
    st.stop()

@st.cache_resource
def load_model():
    return YOLO("best (2).pt")

model = load_model()

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image = Image.open(uploaded)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp:

        image.save(tmp.name)

        results = model.predict(tmp.name)

    result = results[0].plot()

    st.image(
        result,
        caption="Detection Result",
        use_container_width=True
    )
