import streamlit as st
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="AI Pothole Detection")

st.title("AI Based Pothole Detection System")
st.write("Upload a road image for pothole detection")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    results = model.predict(temp_path)

    result_image = results[0].plot()

    st.image(
        result_image,
        caption="Detection Result",
        use_container_width=True
    )

    os.remove(temp_path)
