import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(page_title="AI Pothole Detection")

st.title("AI Based Pothole Detection System")
st.write("Upload a road image")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded)

    st.image(
        image,
        caption="Uploaded Image"
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp:

        image.save(tmp.name)

        result = model.predict(
            source=tmp.name,
            save=False
        )

    plotted = result[0].plot()

    st.image(
        plotted,
        caption="Detection Result"
    )
