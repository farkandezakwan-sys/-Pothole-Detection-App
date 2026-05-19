import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import glob
import gc

st.set_page_config(
    page_title="AI Pothole Detection",
    layout="wide"
)

st.title("AI Based Pothole Detection System")

st.write("Upload one road video")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4","avi","mov"]
)

if uploaded_file:

    st.subheader(uploaded_file.name)

    temp_input = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_input.write(
        uploaded_file.read()
    )

    temp_input.close()

    with st.spinner("Detecting potholes..."):

        results = model.predict(
            source=temp_input.name,
            save=True,
            imgsz=320,
            conf=0.5
        )

    save_dir = str(results[0].save_dir)

    videos = (
        glob.glob(save_dir+"/*.mp4")
        + glob.glob(save_dir+"/*.avi")
    )

    if videos:

        output = videos[0]

        st.success("Detection completed")

        st.video(output)

        with open(output,"rb") as f:

            st.download_button(
                "Download Result",
                data=f,
                file_name="detected_video.mp4"
            )

    # Cleanup memory
    os.remove(temp_input.name)
    gc.collect()
