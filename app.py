import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import glob

st.set_page_config(
    page_title="AI Pothole Detection",
    layout="wide"
)

st.title("AI Based Pothole Detection System")

st.write("Upload one or multiple road videos")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_files = st.file_uploader(
    "Upload Videos",
    type=["mp4","avi","mov"],
    accept_multiple_files=True
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        st.subheader(uploaded_file.name)

        # Save uploaded file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_file.write(uploaded_file.read())
        temp_file.close()

        with st.spinner("Detecting potholes..."):

            results = model.predict(
                source=temp_file.name,
                save=True,
                conf=0.5
            )

        # Get save directory from YOLO
        save_dir = str(results[0].save_dir)

        # Search output videos
        output_files = (
            glob.glob(os.path.join(save_dir,"*.mp4"))
            + glob.glob(os.path.join(save_dir,"*.avi"))
            + glob.glob(os.path.join(save_dir,"*.mov"))
        )

        if output_files:

            output_video = output_files[0]

            st.success("Detection completed")

            st.video(output_video)

            with open(output_video,"rb") as f:

                st.download_button(
                    "Download Result",
                    f,
                    file_name="detected_"+uploaded_file.name
                )

        else:

            st.error("No output video generated")
