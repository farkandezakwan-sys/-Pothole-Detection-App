import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import glob

st.set_page_config(
    page_title="Pothole Detection",
    layout="centered"
)

st.title("AI Based Pothole Detection System")

st.write("Upload a road video for pothole detection")

# Load model once
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Choose Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    st.video(uploaded_file)

    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_file.write(uploaded_file.read())
    temp_file.close()

    st.info("Processing video... Please wait")

    try:

        # Run prediction
        results = model.predict(
            source=temp_file.name,
            save=True,
            imgsz=320,
            conf=0.4
        )

        save_dir = str(results[0].save_dir)

        # Find generated video
        output_videos = glob.glob(
            os.path.join(save_dir, "*.avi")
        )

        output_videos += glob.glob(
            os.path.join(save_dir, "*.mp4")
        )

        if len(output_videos) > 0:

            output_video = output_videos[0]

            st.success("Detection Completed")

            st.video(output_video)

            with open(output_video, "rb") as file:

                st.download_button(
                    label="Download Result",
                    data=file,
                    file_name="detected_video.mp4"
                )

        else:

            st.error("Output video not generated")

    except Exception as e:

        st.error(f"Error: {e}")
