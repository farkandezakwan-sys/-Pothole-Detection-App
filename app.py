
import streamlit as st
from ultralytics import YOLO
import tempfile
import glob

st.set_page_config(
    page_title="AI Pothole Detection",
    layout="wide"
)

st.title("AI Based Pothole Detection System")

st.write("Upload one or multiple road videos")

model = YOLO("best.pt")

uploaded_files = st.file_uploader(
    "Upload Videos",
    type=["mp4","avi","mov"],
    accept_multiple_files=True
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        st.subheader(uploaded_file.name)

        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(
            uploaded_file.read()
        )

        temp_input.close()

        results = model.predict(
            source=temp_input.name,
            save=True,
            conf=0.5
        )

        save_dir = str(results[0].save_dir)

        videos = glob.glob(save_dir+"/*.mp4")
        videos += glob.glob(save_dir+"/*.avi")

        if videos:

            output_video = videos[0]

            st.video(output_video)

            with open(output_video,"rb") as file:

                st.download_button(
                    label="Download Result",
                    data=file,
                    file_name="detected_"+uploaded_file.name
                )
