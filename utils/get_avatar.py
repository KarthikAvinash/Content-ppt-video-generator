import streamlit as st
import requests
import os
import shutil
from utils.avatar_generation_util_functions import *

# Base URL of the FastAPI application

def get_avatar():
    st.title("Generate the Avatar")

    # Display a header
    st.header("We will connect to the api that will process the given audio and make it into a video using a reference video.")

    # Display a message with a clickable link
    st.write("""
        Make sure that audios are ready.
        Also keep a reference video which will be looped over in order to generate a talking avatar video.
        Make sure that the reference video is as small as possible (recommended length: 5-10 seconds). 
        And ensure that the reference video has a person in front of a green screen. 
        And the person must not have any green color on him/her.
    """)

    # Check if server is running
    # if not check_server():
    #     st.error("The FastAPI server is not running. Please start the server.")
    #     return

    st.info("Server is running. Waiting for responses...")

    # Allow user to upload audio files
    audio_files = st.file_uploader("Upload Audio Files (MP3 format)", accept_multiple_files=True, type=["mp3",".wav"])
    if audio_files:
        if os.path.exists(UPLOADED_AUDIOS_DIR):
            shutil.rmtree(UPLOADED_AUDIOS_DIR)
        os.mkdir(UPLOADED_AUDIOS_DIR)
        # Provide hint to upload fewer files
        st.warning("Hint: Upload fewer audio files to reduce processing time.")

        # Save uploaded files to disk and process them
        file_paths = []
        for audio_file in audio_files:
            file_path = os.path.join(UPLOADED_AUDIOS_DIR, audio_file.name)
            with open(file_path, "wb") as f:
                f.write(audio_file.getbuffer())
            file_paths.append(file_path)

        # Upload audio files
        st.write("Uploading audio files...")
        upload_response = upload_audio(file_paths)
        st.write(upload_response)

    # Allow user to upload face video
    face_video = st.file_uploader("Upload Face Video (MP4 format)", type=["mp4"])
    if face_video:
        if os.path.exists(REFERENCE_VIDEO_DIR):
            shutil.rmtree(REFERENCE_VIDEO_DIR)
        os.mkdir(REFERENCE_VIDEO_DIR)
        video_path = os.path.join(REFERENCE_VIDEO_DIR, face_video.name)
        with open(video_path, "wb") as f:
            f.write(face_video.getbuffer())

        # Set face video
        st.write("Setting face video...")
        face_response = set_face_video(video_path)
        st.write(face_response)

    # Process videos
    if st.button("Process Videos"):
        st.write("Processing videos. This may take some time...")
        process_response = process_videos()
        st.write(process_response)
