import streamlit as st
import requests
import os
import shutil

BASE_URL = "http://localhost:8000"

UPLOADED_AUDIOS_DIR = "uploaded_audios"
REFERENCE_VIDEO_DIR = "reference_video"

def check_server():
    """Check if the FastAPI server is running."""
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        return False

def upload_audio(file_paths):
    """Upload audio files to the FastAPI endpoint with error handling."""
    url = f"{BASE_URL}/upload-audio"
    try:
        files_to_upload = [('files', (os.path.basename(file_path), open(file_path, 'rb'), 'audio/mpeg')) for file_path in file_paths]
        response = requests.post(url, files=files_to_upload)
        response.raise_for_status()  # Check if the request was successful
        return response.json()
    except Exception as e:
        st.error(f"Error uploading audio files: {str(e)}")
        return {
            "status": "error",
            "message": "Failed to upload audio files.",
            "error_details": str(e)
        }

def set_face_video(file_path):
    """Set the reference/face video using the FastAPI endpoint with error handling."""
    url = f"{BASE_URL}/set-face-video"
    try:
        with open(file_path, 'rb') as video_file:
            files = {'file': (os.path.basename(file_path), video_file, 'video/mp4')}
            response = requests.post(url, files=files)
            response.raise_for_status()  # Check if the request was successful
            return response.json()
    except Exception as e:
        st.error(f"Error uploading face video: {str(e)}")
        return {
            "status": "error",
            "message": "Failed to upload face video.",
            "error_details": str(e)
        }

def process_videos():
    """Trigger the video processing endpoint and download the processed ZIP file with error handling."""
    url = f"{BASE_URL}/process-videos"
    try:
        response = requests.post(url, stream=True)
        response.raise_for_status()  # Check if the request was successful
        
        # Save the ZIP file to the local directory
        zip_filename = 'processed_videos.zip'
        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return f"ZIP file downloaded and saved as {zip_filename}"
    
    except Exception as e:
        st.error(f"Error processing videos: {str(e)}")
        return {
            "status": "error",
            "message": "Failed to process videos.",
            "error_details": str(e)
        }