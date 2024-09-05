import streamlit as st
import os
import shutil
from utils.ppt_util_functions import *
from utils.video_util_functions import *
from PIL import Image
from moviepy.editor import ImageClip, concatenate_videoclips
import zipfile

# Directories and file paths
images_dir = "images"
avtr_video_dir = "avt_videos"
AUDIO_FILES = "audios"
output_dir = "output_files"  # Directory to store processed files
frames_video_file = os.path.join(output_dir, "main.mp4")
combined_audio_file = os.path.join(output_dir, "combined_audio.wav")
zip_file = os.path.join(output_dir, "processed_files.zip")

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Function to create a zip file
# Function to create a zip file with specified structure
def create_zip():
    with zipfile.ZipFile(zip_file, 'w') as zf:
        # Add combined audio file to the main directory
        if os.path.exists(combined_audio_file):
            zf.write(combined_audio_file, os.path.join('processed_files', 'combined_audio.wav'))
        
        # Add final video file to the main directory
        if os.path.exists(frames_video_file):
            zf.write(frames_video_file, os.path.join('processed_files', 'main.mp4'))
        
        # Create 'audios' folder and add all audio files to it
        audios_folder = os.path.join('processed_files', 'audios')
        os.makedirs(audios_folder, exist_ok=True)  # Ensure 'audios' folder exists in the zip file
        for root, dirs, files in os.walk(AUDIO_FILES):
            for file in files:
                file_path = os.path.join(root, file)
                # Add audio files to the 'audios' folder in the ZIP
                zf.write(file_path, os.path.join(audios_folder, file))


def get_audio():
    st.title("PDF to Video Converter")

    # File upload section
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file:
        # Save the uploaded PDF
        pdf_path = "uploaded_pdf.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the PDF
        if st.button("Process PDF"):
            # Initialize progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Processing PDF...")

            # Clear previous outputs
            delete_folder_and_create(images_dir)
            delete_folder_and_create(avtr_video_dir)
            delete_folder_and_create(output_dir)  # Clear the output directory
            
            progress_bar.progress(20)

            # Convert PDF to images
            status_text.text("Converting PDF to images...")
            convert_to_images(pdf_file_path=pdf_path, images_dir=images_dir)
            slides_content = get_slides_content()
            sent_slide_content = get_sentence_wise_slide_content(slides_content)

            # Text-to-Speech
            progress_bar.progress(45)
            status_text.text("Generating audio files...")
            if os.path.exists(AUDIO_FILES):
                shutil.rmtree(AUDIO_FILES)
            os.makedirs(AUDIO_FILES, exist_ok=True)
            text_to_speech_sentence_wise(sent_slide_content)

            # Combine audio files
            progress_bar.progress(70)
            status_text.text("Combining audio files...")
            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
            durations, slide_durations = combine_mp3_files(AUDIO_FILES, combined_audio_file)
            progress_bar.progress(85)  # Update progress

            # Create video from images and audio
            status_text.text("Creating video from images and audio...")
            image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
            if len(image_files) != len(slide_durations):
                st.error("Number of images and audio files do not match.")
                return

            clips = []
            for idx, img_file in enumerate(image_files):
                image_duration = slide_durations[idx]
                img_clip = ImageClip(os.path.join(images_dir, img_file), duration=image_duration)
                clips.append(img_clip)

            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(frames_video_file, codec="libx264", fps=29.97)
            progress_bar.progress(100)  # Update progress

            # Notify user that processing is complete
            st.success("Processing complete!")

    # Provide download links if the output directory exists
    if os.path.exists(output_dir):
        # # Download combined audio
        # if os.path.exists(combined_audio_file):
        #     with open(combined_audio_file, "rb") as audio_file:
        #         st.download_button(
        #             label="Download Combined Audio",
        #             data=audio_file,
        #             file_name="combined_audio.wav",
        #             mime="audio/wav"
        #         )
        # # Download all audio files
        # audio_files = sorted([f for f in os.listdir(AUDIO_FILES) if f.endswith(('.mp3','.wav'))])
        # for audio_file in audio_files:
        #     file_path = os.path.join(AUDIO_FILES, audio_file)
        #     with open(file_path, "rb") as file:
        #         st.download_button(
        #             label=f"Download {audio_file}",
        #             data=file,
        #             file_name=audio_file,
        #             mime="audio/mpeg"
        #         )
        # # Download final video
        # if os.path.exists(frames_video_file):
        #     with open(frames_video_file, "rb") as video_file:
        #         st.download_button(
        #             label="Download Final Video",
        #             data=video_file,
        #             file_name="main.mp4",
        #             mime="video/mp4"
        #         )

        # Create ZIP file
        create_zip()
        if os.path.exists(zip_file):
            with open(zip_file, "rb") as zf:
                    st.download_button(
                        label="Download Processed Files",
                        data=zf,
                        file_name="processed_files.zip",
                        mime="application/zip"
                    )
