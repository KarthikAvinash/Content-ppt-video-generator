# import streamlit as st
# from avatar_util_functions import *
# from video_util_functions import delete_folder_and_create, convert_to_images

# def get_fin_video():
#     st.title("Here we will overlay the avatar videos with main.mp4 video.")

#     # Display a header
#     st.header("many steps are there.")

#     # Display a message with a clickable link
#     st.write("""
#         Instructions will be written here...
#     """)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
import streamlit as st
import zipfile
import os
import shutil
import cv2
from utils.avatar_util_functions import *
from utils.video_util_functions import delete_folder_and_create, convert_to_images

def get_fin_video():
    st.title("Overlay Avatar Videos with Main Video")

    # Display a header
    st.header("Steps to Create the Final Video")

    # Instructions
    st.write("""
        1. Upload the slides PDF file.
        2. Upload ZIP files containing audio files and avatar videos.
        3. Upload the reference video file.
    """)

    # Upload PDF
    slides_pdf = "processed_files/slides.pdf"
    uploaded_pdf = st.file_uploader("Upload Slides PDF", type=["pdf"])
    if uploaded_pdf is not None:
        delete_folder_and_create(os.path.dirname(slides_pdf))
        with open(slides_pdf, "wb") as f:
            f.write(uploaded_pdf.read())
        st.success("Slides PDF uploaded successfully.")

    # Upload Audio ZIP
    audios_dir = "processed_files/audios"
    uploaded_audio_zip = st.file_uploader("Upload Audios ZIP", type=["zip"])
    if uploaded_audio_zip is not None:
        delete_folder_and_create(audios_dir)
        with zipfile.ZipFile(uploaded_audio_zip, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # Get the file's path without the parent directory
                if not file_info.is_dir():
                    file_name = os.path.basename(file_info.filename)
                    if file_name:  # If the file has a name, extract it to the audios_dir
                        zip_ref.extract(file_info.filename, audios_dir)
                        # Move the file to the correct location
                        extracted_path = os.path.join(audios_dir, file_info.filename)
                        final_path = os.path.join(audios_dir, file_name)
                        os.rename(extracted_path, final_path)
        st.success("Audios extracted successfully.")

    # Upload Avatar Videos ZIP
    avt_vid_dir = "processed_files/avt_videos"
    uploaded_avatar_zip = st.file_uploader("Upload Avatar Videos ZIP", type=["zip"])
    if uploaded_avatar_zip is not None:
        delete_folder_and_create(avt_vid_dir)
        with zipfile.ZipFile(uploaded_avatar_zip, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.is_dir():
                    file_name = os.path.basename(file_info.filename)
                    if file_name:
                        zip_ref.extract(file_info.filename, avt_vid_dir)
                        # Move the file to the correct location
                        extracted_path = os.path.join(avt_vid_dir, file_info.filename)
                        final_path = os.path.join(avt_vid_dir, file_name)
                        os.rename(extracted_path, final_path)
        st.success("Avatar videos extracted successfully.")

    # Upload Reference Video
    reference_video = "processed_files/reference.mp4"
    uploaded_video = st.file_uploader("Upload Reference Video", type=["mp4"])
    if uploaded_video is not None:
        with open(reference_video, "wb") as f:
            f.write(uploaded_video.read())
        st.success("Reference video uploaded successfully.")

    # Process the inputs if all files are uploaded
    if uploaded_pdf and uploaded_audio_zip and uploaded_avatar_zip and uploaded_video:
        st.write("Processing the uploaded files...")

        durations, slide_durations = get_audio_durations(audios_dir)
        bg_vid_dir = "bg_vids"
        combined_videos = "fin_videos"
        temp_frames_store = "temp_frames_store"
        avatar_height = 380
        final_video = "final_video.mp4"
        slides_img_dir = "images"

        # Initialize reference video
        cap = cv2.VideoCapture(reference_video)
        ref_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Prepare directories
        
        refresh_dir(bg_vid_dir)
        refresh_dir(combined_videos)

        delete_folder_and_create(slides_img_dir)
        convert_to_images(pdf_file_path=slides_pdf, images_dir=slides_img_dir)
        ls=[]
        for i in os.listdir(avt_vid_dir):
            if i.endswith(".mp4"):
                ls.append(i)
        for idx, vid in enumerate(ls):
            if vid.endswith(".mp4"):
                img_source = vid[:8]+".png"
                print("img source: ", img_source)
                slide_number = int(img_source[5:8]) 
                print("slide number: ",slide_number)
                slide_duration = durations[idx]
                print("slide_duration: ",slide_duration)
                image_path = os.path.join(slides_img_dir,img_source)
                output_video_path = os.path.join(bg_vid_dir,vid)
                create_video_from_image(image_path=image_path,output_video_path=output_video_path,duration=slide_duration, fps = ref_fps)

        # @@@@@@@@@@@@@

        for avt_file in ls:
            if vid.endswith(".mp4"):
                avtr_file = os.path.join(avt_vid_dir, avt_file)
                save_frames(input_video_path=avtr_file, frames_store_dir=temp_frames_store, threshold=20)
                print("done with storing the frames: ", avtr_file)
                video_path = os.path.join(bg_vid_dir, avt_file)
                output_video_path = os.path.join(combined_videos, avt_file)
                audio_path = os.path.join(audios_dir,avt_file.split(".")[0]+".wav")
                print("audio path: ", avt_file)
                # overlay_images_on_video(video_path=video_path, frames_dir=temp_frames_store, output_video_path=output_video_path, overlay_height=avatar_height,audio_path=video_path, ref_fps=ref_fps)
                overlay_images_on_video(video_path=video_path, frames_dir=temp_frames_store, output_video_path=output_video_path, audio_path = audio_path, overlay_height= 380, ref_fps=30)
        #@@@@@@@@@@@@@@@@@@@@

        process_and_concat_videos(combined_videos_dir=combined_videos, output_file=final_video, seconds_to_remove=1)

        st.success("Final video created successfully!")
        st.video(final_video)





# # Run the app
# if __name__ == "__main__":
#     get_fin_video()

# #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# # Usage example
# slides_pdf = "processed_files/slides.pdf" # take input from ui and save it in this destination. If this destination alreasy exists, then delete and recreate that.
# audios_dir = "processed_files/audios" # This audios should be uploaded from a zip file. Take the zip file as input, and get all the audios from that directory and store it in this directory, if it does not exist create this dir. if already exists, delete and then create it and save.
# avt_vid_dir = "processed_files/avt_videos" # this also a zip. same as before one.
# reference_video = "processed_files/reference.mp4" # this also a video uploaded directly and stored in this location.
# durations, slide_durations = get_audio_durations(audios_dir)
# bg_vid_dir = "bg_vids"
# combined_videos = "fin_videos"
# temp_frames_store = "temp_frames_store"
# avatar_height = 380
# final_video = "final_video.mp4"
# slides_img_dir = "images"
# # @@@@@@@@@@@@@@@@@@@

# cap = cv2.VideoCapture(reference_video)
# ref_fps = cap.get(cv2.CAP_PROP_FPS)
# cap.release()

# refresh_dir(bg_vid_dir)
# refresh_dir(combined_videos)

# delete_folder_and_create(slides_img_dir)
# convert_to_images(pdf_file_path=slides_pdf, images_dir=slides_img_dir)

# for idx, vid in enumerate(os.listdir(avt_vid_dir)):
#     img_source = vid[:8]+".png"
#     print("img source: ", img_source)
#     slide_number = int(img_source[5:8]) 
#     print("slide number: ",slide_number)
#     slide_duration = durations[idx]
#     print("slide_duration: ",slide_duration)
#     image_path = os.path.join(slides_img_dir,img_source)
#     output_video_path = os.path.join(bg_vid_dir,vid)
#     create_video_from_image(image_path=image_path,output_video_path=output_video_path,duration=slide_duration, fps = ref_fps)

# # @@@@@@@@@@@@@

# for avt_file in os.listdir(avt_vid_dir):
#     avtr_file = os.path.join(avt_vid_dir, avt_file)
#     save_frames(input_video_path=avtr_file, frames_store_dir=temp_frames_store, threshold=20)
#     print("done with storing the frames: ", avtr_file)
#     video_path = os.path.join(bg_vid_dir, avt_file)
#     output_video_path = os.path.join(combined_videos, avt_file)
#     overlay_images_on_video(video_path=video_path, frames_dir=temp_frames_store, output_video_path=output_video_path, overlay_height=avatar_height, ref_fps=ref_fps)

# #@@@@@@@@@@@@@@@@@@@@

# process_and_concat_videos(combined_videos_dir=combined_videos, output_file=final_video, seconds_to_remove=1)

