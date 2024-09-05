
from pydub import AudioSegment
from moviepy.editor import concatenate_audioclips, AudioFileClip
import os
import concurrent.futures
import asyncio
import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
from PIL import Image
import mediapipe as mp
from moviepy.editor import ImageSequenceClip, VideoFileClip
from moviepy.editor import ImageClip
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

def refresh_dir(directory_name):
    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
    os.mkdir(directory_name)


def get_audio_durations(directory):
    # List all MP3 or WAV files in the directory
    files = sorted([f for f in os.listdir(directory) if f.endswith(('.mp3','.wav'))])
    print("Files: ", files)
    
    # Initialize lists for clips, durations, and slide durations
    clips = []
    durations = []
    slide_durations = []

    current_slide = None
    current_slide_duration = 0

    for file in files:
        file_path = os.path.join(directory, file)
        
        # Load the audio file using pydub
        audio_clip = AudioFileClip(file_path)
        clips.append(audio_clip)
        
        # Calculate duration and add to the list
        duration = audio_clip.duration
        durations.append(duration)
        
        # Extract slide number from the file name
        slide_number = int(file.split('_')[0][5:])

        if current_slide is None:
            current_slide = slide_number

        if slide_number == current_slide:
            current_slide_duration += duration
        else:
            slide_durations.append(current_slide_duration)
            current_slide_duration = duration
            current_slide = slide_number
    
    # Append the duration of the last slide
    slide_durations.append(current_slide_duration)
    
    return durations, slide_durations


def remove_pixels_based_on_green(image, threshold=5):
    image = image.convert("RGBA")
    data = image.getdata()
    new_data = []
    for item in data:
        r, g, b, a = item
        if (g - r >= threshold) and (g - b >= threshold):
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)
    image.putdata(new_data)
    return image

def save_frames(input_video_path, frames_store_dir, threshold=40):
    if os.path.exists(frames_store_dir):
        shutil.rmtree(frames_store_dir)
        os.mkdir(frames_store_dir)
    else:
        os.mkdir(frames_store_dir)
    
    cap = cv2.VideoCapture(input_video_path)
    output_dir = frames_store_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read video {input_video_path}")
        return
    
    height, width, _ = frame.shape
    background = np.zeros((height, width, 3), dtype=np.uint8)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL image for processing
        pil_image = Image.fromarray(frame)
        
        # Remove green pixels based on the threshold
        pil_image = remove_pixels_based_on_green(pil_image, threshold)
        
        # Convert back to numpy array
        output = np.array(pil_image)

        filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(filename, output)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def create_video_from_image(image_path, output_video_path, duration, fps=30):
    """
    Creates a video using a single image for a specific duration.

    :param image_path: Path to the input image file.
    :param output_video_path: Path where the output video will be saved.
    :param duration: Duration of the video in seconds.
    :param fps: Frames per second for the video. Default is 30.
    """
    # Load the image
    image_clip = ImageClip(image_path)

    # Set the duration of the video
    video_clip = image_clip.set_duration(duration)

    # Set the FPS (frames per second) of the video
    video_clip = video_clip.set_fps(fps)

    # Write the result to a video file
    video_clip.write_videofile(output_video_path, codec="libx264")

def overlay_images_on_video(video_path, frames_dir, output_video_path, audio_path, overlay_height,ref_fps=30):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # List all image files in the directory and sort them
    image_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    num_images = len(image_files)
    
    if num_images == 0:
        print("Error: No images found in the directory.")
        return
    
    # Create a list to store output frames
    output_frames = []

    # Calculate the width of the overlay image maintaining its aspect ratio
    overlay_width = int((overlay_height / height) * width)

    # Process each frame
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the start

    for _ in tqdm(range(total_frames), desc="Processing Frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Determine the image to overlay
        overlay_img_path = os.path.join(frames_dir, image_files[frame_count % num_images])
        overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
        
        # Resize the overlay image to the desired dimensions while maintaining aspect ratio
        if overlay_img.shape[0] != overlay_height:
            aspect_ratio = overlay_img.shape[1] / overlay_img.shape[0]
            new_width = int(overlay_height * aspect_ratio)
            overlay_img = cv2.resize(overlay_img, (new_width, overlay_height))
        
        # Split the overlay image into color and alpha channels
        overlay_color = overlay_img[:, :, :3]
        alpha_channel = overlay_img[:, :, 3] / 255.0
        
        # Create the output frame with an alpha channel
        frame_with_overlay = frame.copy()
        overlay_x = (width - overlay_img.shape[1]) // 2  # Center horizontally
        overlay_y = height - overlay_img.shape[0]  # Align bottom
        
        for c in range(3):
            frame_with_overlay[overlay_y:overlay_y + overlay_img.shape[0], overlay_x:overlay_x + overlay_img.shape[1], c] = \
                (alpha_channel * overlay_color[:, :, c] + (1 - alpha_channel) * frame_with_overlay[overlay_y:overlay_y + overlay_img.shape[0], overlay_x:overlay_x + overlay_img.shape[1], c])
        
        output_frames.append(frame_with_overlay)
        frame_count += 1
    
    cap.release()

    # Create video from frames
    clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in output_frames], fps=ref_fps)
    
    # Add audio to the video
    audio_clip = AudioFileClip(audio_path)

    # Set the audio to the video clip
    clip = clip.set_audio(audio_clip)

    # Write the final video file
    clip.write_videofile(output_video_path, codec='libx264')


def process_and_concat_videos(combined_videos_dir, output_file, seconds_to_remove=0):
    # List all video files in the directory
    files = sorted([f for f in os.listdir(combined_videos_dir) if f.endswith('.mp4')])
    clips = []

    for file in files:
        file_path = os.path.join(combined_videos_dir, file)
        clip = VideoFileClip(file_path)

        # Remove last `seconds_to_remove` seconds if greater than 0
        if seconds_to_remove > 0:
            duration = clip.duration
            if duration > seconds_to_remove:
                clip = clip.subclip(0, duration - seconds_to_remove)
            else:
                print(f"Video '{file}' is shorter than the specified removal time. Skipping.")
                continue

        clips.append(clip)

    # Concatenate all video clips
    final_clip = concatenate_videoclips(clips, method='compose')

    # Write the output file
    final_clip.write_videofile(output_file, codec='libx264')

