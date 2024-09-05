import os
import shutil
import re
import time
import json
import requests
from dotenv import load_dotenv
import numpy as np
import PIL
from PIL import Image
import fitz  # PyMuPDF
from pptxtopdf import convert
import google.generativeai as genai
from moviepy.editor import *
import mediapipe as mp
import cv2
from pydub import AudioSegment
from moviepy.editor import concatenate_audioclips, AudioFileClip
import os

from tqdm import tqdm
import azure.cognitiveservices.speech as speechsdk
from moviepy.editor import concatenate_audioclips, AudioFileClip

if not load_dotenv('./.env'): raise Exception("env file not found")

SERVICE_REGION = os.getenv("SPEECH_SERVICEE_REGION")
SUBSCRIPTION_KEY = os.getenv("SPEECH_SERVICE_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

url_base = f"https://{SERVICE_REGION}.customvoice.api.speech.microsoft.com/api"
genai.configure(api_key=GEMINI_API_KEY)

MODEL = "gemini-1.5-flash-latest" # CAN ALSO USE: "gemini-1.5-pro-latest" - BUT WILL BE A BIT SLOW.
model = genai.GenerativeModel(MODEL)




# WILL BE UPDATED IN APP ##########################################################################################
input_ppt = "output.pptx" # Make sure it in in the current working directory (Example: file_name.pptx)
# holds content
image_names = []
slides_content = []
# directories
images_dir = "images"
avtr_video_dir = "avt_videos"
# output
frames_video_file = "main.mp4"
avtr_video_file = "avtr_vid.webm"
final_video_file = "final.webm"
AUDIO_FILES = "audios"
# WILL BE UPDATED IN APP END #######################################################################################


# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def text_to_speech(text, output_file):
    speech_key = SUBSCRIPTION_KEY # Replace with your Speech service key
    service_region = SERVICE_REGION   # Replace with your Speech service region
    
    # Create a speech configuration object
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_voice_name = 'en-US-AndrewNeural'
    
    # Create a speech synthesizer object with a file output
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # Synthesize the text and save it to the file
    result = synthesizer.speak_text_async(text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f'Text-to-speech conversion successful. Audio saved to {output_file}')
    else:
        print(f'Text-to-speech conversion failed: {result.reason}')

def delete_folder_and_create(path):
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path)

def convert_to_images(pdf_file_path, images_dir = "images"):
    if os.path.exists(images_dir): shutil.rmtree(images_dir)
    os.makedirs(images_dir)
    pdf_document = fitz.open(pdf_file_path)
    num_pages = len(pdf_document)
    num_digits = len(str(num_pages))
    for page_num in range(num_pages):
        page_number_str = str(page_num + 1).zfill(num_digits)
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_path = f"{images_dir}/slide{page_num+1:03}.png"
        image_names.append(f"slide{page_num+1:03}")
        img.save(img_path)
        print(f"Saved {img_path}")

def download_file(url, local_path, index):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        # filename = url.split("/")[-1].split("?")[0]
        filename = image_names[index]+".webm"
        local_filename = os.path.join(local_path, filename)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk: f.write(chunk)
    return local_filename

def preprocess_response(text):
    text = text.replace('*', '')
    return text

def combine_webm_videos(output_file):
    clips = []
    for file in os.listdir(avtr_video_dir):
        path = os.path.join(avtr_video_dir,file)
        clips.append(VideoFileClip(path))
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_file, codec="libvpx", audio_codec="libvorbis")
    final_clip.close()
    for clip in clips: clip.close()

def get_video_duration(file_path):
    video = VideoFileClip(file_path)
    duration = video.duration
    video.close()
    return duration

def custom_resize(clip, newsize):
    def resize_frame(frame):
        img = Image.fromarray(frame)
        return np.array(img.resize(newsize, Image.LANCZOS))
    return clip.fl_image(resize_frame)

def get_slides_content():
    slides_content = []
    prompt_template = """
    You are a professional presenter explaining a PowerPoint presentation to an audience. Analyze the current slide image and provide a concise, insightful explanation. Focus on the key points and main ideas rather than reading any text verbatim from the slide. Your explanation should be brief but informative, suitable for a quick presentation.

    Guidelines:
    1. Do not use greetings or introductions.
    2. Avoid reading sentences directly from the slide.
    3. Provide context and insights beyond what's visibly written.
    4. Keep the explanation concise, aiming for about 2-3 sentences per slide.
    5. Use plain text without any special formatting or markdown.
    6. If this isn't the first slide, ensure your explanation flows naturally from the previous content.
    7. Use examples other than the one given in the ppt.
    8. Provide additional details when and where required.
    9. Connect the contents from previous slides also.
    10. Be as concise as possible.
    11. Focus more on explaining images in each slide which may be flowcharts or images provided that they are technical images with information related to the topic.

    Current slide number: {slide_number}

    Previous slides' content (if any):
    {previous_slides_content}

    Explain this slide succinctly:
    """

    for idx, slide in enumerate(os.listdir(images_dir)):
        print(f"Slide: {idx+1}")
        img = PIL.Image.open(os.path.join("images", slide))

        previous_slides_content = ""
        if len(slides_content) > 0:
            previous_slides_content = " ".join([f"Slide-{j+1}: {text}" for j, text in enumerate(slides_content)])

        prompt = prompt_template.format(
            slide_number=idx+1,
            previous_slides_content=previous_slides_content
        )
        # print(prompt)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")

        response = model.generate_content([prompt, img], stream=True)
        response.resolve()
        description = preprocess_response(response.text)
        slides_content.append(description)
        # print(description)
    return slides_content

# Initialize the list that will hold the structured data
def get_sentence_wise_slide_content(slides_content):
    sent_slide_content = []

    # Iterate over each slide's text
    for i, slide_text in enumerate(slides_content):
        # Split the slide text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', slide_text.strip())
        # print("sentences: ", sentences)
        trimmed_sentences = []
        for sentence in sentences:
            trimmed_sentences.append(" ".join(sentence.split()))
        # Create a dictionary for the current slide
        slide_dict = {
            "slide_number": i + 1,
            "sentences": trimmed_sentences
        }

        # Add the dictionary to the slides list
        sent_slide_content.append(slide_dict)
    return sent_slide_content


def text_to_speech_sentence_wise(sent_slide_content):
    for slide in sent_slide_content:
        slide_number = slide['slide_number']
        for sentence_number, sentence in enumerate(slide['sentences'], start=1):
            # Generate the file name in the format: "slideXXX_sentenceYYY.wav"
            file_name = f"slide{slide_number:03}_sentence{sentence_number:03}.wav"
            output_file_path = os.path.join(AUDIO_FILES, file_name)
            
            # Assuming text_to_speech is your function that generates the audio.
            text_to_speech(sentence, output_file_path)

def combine_mp3_files(directory, output_file):
    # List all MP3 or WAV files in the directory
    files = sorted([f for f in os.listdir(directory) if f.endswith(('.mp3','.wav'))])
    print("Files: ", files)
    
    # Initialize lists for clips, durations, and slide durations
    clips = []
    durations = []
    slide_durations = []

    current_slide = None
    current_slide_duration = 0

    # 1 second of silence
    silence = AudioSegment.silent(duration=1000)  # 1000 ms = 1 second

    for file in files:
        file_path = os.path.join(directory, file)
        
        # Load the audio file using pydub
        audio = AudioSegment.from_file(file_path)
        
        # Add 1 second of silence after the audio
        audio_with_silence = audio + silence
        
        # Export the modified audio to a temporary WAV file
        temp_file_path = file_path
        audio_with_silence.export(temp_file_path, format="wav")
        
        # Load the temporary WAV file with moviepy
        audio_clip = AudioFileClip(temp_file_path)
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

    # Concatenate all audio clips
    final_clip = concatenate_audioclips(clips)
    
    # Write the output file
    final_clip.write_audiofile(output_file)
    
    return durations, slide_durations


def create_video_from_images(images_dir, slide_durations, frames_video_file, avtr_video_dir=None):
    """
    Creates a video from images, matching each image to a specified duration.

    Args:
    - images_dir (str): Directory containing the image files.
    - slide_durations (list): List of durations for each image.
    - frames_video_file (str): Path to save the final video file.
    - avtr_video_dir (str, optional): Directory containing avatar video files, used to match frame durations (not used here).

    Returns:
    - None: Saves the final video file to the specified path.
    """
    # Get a sorted list of image files ending with '.png'
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    
    # Check if the number of images matches the number of slide durations
    if len(image_files) != len(slide_durations):
        raise ValueError("Number of images and slide durations do not match.")
    
    # Initialize an empty list to hold the image clips
    clips = []
    
    # Create image clips with specified durations
    for idx, img_file in enumerate(image_files):
        image_duration = slide_durations[idx]  # Duration for the current image
        img_clip = ImageClip(os.path.join(images_dir, img_file), duration=image_duration)
        clips.append(img_clip)
    
    # Concatenate all image clips into one final video
    final_clip = concatenate_videoclips(clips)
    
    # Write the final video to a file with the specified codec and frame rate
    final_clip.write_videofile(frames_video_file, codec="libx264", fps=29.97)

def create_video_from_image(image_path, output_video_path, duration, fps=30):
    image_clip = ImageClip(image_path)
    video_clip = image_clip.set_duration(duration)
    video_clip = video_clip.set_fps(fps)
    video_clip.write_videofile(output_video_path, codec="libx264")


# Function to remove pixels where green value is greater than both red and blue values by a specified threshold
def remove_pixels_based_on_green(image, threshold=5):
    # Convert image to RGBA
    image = image.convert("RGBA")
    data = image.getdata()

    new_data = []
    for item in data:
        r, g, b, a = item
        if (g - r >= threshold) and (g - b >= threshold):
            new_data.append((0, 0, 0, 0))  # Make pixel transparent
        else:
            new_data.append(item)

    image.putdata(new_data)
    return image

def save_frames(input_video_path, threshold = 5):
    if os.path.exists("frames"):
        shutil.rmtree("frames")
        os.mkdir("frames")
    else: os.mkdir("frames")
    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)

    # Create output directory if it doesn't exist
    output_dir = 'frames'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
    # Read the first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    height, width, _ = frame.shape
    background = np.zeros((height, width, 3), dtype=np.uint8)  # Black background

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process each frame
    frame_count = 0
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = segmentation.process(RGB)
        mask = results.segmentation_mask

        # Create a binary mask for the foreground (foreground is where mask > 0.6)
        mask = np.expand_dims(mask > 0.6, axis=-1)
        mask = np.repeat(mask, 3, axis=-1)  # Convert to 3-channel

        # Create the output frame with an alpha channel
        output = np.zeros((height, width, 4), dtype=np.uint8)
        output[:, :, :3] = np.where(mask, frame, background)  # Apply mask to frame
        output[:, :, 3] = np.where(mask[:, :, 0], 255, 0)    # Set alpha channel based on mask

        # Convert to PIL Image to process transparency and apply blur
        pil_image = Image.fromarray(output, 'RGBA')
        # pil_image = apply_gaussian_blur(pil_image)  # Apply Gaussian blur
        pil_image = remove_pixels_based_on_green(pil_image, threshold)  # Remove pixels based on green value
        output = np.array(pil_image)

        # Save each frame as a PNG file
        filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(filename, output)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def overlay_images_on_video(video_path, images_dir, output_video_path, audio_path, overlay_height):
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
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
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
        overlay_img_path = os.path.join(images_dir, image_files[frame_count % num_images])
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
    clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in output_frames], fps=fps)
    
    # Add audio to the video
    video_clip = VideoFileClip(video_path)
    audio_clip = VideoFileClip(audio_path).audio
    clip = clip.set_audio(audio_clip)

    # Write the final video file
    clip.write_videofile(output_video_path, codec='libx264')


def process_and_concat_videos(directory, output_file, seconds_to_remove=0):
    # List all video files in the directory
    files = sorted([f for f in os.listdir(directory) if f.endswith('.mp4')])
    clips = []

    for file in files:
        file_path = os.path.join(directory, file)
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
