import requests

# Set the URLs for the API endpoints
set_reference_video_url = "http://localhost:5000/set_reference_video"
process_audio_url = "http://localhost:5000/process_audio"

# Paths to the local files
reference_video_file_path = "path/to/your/reference_video.mp4"
audio_file_path = "path/to/your/audio_file.mp3"
output_video_file_path = "path/to/store/output_video.mp4"

# Step 1: Upload the reference video
with open(reference_video_file_path, 'rb') as ref_video_file:
    files = {'file': (reference_video_file_path, ref_video_file)}
    response = requests.post(set_reference_video_url, files=files)
    if response.status_code == 200:
        print("Reference video set successfully.")
    else:
        print(f"Error setting reference video: {response.json()}")

# Step 2: Upload the audio file and get the processed video
with open(audio_file_path, 'rb') as audio_file:
    files = {'file': (audio_file_path, audio_file)}
    response = requests.post(process_audio_url, files=files)
    if response.status_code == 200:
        # Save the received video file
        with open(output_video_file_path, 'wb') as output_file:
            output_file.write(response.content)
        print(f"Processed video saved as: {output_video_file_path}")
    else:
        print(f"Error processing audio: {response.json()}")
