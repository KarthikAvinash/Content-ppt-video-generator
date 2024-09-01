from flask import Flask, request, send_file, jsonify
import os
import subprocess
import tempfile

app = Flask(__name__)

# Paths
reference_video_path = None  # Placeholder to store the path of the reference video
output_directory = "results"
os.makedirs(output_directory, exist_ok=True)

@app.route('/set_reference_video', methods=['POST'])
def set_reference_video():
    global reference_video_path
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith(('.mp4', '.mov', '.avi')):
        # Save the reference video in a temporary directory
        reference_video_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(reference_video_path)
        return jsonify({"message": "Reference video set successfully"}), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/process_audio', methods=['POST'])
def process_audio():
    global reference_video_path
    if reference_video_path is None:
        return jsonify({"error": "Reference video is not set"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.mp3'):
        # Save the audio file in a temporary location
        input_audio_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(input_audio_path)
        
        # Prepare the output file path
        base_output_filename = os.path.splitext(file.filename)[0] + ".mp4"
        output_file_path = os.path.join(output_directory, base_output_filename)
        
        # Check if the file already exists, and find a new name if it does
        counter = 1
        while os.path.exists(output_file_path):
            new_filename = f"{os.path.splitext(base_output_filename)[0]}_{counter}.mp4"
            output_file_path = os.path.join(output_directory, new_filename)
            counter += 1
        
        # Run the inference script with the correct paths
        command = f"python3 inference.py --face {reference_video_path} --audio {input_audio_path} --outfile {output_file_path}"
        subprocess.run(command, shell=True)
        
        # Return the processed video file
        return send_file(output_file_path, as_attachment=True)
    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
