# Import necessary libraries for video processing, argument parsing, numerical operations, and system utilities
import cv2  # OpenCV for image and video processing
import argparse  # For parsing command-line arguments
import numpy as np  # For numerical operations, especially with arrays
from tqdm import tqdm  # For displaying progress bars
import os  # For file and directory operations
import subprocess  # For running external commands
import ffmpeg  # For video and audio processing

# Initialize a global dictionary to track face positions across frames
tracked_faces = {}

# Function to blur faces in a video
def blur_faces(video_path, output_path):
    # Load YOLO model configuration and weights for face detection
    model_cfg = 'yolov3-face.cfg'
    model_weights = 'yolov3-wider_16000.weights'
    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)  # Load the model into memory
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)  # Use the default backend for computation
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU for inference

    # Open the input video file
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video frame width
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video frame height
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video frame rate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Set codec for output video

    # Set paths for temporary video files (used for merging video and audio later)
    temp_output_path = os.path.abspath("./temp/temp_video.mp4")
    audio_temp_output_path = os.path.abspath("./temp/audio_temp_video.mp4")

    # Initialize VideoWriter to write processed frames into a new video file
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames in the video

    # Inner function to process each video frame
    def process_frame(frame, frame_index):
        global tracked_faces  # Use the global dictionary for tracking faces

        # Resize the frame to reduce processing time and normalize dimensions
        frame_resized = cv2.resize(frame, (640, 360))
        height, width = frame_resized.shape[:2]  # Get dimensions of the resized frame

        # Preprocess the frame for YOLO input
        blob = cv2.dnn.blobFromImage(frame_resized, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)  # Pass the preprocessed frame to the network
        output_layers = net.getUnconnectedOutLayersNames()  # Get output layer names for inference
        detections = net.forward(output_layers)  # Perform forward pass to detect objects

        # Initialize lists to store bounding boxes and confidence scores
        boxes = []
        confidences = []

        # Process detections to extract bounding boxes and confidence scores
        for detection in detections:
            for detect in detection:
                scores = detect[5:]  # Get confidence scores for all classes
                confidence = np.max(scores)  # Take the maximum score as confidence
                if confidence > 0.4:  # Filter detections with confidence > 0.4
                    # Scale bounding box coordinates back to the resized frame
                    box = detect[:4] * np.array([width, height, width, height])
                    (centerX, centerY, boxW, boxH) = box.astype("int")
                    x = int(centerX - (boxW / 2))
                    y = int(centerY - (boxH / 2))
                    x, y, boxW, boxH = max(0, x), max(0, y), int(boxW), int(boxH)
                    boxes.append([x, y, boxW, boxH])  # Append the bounding box
                    confidences.append(float(confidence))  # Append the confidence score

        # Apply non-maximum suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        current_faces = []  # List to store faces detected in the current frame

        # Extract bounding boxes for retained detections
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, boxW, boxH = boxes[i]
                current_faces.append((x, y, boxW, boxH))

        # Update the global face tracking dictionary with new or extrapolated positions
        new_tracked_faces = {}
        for i, (x, y, boxW, boxH) in enumerate(current_faces):
            new_tracked_faces[i] = (x, y, boxW, boxH, 0)  # Reset the age of the face to 0

        # Retain faces from the previous frame if they were not detected in the current frame
        for face_id, (x, y, boxW, boxH, age) in tracked_faces.items():
            if face_id not in new_tracked_faces:
                if age < 3:  # Allow tracking for up to 3 frames without detection
                    new_tracked_faces[face_id] = (x, y, boxW, boxH, age + 1)

        tracked_faces = new_tracked_faces  # Update the global dictionary

        # Blur all tracked faces
        for (x, y, boxW, boxH, age) in tracked_faces.values():
            x = max(0, x - 10)  # Expand the bounding box slightly
            y = max(0, y - 10)
            boxW = min(width, boxW + 20)
            boxH = min(height, boxH + 20)
            face_roi = frame_resized[y:y+boxH, x:x+boxW]  # Extract the face region
            if face_roi.size > 0:
                blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)  # Apply Gaussian blur
                frame_resized[y:y+boxH, x:x+boxW] = blurred_face  # Replace the face with the blurred version

        # Resize the frame back to its original dimensions
        frame_resized = cv2.resize(frame_resized, (frame_width, frame_height))
        return frame_resized  # Return the processed frame

    # Loop through all frames in the video
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        for frame_index in range(total_frames):
            ret, frame = cap.read()  # Read the next frame
            if not ret:  # Break if no frame is returned
                break
            processed_frame = process_frame(frame, frame_index)  # Process the frame
            out.write(processed_frame)  # Write the processed frame to the output video
            pbar.update(1)  # Update the progress bar

    # Release video resources after processing
    cap.release()
    out.release()

    # Extract audio from the input video using FFmpeg
    try:
        ffmpeg.input(video_path).output(audio_temp_output_path, map='0:a', acodec='aac', strict='experimental').run(overwrite_output=True)
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return

    # Merge the processed video and extracted audio using FFmpeg
    try:
        command = [
            "ffmpeg",
            "-i", temp_output_path,  # Input video (processed)
            "-i", audio_temp_output_path,  # Input audio
            "-c:v", "libx264",  # Re-encode video using H.264 codec
            "-c:a", "aac",  # Re-encode audio using AAC codec
            "-strict", "experimental",  # Ensure compatibility with older FFmpeg versions
            "-shortest",  # Match the duration of the shortest stream
            output_path  # Path for the final output video
        ]
        subprocess.run(command, check=True)  # Execute the FFmpeg command
        print(f"Video saved with audio to {output_path}")
    except Exception as e:
        print(f"Error during video and audio merging: {e}")
        return

def modify_voices(video_path, output_path):
    # TODO: Implement voice detection and modification
    print(f"Modifying voices in {video_path}")
    # Save the modified video to output_path
    pass

def blur_signs(video_path, output_path):
    # TODO: Implement object detection for signs (street names, license plates, etc.)
    print(f"Blurring signs in {video_path}")
    # Save the modified video to output_path
    pass

def overwrite_metadata(video_path, output_path):
    # TODO: Overwrite metadata with incorrect/fake information
    print(f"Overwriting metadata for {video_path}")
    # Save the modified video to output_path
    pass

def process_video(video_path, output_path, args):
    """
    Process a single video file based on command-line arguments.
    """
    if args.blur_faces:
        blur_faces(video_path, output_path)
    if args.modify_voices:
        modify_voices(video_path, output_path)
    if args.blur_signs:
        blur_signs(video_path, output_path)
    if args.overwrite_metadata:
        overwrite_metadata(video_path, output_path)

def batch_process(input_folder, output_folder, args):
    """
    Process all videos in the input folder and save results to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))]
    with tqdm(total=len(video_files), desc="Processing Videos", unit="file") as pbar:
        for filename in video_files:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"modified_{filename}")
            
            try:
                process_video(input_path, output_path, args)
                print(f"Saved modified video to {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            finally:
                pbar.update(1)

def main():
    # Set up argument parsing
    '''
        Flags and Options:
        -i or --input: Specify the input folder. Defaults to input/.
        -o or --output: Specify the output folder. Defaults to output/.
        -f or --blur-faces: Enable face blurring.
        -d or --modify-voices: Enable voice modification.
        -s or --blur-signs: Enable sign blurring.
        -m or --overwrite-metadata: Enable metadata overwriting.

        main.py -i input -o output -f -d -s -m
    '''

    parser = argparse.ArgumentParser(description="Auto Privacy Script for Video Processing")
    parser.add_argument("-i", "--input", default="input", help="Input folder containing videos")
    parser.add_argument("-o", "--output", default="output", help="Output folder for processed videos")
    parser.add_argument("-f", "--blur-faces", action="store_true", help="Enable face blurring")
    parser.add_argument("-d", "--modify-voices", action="store_true", help="Enable voice modification")
    parser.add_argument("-s", "--blur-signs", action="store_true", help="Enable sign/license plate blurring")
    parser.add_argument("-m", "--overwrite-metadata", action="store_true", help="Enable metadata overwriting")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist.")
        return

    # Call batch process with the provided arguments
    batch_process(args.input, args.output, args)

if __name__ == "__main__":
    main()
