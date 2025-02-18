import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os
from tqdm import tqdm
import cv2

last_detected_faces = [] # Global variable to store the last detected faces

def blur_faces(video_path, output_path):
    # Load YOLO model
    model_cfg = 'yolov3-face.cfg'  # Path to config file
    model_weights = 'yolov3-wider_16000.weights'  # Path to weights file

    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)  # Use default backend
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU

    # Load video
    cap = cv2.VideoCapture(video_path)

    # Get video properties for saving output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object (use H264 for MP4 output)
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # H264 for .mp4 output
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Get total number of frames to initialize tqdm
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Function to process each frame
    # Initialize a global list to store the last detected bounding boxes

    def process_frame(frame):
        global last_detected_faces

        # Resize the frame for faster processing (optional)
        frame_resized = cv2.resize(frame, (640, 360))
        height, width = frame_resized.shape[:2]

        # Prepare input for YOLO
        blob = cv2.dnn.blobFromImage(frame_resized, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers = net.getUnconnectedOutLayersNames()
        detections = net.forward(output_layers)

        boxes = []
        confidences = []
        for detection in detections:
            for detect in detection:
                scores = detect[5:]
                confidence = np.max(scores)
                if confidence > 0.5:
                    # Extract bounding box
                    box = detect[:4] * np.array([width, height, width, height])
                    (centerX, centerY, boxW, boxH) = box.astype("int")
                    x = int(centerX - (boxW / 2))
                    y = int(centerY - (boxH / 2))
                    x, y, boxW, boxH = max(0, x), max(0, y), int(boxW), int(boxH)
                    boxes.append([x, y, boxW, boxH])
                    confidences.append(float(confidence))

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        current_faces = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, boxW, boxH = boxes[i]
                current_faces.append((x, y, boxW, boxH))

        # Merge current and last detected faces
        if not current_faces:
            current_faces = last_detected_faces  # Use the last known faces if none are detected
        else:
            last_detected_faces = current_faces  # Update the last detected faces

        # Blur all current faces
        for (x, y, boxW, boxH) in current_faces:
            # Expand the box a little for robustness
            x = max(0, x - 10)
            y = max(0, y - 10)
            boxW = min(width, boxW + 20)
            boxH = min(height, boxH + 20)
            face_roi = frame_resized[y:y+boxH, x:x+boxW]
            if face_roi.size > 0:
                blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                frame_resized[y:y+boxH, x:x+boxW] = blurred_face

        # Resize back to original resolution
        frame_resized = cv2.resize(frame_resized, (frame_width, frame_height))
        return frame_resized


    # Create the progress bar using tqdm
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame in a separate function
            processed_frame = process_frame(frame)

            # Write the processed frame to the output video
            out.write(processed_frame)

            # Update progress bar
            pbar.update(1)

    cap.release()
    out.release()  # Save the output video
    cv2.destroyAllWindows()

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
