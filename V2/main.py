import argparse
import os
from tqdm import tqdm
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def blur_faces(video_path, output_path):
    """
    Detects and blurs faces in a video.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video with blurred faces.
    """
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Convert the frame to RGB (MediaPipe expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                # Extract bounding box information
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * width)
                y = int(bboxC.ymin * height)
                w = int(bboxC.width * width)
                h = int(bboxC.height * height)

                # Ensure the bounding box stays within frame dimensions
                x, y, w, h = max(0, x), max(0, y), min(w, width - x), min(h, height - y)

                # Blur the region of interest (face area)
                roi = frame[y:y+h, x:x+w]
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
                frame[y:y+h, x:x+w] = blurred_roi

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    face_detection.close()
    print(f"Video with blurred faces saved to: {output_path}")

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
