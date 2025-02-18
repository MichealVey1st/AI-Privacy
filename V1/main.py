import cv2
import numpy as np
from tqdm import tqdm

# Load YOLO model
model_cfg = 'yolov3-face.cfg'  # Path to config file
model_weights = 'yolov3-wider_16000.weights'  # Path to weights file

net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)  # Use default backend
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU

# Load video
video_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties for saving output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = 'output_video.mp4'  # MP4 output format

# Define the codec and create VideoWriter object (use H264 for MP4 output)
fourcc = cv2.VideoWriter_fourcc(*'H264')  # H264 for .mp4 output
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Get total number of frames to initialize tqdm
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Function to process each frame
def process_frame(frame):
    # Resize the frame for faster processing (optional)
    frame_resized = cv2.resize(frame, (640, 360))  # Resize to smaller resolution

    # Prepare input for YOLO
    height, width = frame_resized.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_resized, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run detection
    output_layers = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for detection in detections:
        for detect in detection:
            scores = detect[5:]  # YOLO score outputs start from index 5
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detect[0:4] * np.array([width, height, width, height])
                (centerX, centerY, boxW, boxH) = box.astype("int")

                # Expand the box
                boxW = int(boxW * 1.2)
                boxH = int(boxH * 1.2)
                x = int(centerX - (boxW / 2))
                y = int(centerY - (boxH / 2))
                y = max(0, y - int(boxH * 0.1))  # Ensure box stays within bounds

                # Append the box, confidence, and class id for NMS
                boxes.append([x, y, boxW, boxH])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Only keep the best box from NMS
    if len(indices) > 0:
        # Get the box with the highest confidence
        i = indices.flatten()[0]  # Get the first (highest confidence) box
        x, y, boxW, boxH = boxes[i]

        # Extract ROI and apply blur
        face_roi = frame_resized[y:y+boxH, x:x+boxW]
        if face_roi.size > 0:  # Check to avoid errors on invalid ROI
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)  # Apply Gaussian Blur
            frame_resized[y:y+boxH, x:x+boxW] = blurred_face  # Replace ROI with blurred face

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
