# AI Privacy Project

Welcome to the AI Privacy Project! This project aims to automate the protection of personal privacy using AI techniques.

## Features

- **Face Detection**: Utilizes YOLOv3 for accurate face detection (using one of the most well known training called WIDER).
- **Voice Distortion**: Implements a Voice Activity Detector (VAD) to distort voices.
- ~~**Metadata Removal**: Working on removing all overhead metadata.~~
- ~~**Blurring Sensitive Information**: Plans to blur sensitive information such as street signs, house numbers, ID cards, etc.~~

## Current Work

- **Compiling OpenCV with CUDA**: Currently solving issues related to compiling OpenCV with CUDA.
- **Training YOLOv11**: Working on training a custom YOLOv11 system to detect and blur additional sensitive items.

## Structure

This is set up in a way to reflect the iterations of my program with the latest being v6. Each has a unique description of what was changed in that iteration. 

_NOTE: please note that this repo is missing the .cfg, .weights, and outputs for each versions due to githubs upload limits. You can find the cfg and weights below, for the outputs you will need to run the program to get those._

- [.cfg] (https://raw.githubusercontent.com/sthanhng/yoloface/refs/heads/master/cfg/yolov3-face.cfg)
- [.weights] (https://drive.google.com/file/d/1xYasjU52whXMLT5MtF7RCPQkV66993oR/view?usp=sharing) 

## Future Plans

- Enhance the accuracy and efficiency of the detection and blurring systems.
- Integrate more privacy protection features.

Thank you for your interest in the AI Privacy Project. Contributions and feedback are welcome!
