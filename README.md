# Event Driven Pedestrian Vehicle Detection with OpenCV and Yolo

## Abstract: 
Most of the deep neural object detection/classification models such as R-CNN, Faster R-CNN & SSD require significantly higher computational power to achieve respectable FPS & Accuracy. But, we need a different cnn-based neural architecture to cater the scenarios where the object detection task is to be executed in real-time rather than waiting for hours to process a 10 min video stream or to run the detector on limited/low-power hardware. Thus, in this project, we implement a real-time event-driven Vehicle Detection & Counting system using the OpenCV's Yolo v3 (608 x 608) implementation. The model detects and counts pedestrians/vehicles that cross a line of interest projected in any axis within the video stream.

## Implemented Tasks:

1. Experimented & contrasted different YOLO v3 configurations.
2. Optimized Object Detection & Classification for large video streams.
3. Implemented vehicle density counter at per-frame/video level.
4. Developed efficient frame-level per-class object counter.
5. Designed Event-based object counter to detect incoming & outgoing vehicles in different lanes.
6. Refactored and optimized project code. 
7. Envisioned an End-to-End system by implementing a real-time object detection progress bar.
8. Set-up, executed & analysed the model on Google Cloud Platform and verified the overall model accuracies/fps.
9. Sped-up OpenCV‘s video writer output by utilizing different video extensions.
10. Fixed FFmpeg dependency issues in Ubuntu Docker container.

## Instructions:

### Execute in Windows/Linux:

1. Install Python 3
2. Install PIP
3. Install project dependencies: "pip install -r Requirement.txt"
4. Download Yolo v3 Weights (608x608) from this [**link**](https://drive.google.com/drive/folders/1jFs9NSD_kiRR7wzLuC6o-IzBjzq9h0jW?usp=sharing). Could not store them in GitHub repo because of file size restrictions!
5. Save the "yolov3.weights" (downloaded in the above step) file inside the "yolo-coco" folder within the project directory.
6. Download a sample input video from the [**link**](https://drive.google.com/file/d/1k9mTMGVxDpLqlqr4T7mci-viInS5Pe_M/view?usp=sharing)
7. Save the "overpass.mp4" file inside the videos folder within the project directory.
8. Run the object detector model inside the project directory: "python Yolo_Detector.py -i videos/overpass.mp4 -o output/final_output.avi


### Execute in Docker Container:
**Information:** The final project docker image is publicly hosted in the Docker Hub portal: https://hub.docker.com/r/poojiyengar5/computer_vision

#### For Beginner Users:
[Who just wants to run the Object Detector once and see the results - cannot modify or view the code, so don't use this command if you want to continue the project development in the future!]

- **Command**: docker run --rm --name Akshay_Container -ti poojiyengar5/computer_vision:latest python3 HIS.py -i videos/overpass.mp4 -o output/final_output.avi

#### For Advanced Users:
[For further development or modification of the project code]

- **Please follow the following steps to run the project:**
	- **Step 1:** Container creation
		* **Command:** docker run --name Akshay_Container -ti poojiyengar5/computer_vision:latest

	- **Step 2:** Start the container
		* **Command:** docker start Akshay_Container

	- **Step 3:** Log-in to the container
		* **Command:** docker exec -ti Akshay_Container /bin/bash

	- **Step 4:** Run the Object Detector
		* **Command:** python3 HIS.py -i videos/overpass.mp4 -o output/test.avi

	- **Step 5:** Modify the Object Detector Code
		* **Command:** vi HIS.py

	- **Step 6:** Exit from the container
		* **Command:** exit

	- **Step 7:** Stop the container
		* **Command:** docker stop Akshay_Container

	- **Step 8:** Delete the container
		* **Command:** docker rm Akshay_Container


## Future Works:

1. Post object detection generate detailed vehicle information reports for further statistical analysis.
2. Exploit pre-trained YOLO models from TensorFlow/PyTorch instead of OpenCV.
3. Rather than downloading OpenCV binaries from PIP/APT, compile OpenCV source code manually to enable CUDA backend.
