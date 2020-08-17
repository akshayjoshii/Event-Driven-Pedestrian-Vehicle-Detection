# Robust Event Based Object Detection with YOLO

#Code Changes:
1. Changed the input method from argparse
2. Altered the video writer output to AVI for better processing speed
3. Implemented a frame-level total object counter to understand the density of objects at any point of time
4. Implemented frame-level class-wise object counter
5. Implemented ROI mechanism to detect vehicles crossing the ROI line
6. Implemented & Optimised IN and OUT counter to count vehicles in different lanes.


Pending tasks:
1. Compile the OpenCV DNN module with CUDA backend
