# ENGN6528 CLab-3: Camera Calibration & Homography Estimation

This folder contains code and analysis for two major tasks in computer vision: 3D-2D Camera Calibration and Two-View DLT-Based Homography Estimation. These tasks are part of the CLab-3 for the course ENGN6528, focusing on camera calibration, 3D projection, and homography estimation between image pairs. The main goals are to:
- **Camera Calibration**: Understand the geometric relationship between 3D world coordinates and their 2D projections in the image.
- **Homography Estimation**: Learn to apply the Direct Linear Transformation (DLT) method to estimate a homography matrix between two images.

This submission made contains the following:
- **Task Code**: Code for each task
- **Lab Report**: A report detailing the tasks, analysis, and results of the lab
- **Result Images**: The resulting images from the code, analysed in report

## Tasks Overview

The code in this repository solves the following tasks:
#### Task 1: Camera Calibration
- Objective: To calibrate a camera using known 3D world points and corresponding 2D image points. The goal is to extract intrinsic parameters such as focal length, principal point, and distortion coefficients.

#### Task 2: Homography
- Objective: To compute and apply a homography matrix to warp an image. The homography is computed from a set of corresponding points in two images, and the result is used for tasks like image stitching and object recognition.
