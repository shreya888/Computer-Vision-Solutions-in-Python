# ENGN6528 CLab-3: Camera Calibration & Homography Estimation

This folder contains code and analysis for two major tasks in computer vision: 3D-2D Camera Calibration and Two-View DLT-Based Homography Estimation. These tasks are part of the CLab-3 for the course ENGN6528, focusing on camera calibration, 3D projection, and homography estimation between image pairs. The main goals are to:
- **Camera Calibration**: Understand the geometric relationship between 3D world coordinates and their 2D projections in the image.
- **Homography Estimation**: Learn to apply the Direct Linear Transformation (DLT) method to estimate a homography matrix between two images.

## Tasks Overview

The code in this repository solves the following tasks:
#### Task 1: Camera Calibration
- Objective: To calibrate a camera using known 3D world points and corresponding 2D image points. The goal is to extract intrinsic parameters such as focal length, principal point, and distortion coefficients.

#### Task 2: Homography
- Objective: To compute and apply a homography matrix to warp an image. The homography is computed from a set of corresponding points in two images, and the result is used for tasks like image stitching and object recognition.

## Folder Structure
```
Computer-Vision-Solutions-in-Python/
└── Computer Lab 3/
    ├── code/
    │   ├── camera_calibration.py        # Camera calibration code
    │   ├── vgg_KR_from_P.py             # Code to decompose camera matrix into K, R, and t
    ├── data/
    │   ├── camera_calib.npy             # Saved camera calibration data
    │   └── camera_calib_resize.npy      # Camera calibration data for resized image
    ├── images/
    │   ├── Left.jpg                     # Left image for homography estimation
    │   ├── Right.jpg                    # Right image for homography estimation
    ├── results/
    │   ├── visualizing XYZ.png          # Visualization of 3D coordinates projection
    │   ├── visualizing points left.png  # Points selected on Left image
    │   ├── visualizing points right.png # Points selected on Right image
    │   ├── visualizing points selected.png # Points selected on both images
    │   ├── visualizing points selected resized.png # Points on resized image
    │   ├── T2 My homography warp result with near points.png # Result of warping with near points
    │   ├── T2 My homography warp result when points are far away.png # Result of warping with far points
    ├── reports/
    │   └── CLab3-Report-U7195872.pdf    # Final lab report detailing the tasks, analysis, and results of the lab
    ├── README.md                        # This file, with instructions and project info
```
