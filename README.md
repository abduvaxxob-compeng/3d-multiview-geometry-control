# 3D Multi-View Geometry & Dynamic Control System Modeling

Undergraduate thesis and research repository implementing spatial 3D perception pipelines (camera calibration, stereo feature matching, epipolar geometry, and 3D point triangulation) alongside closed-loop dynamic kinematic control simulations for autonomous mobile robotics.

## Repository Structure

- `camera_calibration.py` — Intrinsic and extrinsic camera calibration using chessboard corner detection and distortion parameter estimation (OpenCV).
- `multiview_triangulation.py` — SIFT feature matching, Essential matrix estimation via RANSAC, relative pose recovery, and linear 3D point triangulation.
- `mobile_robot_kinematics.m` — MATLAB closed-loop PID trajectory tracking and differential-drive dynamic control simulation.
- `requirements.txt` — Python dependencies for computer vision and numerical computation.

## Technical Highlights
- **3D Visual Perception:** Full two-view geometry pipeline recovering camera motion (R, t) and 3D scene structure from stereo image pairs.
- **Control & Dynamics Modeling:** Closed-loop PID control simulation evaluating step response, settling time, and path-tracking accuracy.

## Quickstart

1. Install Python dependencies:
   pip install -r requirements.txt

2. Run camera calibration:
   python camera_calibration.py

3. Run two-view epipolar geometry & 3D reconstruction:
   python multiview_triangulation.py
