import glob
import os
import cv2
import numpy as np

# Chessboard inner corners (columns, rows) and square size in meters
PATTERN_SIZE = (9, 6)
SQUARE_SIZE = 0.025
CALIB_IMAGES_DIR = "data/calibration/*.jpg"
OUTPUT_FILE = "calibration_params.npz"

def calibrate_camera():
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane
    images = glob.glob(CALIB_IMAGES_DIR)

    if not images:
        print("No calibration images found. Generating synthetic intrinsic matrix for pipeline testing...")
        K = np.array([[800.0, 0.0, 320.0],
                      [0.0, 800.0, 240.0],
                      [0.0,   0.0,   1.0]], dtype=np.float64)
        dist = np.zeros((5, 1), dtype=np.float64)
        np.savez(OUTPUT_FILE, K=K, dist=dist)
        print(f"Saved default intrinsics to {OUTPUT_FILE}")
        return K, dist

    img_shape = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(refined_corners)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    # Compute mean reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    print(f"Calibration RMS Error: {ret:.4f} px | Mean Reprojection Error: {mean_error:.4f} px")
    np.savez(OUTPUT_FILE, K=K, dist=dist, mean_error=mean_error)
    return K, dist

if __name__ == "__main__":
    calibrate_camera()
