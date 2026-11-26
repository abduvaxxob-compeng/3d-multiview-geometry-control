import os
import cv2
import numpy as np

def load_intrinsics(path="calibration_params.npz"):
    if os.path.exists(path):
        data = np.load(path)
        return data["K"], data["dist"]
    K = np.array([[800.0, 0.0, 320.0],
                  [0.0, 800.0, 240.0],
                  [0.0,   0.0,   1.0]], dtype=np.float64)
    return K, np.zeros((5, 1))

def reconstruct_two_view(img1_path, img2_path, K):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Input stereo image pair not found.")

    # 1. Feature detection and descriptor matching (SIFT + Lowe's ratio test)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # 2. Essential Matrix estimation via RANSAC & Relative Pose Recovery
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]

    _, R, t, pose_mask = cv2.recoverPose(E, inliers1, inliers2, K)

    # 3. Projection matrices P1 = K[I|0] and P2 = K[R|t]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    # 4. Linear 3D Point Triangulation
    pts4d_hom = cv2.triangulatePoints(P1, P2, inliers1.T, inliers2.T)
    pts3d = (pts4d_hom[:3] / pts4d_hom[3]).T

    print(f"Matched features: {len(good_matches)} | RANSAC inliers: {len(inliers1)}")
    print(f"Triangulated 3D point cloud shape: {pts3d.shape}")
    return pts3d, R, t

if __name__ == "__main__":
    K, _ = load_intrinsics()
    if os.path.exists("data/view1.jpg") and os.path.exists("data/view2.jpg"):
        reconstruct_two_view("data/view1.jpg", "data/view2.jpg", K)
    else:
        print("Ready for two-view triangulation. Place stereo pairs in data/view1.jpg and data/view2.jpg.")
