import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot_trajectory(rotations, translations):
    # Initialize the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory')

    # Initialize the origin position
    origin = -rotations[0].T @ translations[0]
    # Iterate over the rotations and translations
    i=0
    for R, t in zip(rotations, translations):
        # Calculate the camera position
        position = -R.T @ t
        
        # Plot the camera position
        ax.scatter(position[0], position[1], position[2], c='r', marker='o')
        print("Position ",i)
        print(position)
        # Plot the trajectory line
        ax.plot([origin[0, 0], position[0, 0]], [origin[1, 0], position[1, 0]], [origin[2, 0], position[2, 0]], c='b')
        origin = position
        # Update the origin position
        i=i+1

    # Show the plot
    plt.show()

# Example rotations and translations




def estimate_camera_pose(img1, img2):
    # Detect keypoints and compute descriptors
    detector = cv2.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    # Match keypoints
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)

    # Estimate homography matrix using RANSAC
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned = cv2.warpPerspective(img1, H, (1920, 1080))
    # Decompose homography matrix
    camera_matrix = np.float32([[100,0,960],[0,100,540],[0,0,1]])
    _, rotations, translations, _ = cv2.decomposeHomographyMat(H, camera_matrix)

    # Extract rotation and translation
    R = rotations[0]
    t = translations[0]

    return R, t,aligned
camera_matrix = np.float32([[100,0,960],[0,100,540],[0,0,1]])
# Load images
img1 = cv2.imread('img1.png', cv2.IMREAD_ANYCOLOR)
img2 = cv2.imread('img2.png', cv2.IMREAD_ANYCOLOR)
img3 = cv2.imread('img3.png', cv2.IMREAD_ANYCOLOR)
points_2d = np.load('vr2d.npy')
points_3d = np.load('vr3d.npy')

_,rvecs, t1,_ = cv2.solvePnPRansac(points_3d, points_2d, camera_matrix, None,useExtrinsicGuess=True)
R1, _ = cv2.Rodrigues(rvecs)
print("Camera pose for img1:")
print("Rotation:")
print(R1)
print("Translation:")
print(t1)
R2, t2,img1img2_transformed = estimate_camera_pose(img1, img2)
R3, t3,img1img3_transformed = estimate_camera_pose(img1, img3)
# Print the results
print("Camera pose for img2:")
print("Rotation:")
print(R2)
print("Translation:")
print(t2)

print("Camera pose for img3:")
print("Rotation:")
print(R3)
print("Translation:")
print(t3)
rotations = [np.array(R1),np.array(R2),np.array(R3)]
translations = [np.array(t1),np.array(t2), np.array(t3)]
plot_trajectory(rotations,translations)

