import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
from statistics import mean

from _cameraCalibration import read_calib_mtx

def generate_aruco_marker(marker_id, marker_size, output_path):
    # Define dictionary type (e.g., aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    # Generate the marker image
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size, marker_image, 1)

    # Save the marker image to a file
    cv2.imwrite(output_path, marker_image)

    print(f"ArUco marker with ID {marker_id} generated and saved to {output_path}")

def detect_aruco_markers(image_path, camera_matrix, dist_coeffs):
    # Load the test image
    test_image = cv2.imread(image_path)
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Define dictionary type (e.g., aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    # Initialize the detector parameters
    parameters = cv2.aruco.DetectorParameters_create()

    # Detect markers in the image
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    #print(rejected_img_points)

    # Draw detected markers on the image
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(test_image, corners, ids)

        # Estimate pose of each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 50, camera_matrix, dist_coeffs)

        # Draw axis for each marker
        for i in range(len(ids)):
            cv2.aruco.drawAxis(test_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 100)
            """X-axis: Shown in red.
            Y-axis: Shown in green.
            Z-axis: Shown in blue."""

            # Transformation matrix from camera frame to marker frame
            R = cv2.Rodrigues(rvecs[i])[0]  # Convert rotation vector to rotation matrix
            t = tvecs[i].reshape(-1, 1)     # Translation vector

            #print(R)

            R = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
            #print(t)

            # Transformation matrix from camera frame to marker frame
            T_cm_to_marker = np.hstack((R, t))
            T_cm_to_marker = np.vstack((T_cm_to_marker, np.array([0, 0, 0, 1])))

            #print(f"Transformation matrix from camera frame to marker frame (Marker ID {ids[i]}):\n{T_cm_to_marker}")

        # Display the image with detected markers and pose estimation
        #cv2.imshow('Detected ArUco Markers', test_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    else:
        print("No ArUco markers detected in the image.")

    with open(os.path.join(image_path.replace('marker_0.png', 'pntRepair.txt')), newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            [xrob, yrob, zrob, urob] = [float(row[0]), float(row[1]), float(row[2]), float(row[3])] # TRANSFORMAR EM INTEIRO
    csvfile.close()
    zrob = zrob-30
    #print(xrob, yrob, zrob)

    # Real position of the marker in the world coordinate system
    marker_pos_in_world = np.array([[np.sin(urob*np.pi/180), np.cos(urob*np.pi/180), 0, -yrob],   # Example: identity matrix, assuming marker at origin
                                    [0, 0, 1, -zrob], 
                                    [np.cos(urob*np.pi/180), -np.sin(urob*np.pi/180), 0, -xrob],
                                    [0, 0, 0, 1]])
                                    

    T_cm_to_pos_in_world = T_cm_to_marker@marker_pos_in_world

    if __name__ == "__main__":
        T_cm_to_pos_in_world[:3, :3] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
        #T_cm_to_pos_in_world[:3, 3] = [T_cm_to_pos_in_world[0, 3] - (50-56.1310286), T_cm_to_pos_in_world[1, 3] - (75.0-70.53660587), T_cm_to_pos_in_world[0, 3] - (26-25.97873863)]
        return T_cm_to_pos_in_world, marker_pos_in_world, T_cm_to_marker

    return T_cm_to_pos_in_world

if __name__ == "__main__":
    folder_path = os.path.join(os.getcwd(), 'dataFolder_240729')
    output_path = os.path.join(folder_path, 'QRcode', 'marker.png')  # Path to save the generated marker
    test_image_path = os.path.join(folder_path, 'QRcode2', 'marker_0.png') # Path to the test image containing ArUco markers

    # Camera matrix and distortion coefficients (you need to provide actual values)
    camera_matrix, dist_coeffs = read_calib_mtx(os.path.join(folder_path, 'chessPictures', 'calibMtx_L.txt'))

    #generate_aruco_marker(marker_id = 0, marker_size = 50, output_path)
    T_cm_to_pos_in_world, marker_pos_in_world, T_cm_to_marker  = detect_aruco_markers(test_image_path, camera_matrix, dist_coeffs)

    #print(f"Transformation matrix from camera frame to world frame :\n{T_cm_to_pos_in_world}")

    T_cm_to_pos_in_world = linalg.inv(T_cm_to_pos_in_world)
    marker_pos_in_world = linalg.inv(marker_pos_in_world)
    T_cm_to_marker = T_cm_to_pos_in_world@T_cm_to_marker
    #print(T_cm_to_marker[:3, 3])
    T_cm_to_pos_in_world[:3, 3]=[mean([965.77487983, 967.50582992, 952.06838467]), mean([-58.79514251, -57.21729156,-56.88161877]), mean([-152.55068461, -164.48017096,-147.73394087])]
    T_cm_to_pos_in_world[:3, 3]=[961.78303147,-57.63135095,-154.92159881]
    print(T_cm_to_pos_in_world)

    # Extract rotation matrix and translation vector
    R = T_cm_to_pos_in_world[:3, :3]
    T = T_cm_to_pos_in_world[:3, 3]

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot reference frame axes
    # X-axis
    ax.quiver(T[0], T[1], T[2], R[0, 0], R[1, 0], R[2, 0], color='orange', label='X-axis (Cam)', length=100)
    # Y-axis
    ax.quiver(T[0], T[1], T[2], R[0, 1], R[1, 1], R[2, 1], color='purple', label='Y-axis (Cam)', length=100)
    # Z-axis
    ax.quiver(T[0], T[1], T[2], R[0, 2], R[1, 2], R[2, 2], color='brown', label='Z-axis (Cam)', length=100)

    # Set plot limits
    ax.set_xlim([100, 800])
    ax.set_ylim([-350, 350])
    ax.set_zlim([0, 700])

    # Set tick marks distance
    ax.set_xticks(np.arange(-100, 800, 100))
    ax.set_yticks(np.arange(-350, 350, 100))
    ax.set_zticks(np.arange(0, 700, 100))

    # Extract rotation matrix and translation vector
    R = marker_pos_in_world[:3, :3]
    T = marker_pos_in_world[:3, 3]

    # Plot reference frame axes
    # X-axis
    ax.quiver(T[0], T[1], T[2], R[0, 0], R[1, 0], R[2, 0], color='c', label='X-axis (Marker)', length=100)
    # Y-axis
    ax.quiver(T[0], T[1], T[2], R[0, 1], R[1, 1], R[2, 1], color='m', label='Y-axis (Marker)', length=100)
    # Z-axis
    ax.quiver(T[0], T[1], T[2], R[0, 2], R[1, 2], R[2, 2], color='y', label='Z-axis (Marker)', length=100)

    # Plot reference frame axes
    # X-axis
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X-axis (World)', length=100)
    # Y-axis
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y-axis (World)', length=100)
    # Z-axis
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z-axis (World)', length=100)
    
    # Extract rotation matrix and translation vector
    R = T_cm_to_marker[:3, :3]
    T = T_cm_to_marker[:3, 3]

    # Plot reference frame axes
    # X-axis
    ax.quiver(T[0], T[1], T[2], R[0, 0], R[1, 0], R[2, 0], color='c', length=100, linestyle='dotted')
    # Y-axis
    ax.quiver(T[0], T[1], T[2], R[0, 1], R[1, 1], R[2, 1], color='m', length=100, linestyle='dotted')
    # Z-axis
    ax.quiver(T[0], T[1], T[2], R[0, 2], R[1, 2], R[2, 2], color='y', length=100, linestyle='dotted')

    # Set azimuth and elevation
    ax.view_init(azim=0, elev=20)

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()