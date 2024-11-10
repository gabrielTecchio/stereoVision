from scipy import linalg
import numpy as np

def calculate_rotation_matrix(center_point, corners):
    # Convert input to numpy arrays for consistency
    center_point = np.array(center_point)
    corners = np.array(corners)

    # Calculate the vectors from the center point to each corner
    vectors = np.array([corner - center_point for corner in corners])
    
    # Calculate the covariance matrix
    covariance_matrix = np.dot(vectors.T, vectors)
    
    # Perform Singular Value Decomposition (SVD) to get rotation matrix
    _, _, v_transpose = np.linalg.svd(covariance_matrix)
    
    # Ensure that the transformation matrix has proper orientation (det = 1)
    if np.linalg.det(v_transpose) < 0:
        v_transpose[-1] *= -1
    
    # Return the rotation matrix
    return v_transpose.T

def rotation_matrix_to_euler_angles(rotation_matrix):
    # Extract individual rotation angles from the rotation matrix
    #sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])

    # Ensure the matrix is a numpy array
    rotation_matrix = np.array(rotation_matrix)
    
    # Extract the angles from the rotation matrix
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def stereoVisionCalculation(mtxL, mtxR, R, T, centerL, boxL, centerR, boxR):
    # Parei aqui...
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtxL @ RT1 #projection matrix for C1
        
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtxR @ RT2 #projection matrix for C2

    def DLT(P1, P2, point1, point2):
        #print(P1, P2, point1, point2)
        A = [point1[1]*P1[2,:] - P1[1,:], P1[0,:] - point1[0]*P1[2,:], point2[1]*P2[2,:] - P2[1,:], P2[0,:] - point2[0]*P2[2,:]]
        A = np.array(A).reshape((4,4))

        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices = False)
        
        return Vh[3,0:3]/Vh[3,3]

    mtxPointsL = [(centerL)]
    for point in boxL:
        mtxPointsL.append((point[0], point [1]))

    mtxPointsR = [(centerR)]
    for point in boxR:
        mtxPointsR.append((point[0], point [1]))

    pointsVector = []
    for p1, p2 in zip(mtxPointsL, mtxPointsR):
        _p3d = DLT(P1, P2, p1, p2)
        pointsVector.append(_p3d)

    center_point = pointsVector[0]
    corners = pointsVector[1:3]

    rotation_matrix = calculate_rotation_matrix(center_point, corners)
    rotation_angles = rotation_matrix_to_euler_angles(rotation_matrix)
                
    #x = np.array([pointsVector[1][0], pointsVector[2][0], pointsVector[3][0], pointsVector[4][0]]).reshape((-1, 1))
    #y = np.array([pointsVector[1][1], pointsVector[2][1], pointsVector[3][1], pointsVector[4][1]]).reshape((-1, 1))
    #z = np.array([pointsVector[1][2], pointsVector[2][2], pointsVector[3][2], pointsVector[4][2]])
    #print(x, z)

    #model = LinearRegression().fit(x, z)
    #model2 = LinearRegression().fit(x, y)
    #print(r_sq)

    return center_point, rotation_angles