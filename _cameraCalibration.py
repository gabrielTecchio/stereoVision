from datetime import datetime
import numpy as np
import glob
import cv2
import csv
import os

# Chess pattern
rows = 9 #number of checkerboard rows.
columns = 6 #number of checkerboard columns.
world_scaling = 22 #change this to the real world square size in mm.

def capture_and_save_image(capL, capR, output_folder, picturesNumb):
    input('Pour prendre des photos, appuyez sur une touche quelconque.')
    print('...toutes les 8 secondes, une nouvelle photo est prise.')

    ## Variables ##
    index = 0
    now = datetime.now()

    while True:
        # Frame capture
        _, frameLeft = capL.read()
        _, frameRight = capR.read()

        if (datetime.now() - now).total_seconds() >= 8:
            print(f"Picture_{index}!")
            # Generate a unique filename based on the index number
            filenameLeft = f"L_chess_{index}.jpg"
            filepathLeft = os.path.join(output_folder, filenameLeft)
            filenameRight = f"R_chess_{index}.jpg"
            filepathRight = os.path.join(output_folder, filenameRight)

            # Save the captured frame as an image each 8 seconds
            cv2.imwrite(filepathLeft, frameLeft)
            cv2.imwrite(filepathRight, frameRight)
            
            index = index + 1
            now = datetime.now()
        else:
            pass
        
        frameLeft_copy = cv2.resize(frameLeft, (640, 420))
        frameRight_copy = cv2.resize(frameRight, (640, 420))
        cv2.imshow('Webcams Feed', cv2.hconcat([frameLeft_copy, frameRight_copy]))

        if (cv2.waitKey(1) & 0xFF == ord('q')) or index == picturesNumb:
            # On keyboard interrupt or reaching the defined number of pictures, close all windows and stop the loop
            cv2.destroyAllWindows()
            break

def cam_calibration(output_folder):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    # Arrays to store object points and image points from all the images.
    objpointsL = [] # 3d point in real world space
    imgpointsL = [] # 2d points in image plane.
    objpointsR = [] # 3d point in real world space
    imgpointsR = [] # 2d points in image plane.

    imagesL = glob.glob(output_folder + r'\L_chess_*.jpg')
    imagesR = glob.glob(output_folder + r'\R_chess_*.jpg')        

    for fname in imagesL:
        img = cv2.imread(fname)
        grayL = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(grayL, (9,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpointsL.append(objp)
            corners2 = cv2.cornerSubPix(grayL,corners, (11,11), (-1,-1), criteria)
            imgpointsL.append(corners2)
            """# Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners2, ret)
            cv2.imshow('Chess Images Left', img)
            cv2.waitKey(50)"""    
        
    for fname in imagesR:
        img = cv2.imread(fname)
        grayR = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(grayR, (9,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpointsR.append(objp)
            corners2 = cv2.cornerSubPix(grayR,corners, (11,11), (-1,-1), criteria)
            imgpointsR.append(corners2)
            """# Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners2, ret)
            cv2.imshow('Chess Images Right', img)
            cv2.waitKey(500)"""

    cv2.destroyAllWindows()

    camera_matrix = np.array([[1000,0,640],[0,1000,360],[0,0,1]])
    
    ret, mtxL, distL, rvecs, tvecs = cv2.calibrateCamera(objpointsL, imgpointsL, grayL.shape[::-1], camera_matrix, None, flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO))
    ret, mtxR, distR, rvecs, tvecs = cv2.calibrateCamera(objpointsR, imgpointsR, grayR.shape[::-1], camera_matrix, None, flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO))
    
    file_object = open(os.path.join(output_folder,'calibMtx_L.txt'), 'w', newline='')
    writer = csv.writer(file_object)
    writer.writerows(mtxL)
    writer.writerows(distL)
    file_object.close()

    file_object = open(os.path.join(output_folder,'calibMtx_R.txt'), 'w', newline='')
    writer = csv.writer(file_object)
    writer.writerows(mtxR)
    writer.writerows(distR)
    file_object.close()

    return mtxL, distL, mtxR, distR

def stereo_calibration(mtx1, dist1, mtx2, dist2, output_folder):
    #read the synched frames
    frames_folder = glob.glob(output_folder + r'\*.jpg')
    images_names = sorted(frames_folder)
    c1_images_names = images_names[:len(images_names)//2]
    c2_images_names = images_names[len(images_names)//2:]
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv2.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv2.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            """# Draw and display the corners
            cv2.drawChessboardCorners(frame1, (5,8), corners1, c_ret1) 
            cv2.drawChessboardCorners(frame2, (5,8), corners2, c_ret2)
            cv2.imshow('Webcams Feed', cv2.hconcat([cv2.resize(frame1, (640, 420)), cv2.resize(frame2, (640, 420))]))
            cv2.waitKey(500)"""
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    cv2.destroyAllWindows()

    ret, CM1, new_dist1, CM2, new_dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1, mtx2, dist2, (width, height), criteria = criteria, flags = cv2.CALIB_FIX_INTRINSIC)
    
    #print("Distortion Vectors Before", dist1, dist2)
    #print("Distortion Vectors After", new_dist1, new_dist2)

    file_object = open(os.path.join(output_folder,'calibStereoMtx.txt'), 'w', newline='')
    writer = csv.writer(file_object)
    writer.writerows(R)
    writer.writerows(T)
    file_object.close()

    return R, T

def read_calib_mtx(path):
    i = 0
    camera_matrix = []
    dist_coeffs = []

    with open(os.path.join(path), newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if len(row) == 3 and i <= 2:
                camera_matrix.append([float(row[0]), float(row[1]), float(row[2])])
            elif len(row) == 5:
                dist_coeffs.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
    csvfile.close()

    return np.array(camera_matrix), np.array(dist_coeffs)

def read_stereo_mtx(path):
    i = 0
    R = []
    T = []

    with open(os.path.join(path), newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if len(row) == 3 and i <= 2:
                R.append([float(row[0]), float(row[1]), float(row[2])])
            elif len(row) == 1:
                T.append([float(row[0])])
    csvfile.close()

    return np.array(R), np.array(T)

if __name__ == "__main__":
    calib_mtx_path = project_folder = os.path.join(os.getcwd(), 'dataFolder_240529', 'chessPictures', 'calibStereoMtx.txt')
    R, T = read_stereo_mtx(calib_mtx_path)