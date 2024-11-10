from datetime import datetime
from scipy import linalg
import tensorflow as tf
import numpy as np
import socket
import csv
import cv2
import os
from statistics import mean

# Global Variables #
tresholdValueL = 50
tresholdValueR = 50

displayBefore = False
chessCaptures = False
camCalibration = False
QRCodeCapture = False
linearCoeffsCorrection = False
logSaving = False
aiTesting = False

#project_directory = os.path.join(os.getcwd(), 'dataFolder_' + str(datetime.now().strftime("%y%m%d")))
project_directory = os.path.join(os.getcwd(), 'dataFolder_240805')

from _cameraCalibration import capture_and_save_image, cam_calibration, stereo_calibration, read_calib_mtx, read_stereo_mtx
# capture_and_save_image(capL, capR, chess_folder, picturesNumb)
# mtxL, distL, mtxR, distR = cam_calibration(chess_folder)
# R, T = stereo_calibration(mtx1, dist1, mtx2, dist2, chess_folder)
# mtx, dist = read_calib_mtx(path)
# R, T = read_stereo_mtx(path)

from _findAruco import detect_aruco_markers
# T_cm_to_pos_in_world = detect_aruco_markers(image_path, camera_matrix, dist_coeffs)

from _findObject import find_object
# frameL, centerL, boxL, frameR, centerR, boxR, img_dilationL, img_dilationR, count = find_object(capL, capR, tresholdValueL = 40, tresholdValueR = 120, minArea = 5000, maxArea = 500000, histogram = False, drawContours = True)

from _linearRegression import calculateLinearCoefficients
# [interceptX, slopeX, interceptY, slopeY, interceptZ, slopeZ, interceptU, slopeU] = calculateLinearCoefficients(project_folder, mtxL, mtxR, R, T, T_cm_to_pos_in_world, _print = False)

from _saveLog import saveLog
# saveLog(folder, mtx)

from _stereoCalculations import stereoVisionCalculation
# [x, y, z], [u, v, w] = stereoVisionCalculation(mtxL, mtxR, R, T, centerL, boxL, centerR, boxR)

if __name__ == "__main__":
    startTime = datetime.now()
    informationsLog = []
    
    HOST = "0.0.0.0"  # Standard loopback interface address (localhost)
    PORT = 5000  # Port to listen on (non-privileged ports are > 1023)

    chess_folder = os.path.join(project_directory, 'chessPictures')
    aruco_folder = os.path.join(project_directory, 'QRcode')
    pictures_folder = os.path.join(project_directory, 'stereoVideoPictures')
    model_path = os.path.join(project_directory, 'cnnModel_0_5k')

    # Create the project folder if it doesn't exist
    if not os.path.exists(project_directory):
        os.makedirs(project_directory)
        os.makedirs(chess_folder)
        os.makedirs(aruco_folder)
        os.makedirs(pictures_folder)

    # Load CNN:
    if aiTesting:
        model = tf.keras.models.load_model(model_path)
        chessCaptures = False
        camCalibration = False
        QRCodeCapture = False
        linearCoeffsCorrection = False

    # Starting webcams
    capL = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capL.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capL.set(cv2.CAP_PROP_GAIN, 0)
    capL.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    capL.set(cv2.CAP_PROP_EXPOSURE, -5.0)
    capL.set(cv2.CAP_PROP_AUTO_WB, 0)
    capL.set(cv2.CAP_PROP_FOCUS, 0)
    capL.set(cv2.CAP_PROP_SETTINGS, 1)

    capR = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capR.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capR.set(cv2.CAP_PROP_GAIN, 0)
    capR.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    capR.set(cv2.CAP_PROP_EXPOSURE, -5.0)
    capR.set(cv2.CAP_PROP_AUTO_WB, 0)
    capR.set(cv2.CAP_PROP_FOCUS, 0)
    capR.set(cv2.CAP_PROP_SETTINGS, 1)

    while displayBefore:
        # Capture a frame
        _, frameL = capL.read()
        _, frameR = capR.read()
 
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            # On keyboard interrupt or reaching the defined number of pictures, close all windows and stop the loop
            cv2.destroyAllWindows()
            break

        _, gframeL = cv2.threshold(cv2.cvtColor(frameL, cv2.COLOR_RGB2GRAY), tresholdValueL, 255, cv2.THRESH_BINARY_INV) #cv2.THRESH_BINARY
        _, gframeR = cv2.threshold(cv2.cvtColor(frameR, cv2.COLOR_RGB2GRAY), tresholdValueR, 255, cv2.THRESH_BINARY_INV) #cv2.THRESH_BINARY

        cv2.line(frameL, (0, 638), (1280, 638), (0, 255, 0), 2) # Draw horizontal line (green crosshair)
        cv2.line(frameL, (640, 0), (640, 720), (0, 255, 0), 2) # Draw vertical line (green crosshair)
        cv2.line(frameR, (0, 638), (1280, 638), (0, 255, 0), 2) # Draw horizontal line (green crosshair)
        cv2.line(frameR, (640, 0), (640, 720), (0, 255, 0), 2) # Draw vertical line (green crosshair)
        cv2.putText(frameL, "Left", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frameR, "Right", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        frame = cv2.hconcat([cv2.resize(frameL, (640, 420)), cv2.resize(frameR, (640, 420))])
        gframe = cv2.hconcat([cv2.resize(gframeL, (640, 420)), cv2.resize(gframeR, (640, 420))])
        cv2.imshow('Webcams Feed', frame)
        cv2.imshow('Webcams Feed Black and White', gframe)
    
    informationsLog.append(['Starting Time', startTime,'Duration (s)', (datetime.now() - startTime).total_seconds()])

    if chessCaptures:
        capturePhotosStart = datetime.now()
        capture_and_save_image(capL, capR, chess_folder, picturesNumb = 100)
        informationsLog.append(['Chess Capturing Time', capturePhotosStart, 'Duration (s)', (datetime.now() - capturePhotosStart).total_seconds()])
    else:
        informationsLog.append(['No calibration pictures taken'])
    
    if camCalibration:
        camCalibrationStart = datetime.now()
        mtxL, distL, mtxR, distR = cam_calibration(chess_folder)
        informationsLog.append(['Cam Calib Time', camCalibrationStart, 'Duration (s)', (datetime.now() - camCalibrationStart).total_seconds()])
        stereoCalibrationStart = datetime.now()
        R, T = stereo_calibration(mtxL, distL, mtxR, distR, chess_folder)
        informationsLog.append(['Stereo Calib Time', stereoCalibrationStart, 'Duration (s)', (datetime.now() - stereoCalibrationStart).total_seconds()])
    else:
        camCalibrationStart = datetime.now()
        mtxL, distL = read_calib_mtx(os.path.join(chess_folder, 'calibMtx_L.txt'))
        mtxR, distR = read_calib_mtx(os.path.join(chess_folder, 'calibMtx_R.txt'))
        informationsLog.append(['Read Cam Calib Time', camCalibrationStart, 'Duration (s)', (datetime.now() - camCalibrationStart).total_seconds()])
        stereoCalibrationStart = datetime.now()
        R, T = read_stereo_mtx(os.path.join(chess_folder, 'calibStereoMtx.txt'))
        informationsLog.append(['Read Stereo Calib Time', stereoCalibrationStart, 'Duration (s)', (datetime.now() - stereoCalibrationStart).total_seconds()])
    
    while QRCodeCapture:
        QRCodeAcquisitionStart = datetime.now()
        
        # Capture a frame
        _, frameL = capL.read()

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            
            # On keyboard interrupt or reaching the defined number of pictures, close all windows and stop the loop
            xrob = input("X Robot :")
            yrob = input("Y Robot :")
            zrob = input("Z Robot :")
            urob = input("U Robot :")

            cv2.imwrite(os.path.join(aruco_folder, 'marker_0.png'), frameL)

            file_object = open(os.path.join(aruco_folder, 'pntRepair.txt'), 'w', newline='')
            writer = csv.writer(file_object)
            writer.writerow([str(xrob), str(yrob), str(zrob), str(urob)])
            file_object.close()

            informationsLog.append(['QRcode Acquisition Time', QRCodeAcquisitionStart, 'Duration (s)', (datetime.now() - QRCodeAcquisitionStart).total_seconds()])
            break

        cv2.imshow('Webcams Feed', frameL)

    referenciaArucoStart = datetime.now()
    #cv2.imshow("QRCODE", cv2.imread(os.path.join(aruco_folder, 'marker_0.png')))
    #cv2.waitKey(0)
    T_cm_to_pos_in_world = detect_aruco_markers(os.path.join(aruco_folder, 'marker_0.png'), mtxL, distL)
    
    ###-Teste_Média_de_três_medidas-###
    #T_cm_to_pos_in_world[:3, 3]=[961.78303147,-57.63135095,-154.92159881]
    ####################
    
    informationsLog.append(['Aruco Ref Calculation Time', referenciaArucoStart, 'Duration (s)', (datetime.now() - referenciaArucoStart).total_seconds()])

    #print(T_cm_to_pos_in_world)
    #T_cm_to_pos_in_world[:, 0:3] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0], [0, 0, 0]]
    #T_cm_to_pos_in_world = [[0, 1, 0, 0], [0, 0, -1, 135], [-1, 0, 0, 596], [0, 0, 0, 1]]
    #print(T_cm_to_pos_in_world)
    #input()

    if linearCoeffsCorrection:
        linearCoeffsStart = datetime.now() 
        try:
            coeffs = calculateLinearCoefficients(project_directory, mtxL, mtxR, R, T, T_cm_to_pos_in_world, _print = False)
            informationsLog.append(['Linear Correction Calculation Time', linearCoeffsStart, 'Duration (s)', (datetime.now() - linearCoeffsStart).total_seconds()])
        except Exception as error:
            linearCoeffsCorrection = False
            informationsLog.append(['ERROR @', linearCoeffsStart,'No Linear Coeffs Calculated'])
            input("An exception occurred in Linear Coeff Calculation:", error)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    print(f"Connected by {addr}")
    test = conn.recv(1024).decode()
    input(f"Message recieve by {addr}: {test}. Press ENTER to continue.")

    # Detection and estimation starts here
    while True:
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            # On keyboard interrupt or reaching the defined number of pictures, close all windows and stop the loop
            cv2.destroyAllWindows()
            break
        
        try:
            searchTime = datetime.now()
            frameL, centerL, boxL, frameR, centerR, boxR, bwL, bwR, count = find_object(capL, capR, 
                                                                                        tresholdValueL, tresholdValueR, 
                                                                                        minArea = 10000, maxArea = 500000, 
                                                                                        histogram = False, drawContours = True)
            
            informationsLog.append(['Search Time', searchTime,'Duration (s)', (datetime.now() - searchTime).total_seconds()])
            
            stereoCalcTime = datetime.now()
            
            [x, y, z], [u, v, w] = stereoVisionCalculation(mtxL, mtxR, R, T, centerL, boxL, centerR, boxR)
            pntRobo = linalg.inv(T_cm_to_pos_in_world).dot(np.array([[x], [y], [z], [1]]))
            #print(f"Robot coordinates: X {pntRobo[0][0]}, Y {pntRobo[1][0]}, Z {pntRobo[2][0]}, U {u}, V {v}, W {w}")
         
            if aiTesting:
                temp = [centerL[0]/1280, centerL[1]/720,  centerR[0]/1280, centerR[1]/720,
                        boxL[0][0]/1280, boxL[0][1]/720, boxL[1][0]/1280, boxL[1][1]/720, boxL[2][0]/1280, boxL[2][1]/720, boxL[3][0]/1280, boxL[3][1]/720, 
                        boxR[0][0]/1280, boxR[0][1]/720, boxR[1][0]/1280, boxR[1][1]/720, boxR[2][0]/1280, boxR[2][1]/720, boxR[3][0]/1280, boxR[3][1]/720]
                input_point = np.array(temp).reshape(1, -1)  # Reshape to (1, 16)
                predictions = model.predict(input_point)
                pntRobo[0][0] = predictions[0][0]
                pntRobo[1][0] = predictions[0][1]
                pntRobo[2][0] = predictions[0][2]
                v = predictions[0][3]
                #print(f"Robot coordinates: X {pntRobo[0][0]}, Y {pntRobo[1][0]}, Z {pntRobo[2][0]}, V {v}")

            if linearCoeffsCorrection:
                pntRobo[0][0]   = coeffs[0] + coeffs[1]*pntRobo[0][0]
                pntRobo[1][0]   = coeffs[2] + coeffs[3]*pntRobo[1][0]
                pntRobo[2][0]   = coeffs[4] + coeffs[5]*pntRobo[2][0]
                v               = coeffs[6] + coeffs[7]*v
            
            informationsLog.append(['Estimation Position Time', stereoCalcTime,'Duration (s)', (datetime.now() - stereoCalcTime).total_seconds()])
            #print(f"Robot coordinates: X {pntRobo[0][0]}, Y {pntRobo[1][0]}, Z {pntRobo[2][0]}, V {v}")

            ###-Teste_acerto_de_pontos-###
            if not aiTesting:
                pntRobo[0][0] = pntRobo[0][0] + 9 #+ 33.08206562
                #pntRobo[1][0] = pntRobo[0][0] - 10.62173034
                #pntRobo[2][0] = pntRobo[0][0] + 150 #+ 22.01714754
                v = np.degrees(v) #-3.283750123
                if abs(v)>75:
                    v=0
            if aiTesting:
                pntRobo[0][0] = pntRobo[0][0] + 85.05722225 + 3
                #pntRobo[1][0] = pntRobo[0][0] + 2.219887094
                #pntRobo[2][0] = pntRobo[0][0] + 71.28844106
                #v = np.degrees(v) -2.002228625
            ###################""""""

            if 'meanX' in locals():
                meanX = mean([meanX, pntRobo[0][0]])
                meanY = mean([meanY, pntRobo[1][0]])
                meanZ = mean([meanZ, pntRobo[2][0]])
                meanU = mean([meanU, v])
            else:
                meanX = pntRobo[0][0]
                meanY = pntRobo[1][0]
                meanZ = pntRobo[2][0]
                meanU = v

            print(f"Mean robot coordinates: X {round(meanX, 2)}, Y {round(meanY, 2)}, Z {round(meanZ, 2)}, U {round(meanU, 2)} - Press R if ready to move the robot.")
            
            if (cv2.waitKey(1) & 0xFF == ord('r')):
                # Print position...
                conn.sendall(b"OK_py")
                response = conn.recv(1024).decode()
                print(response)
                templen = len(bytes("{:.2f}".format(meanX), 'utf-8'))
                conn.sendall(bytes(str(templen), 'utf-8'))
                conn.sendall(bytes("{:.2f}".format(meanX), 'utf-8'))
                confirmation = conn.recv(1024).decode()
                if confirmation == "x_Recieved":
                    templen = len(bytes("{:.2f}".format(meanY), 'utf-8'))
                    conn.sendall(bytes(str(templen), 'utf-8'))
                    conn.sendall(bytes("{:.2f}".format(meanY), 'utf-8'))
                    confirmation = conn.recv(1024).decode()
                if confirmation == "y_Recieved":
                    templen = len(bytes("{:.2f}".format(meanZ), 'utf-8'))
                    conn.sendall(bytes(str(templen), 'utf-8'))
                    conn.sendall(bytes("{:.2f}".format(meanZ), 'utf-8'))
                    confirmation = conn.recv(1024).decode()
                if confirmation == "z_Recieved":
                    templen = len(bytes("{:.2f}".format(meanU), 'utf-8'))
                    conn.sendall(bytes(str(templen), 'utf-8'))
                    conn.sendall(bytes("{:.2f}".format(meanU), 'utf-8'))
                    confirmation = conn.recv(1024).decode()
                if confirmation == "u_Recieved":
                    print("Point coordinate was successfully transmitted!")

                robotInPos = conn.recv(1024).decode()
                del meanX,meanY,meanZ,meanU

            #print([x, y, z])

        except Exception as error:
            # Capture a frame
            _, frameL = capL.read()
            _, frameR = capR.read()
            _, bwL = cv2.threshold(cv2.cvtColor(frameL, cv2.COLOR_RGB2GRAY), tresholdValueL, 255, cv2.THRESH_BINARY_INV)
            _, bwR = cv2.threshold(cv2.cvtColor(frameR, cv2.COLOR_RGB2GRAY), tresholdValueR, 255, cv2.THRESH_BINARY_INV)
            print("An exception occurred:", error)

        cv2.imshow('Webcams Feed', cv2.hconcat([cv2.resize(frameL, (640, 420)), cv2.resize(frameR, (640, 420))]))
        cv2.imshow('Webcams Feed B&W', cv2.hconcat([cv2.resize(bwL, (640, 420)), cv2.resize(bwR, (640, 420))]))

    capL.release()
    capR.release()

    informationsLog.append(['End Time', datetime.now(),'Duration (s)', (datetime.now() - startTime).total_seconds()])

    if logSaving == True: saveLog(project_directory, informationsLog)