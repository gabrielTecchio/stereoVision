import os
import cv2
import time
import socket
from datetime import datetime

def capture_and_save_image(capL, capR, output_folder):
    HOST = "0.0.0.0"  # Standard loopback interface address (localhost)
    PORT = 5000  # Port to listen on (non-privileged ports are > 1023)

    #HOST = "192.168.0.110"  # Standard loopback interface address (localhost)
    #PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    print(f"Connected by {addr}")
    test = conn.recv(1024).decode()
    print(test)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        f = open(os.path.join(output_folder, 'data.txt'), 'x')
        f.close()

    # Creating usable variables
    index = 0

    file = open(os.path.join(output_folder, 'data.txt'), "a")
    file.write("ID,X,Y,Z,U\n")
    #input("press any key to continue...")
    conn.sendall(b"OK")

    while True:
        data = conn.recv(1024).decode().split(",")
        time.sleep(1) # Sleep for 10 seconds
        print("Picture :", index)

        # Capture the frames and display them in separate windows
        _, frameLeft = capL.read()
        _, frameRight = capR.read()
        
        frameLeft_copy = cv2.resize(frameLeft, (640, 420))
        frameRight_copy = cv2.resize(frameRight, (640, 420))
        cv2.imshow('Webcams Feed', cv2.hconcat([frameLeft_copy, frameRight_copy]))

        if data[0] == 'In position' : #f.read() == "1":
            # Generate a unique filename based on the index number
            filenameLeft = f"L_{index}.jpg"
            filenameRight = f"R_{index}.jpg"
            filepathLeft = os.path.join(output_folder, filenameLeft)
            filepathRight = os.path.join(output_folder, filenameRight)
            
            # Save the captured frame as an image
            cv2.imwrite(filepathLeft, frameLeft)
            cv2.imwrite(filepathRight, frameRight)
            file.write(str(index-1)+","+data[1]+","+data[2]+","+data[3]+","+data[4]+"\n")
            
            index = index + 1
            conn.sendall(b"Done")
        else:
            pass

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            # On keyboard interrupt or reaching the defined number of pictures, close all windows and stop the loop
            cv2.destroyAllWindows()
            file.close()
            break

if __name__ == "__main__":
    # Starting both webcams
    capL = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capL.set(cv2.CAP_PROP_EXPOSURE, -5.0)
    capL.set(cv2.CAP_PROP_SETTINGS, 1)
    
    capR = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capR.set(cv2.CAP_PROP_EXPOSURE, -5.0)
    capR.set(cv2.CAP_PROP_SETTINGS, 1)

    while True:
        # Capture a frame
        _, frameL = capL.read()
        _, frameR = capR.read()
 
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            # On keyboard interrupt or reaching the defined number of pictures, close all windows and stop the loop
            cv2.destroyAllWindows()
            break

        _, gframeL = cv2.threshold(cv2.cvtColor(frameL, cv2.COLOR_RGB2GRAY), 50, 255, cv2.THRESH_BINARY_INV) #cv2.THRESH_BINARY
        _, gframeR = cv2.threshold(cv2.cvtColor(frameR, cv2.COLOR_RGB2GRAY), 50, 255, cv2.THRESH_BINARY_INV) #cv2.THRESH_BINARY

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
    
    output_folder = os.path.join(os.getcwd(), 'dataFolder_' + str(datetime.now().strftime("%y%m%d")), 'stereoVisionPictures2')
    capture_and_save_image(capL, capR, output_folder)

    # Release the webcam
    capL.release()
    capR.release()