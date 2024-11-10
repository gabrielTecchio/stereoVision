import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
import os

def reorder_points(points):
    # Convert to numpy array if it's not already
    points = np.array(points)
    
    # Find the index of the bottom-left corner
    bottom_left_index = np.argmin(points[:, 0])
    
    # If there are ties for the x-coordinate, choose the one with the largest y-coordinate
    min_x = points[:, 0].min()
    bottom_left_candidates = [i for i in range(len(points)) if points[i, 0] == min_x]
    bottom_left_index = max(bottom_left_candidates, key=lambda i: points[i, 1])
    
    # Reorder the points starting from the bottom-left corner
    reordered_points = np.roll(points, -bottom_left_index, axis=0)
    
    return reordered_points

def find_object(capL, capR, tresholdValueL = 40, tresholdValueR = 120, minArea = 10000, maxArea = 1000000, histogram = False, drawContours = True):
    try:
        _, frameL = capL.read()
        _, frameR = capR.read()
    
    except:
        frameL = cv2.imread(capL)
        frameR = cv2.imread(capR)

        if histogram:
            valsL = frameL.mean(axis=2).flatten()
            valsR = frameR.mean(axis=2).flatten()
            b, bins, patches = plt.hist(valsL, 255, label='Left')
            b, bins, patches = plt.hist(valsR, 255, label='Right')
            plt.legend(loc="upper right")
            plt.xlim([0,255])
            plt.show()

    # Obter as dimensões da imagem
    altura, largura = frameL.shape[:2]

    # Definir partes brancas nas imagens obs: [linhas, colunas]
    #frameL[altura*3//5:, :] = (255, 255, 255)
    #frameR[altura*3//5:, :] = (255, 255, 255)

    #frameL[:, :largura//4] = (255, 255, 255)
    #frameR[:, :largura//4] = (255, 255, 255)
    
    grayL = cv2.cvtColor(frameL, cv2.COLOR_RGB2GRAY)
    _, bwL = cv2.threshold(grayL, tresholdValueL, 255, cv2.THRESH_BINARY_INV) #cv2.THRESH_BINARY
    grayR = cv2.cvtColor(frameR, cv2.COLOR_RGB2GRAY)
    _, bwR = cv2.threshold(grayR, tresholdValueR, 255, cv2.THRESH_BINARY_INV) #cv2.THRESH_BINARY_INV

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    bwLsharpened = cv2.filter2D(bwL, -1, kernel)
    bwRsharpened = cv2.filter2D(bwR, -1, kernel)

    kernel = np.ones((5, 5), np.uint8)
    img_erosionL = cv2.erode(bwLsharpened, kernel, iterations=2) 
    img_erosionR = cv2.erode(bwRsharpened, kernel, iterations=2) 
    img_dilationL = cv2.dilate(img_erosionL, kernel, iterations=2)
    img_dilationR = cv2.dilate(img_erosionR, kernel, iterations=2)
   
    _, cntsL, _ = cv2.findContours(img_dilationL, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    _, cntsR, _ = cv2.findContours(img_dilationR, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    #frame = cv2.hconcat([cv2.resize(img_dilationL, (640, 420)), cv2.resize(img_dilationR, (640, 420))])
    #cv2.imshow('Black & White INSIDE', frame)
    #cv2.waitKey()

    count = 0

    for i, c in enumerate(cntsL):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
            
        # Ignore contours that are too small or too large
        if area < minArea:# or maxArea < area:
            continue

        approx = cv2.approxPolyDP(c, 0.05 * cv2.arcLength(c, True), True)
        boxL = approx.reshape(-1, 2)
        if boxL.size != 8:
            continue

        #boxL = reorder_points(boxL)

        # cv.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        #rect = cv2.minAreaRect(c)
        #boxL = np.int0(cv2.boxPoints(rect))
            
        # Retrieve the key parameters of the rotated bounding box
        #centerL = (int(rect[0][0]),int(rect[0][1]))
        #mid_x = np.mean(boxL[:, 0])
        #mid_y = np.mean(boxL[:, 1])

        # Calculate the moments of the contour
        M = cv2.moments(c)

        centerL = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if drawContours:
            cv2.drawContours(frameL,[boxL],0,(0,255,255),3)
            for corner in (boxL):
                cv2.circle(frameL, corner, 3, (0, 0, 255), -1)
        
        count = count + 1
    
    for i, c in enumerate(cntsR):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
            
        # Ignore contours that are too small or too large
        if area < minArea:# or maxArea < area:
            continue

        # cv.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        #rect = cv2.minAreaRect(c)
        #boxR = cv2.boxPoints(rect)
        #boxR = np.int0(boxR)

        approx =  cv2.approxPolyDP(c, 0.05 * cv2.arcLength(c, True), True)
        boxR = approx.reshape(-1, 2)
        if boxR.size != 8:
            continue
        
        #boxR = reorder_points(boxR)

        # Retrieve the key parameters of the rotated bounding box
        #centerR = (int(rect[0][0]),int(rect[0][1]))
        
        # Calculate the moments of the contour
        M = cv2.moments(c)

        centerR = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if drawContours:
            cv2.drawContours(frameR,[boxR],0,(0,255,255),3)
            for corner in (boxR):
                cv2.circle(frameR, corner, 3, (0, 0, 255), -1)
        
        count = count + 1

    return frameL, centerL, np.array(boxL, dtype=object), frameR, centerR, np.array(boxR, dtype=object), img_dilationL, img_dilationR, count

def find_object_2(capL, capR, tresholdValueL = 50, tresholdValueR = 50, minArea = 10000, maxArea = 1000000, histogram = False, drawContours = True):
    try:
        _, frameL = capL.read()
        _, frameR = capR.read()
    
    except:
        frameL = cv2.imread(capL)
        frameR = cv2.imread(capR)

        if histogram:
            valsL = frameL.mean(axis=2).flatten()
            valsR = frameR.mean(axis=2).flatten()
            b, bins, patches = plt.hist(valsL, 255, label='Left')
            b, bins, patches = plt.hist(valsR, 255, label='Right')
            plt.legend(loc="upper right")
            plt.xlim([0,255])
            plt.show()

    # Obter as dimensões da imagem
    altura, largura = frameL.shape[:2]

    # Definir partes brancas nas imagens obs: [linhas, colunas]
    #frameL[altura*3//5:, :] = (255, 255, 255)
    #frameR[altura*3//5:, :] = (255, 255, 255)

    #frameL[:, :largura//4] = (255, 255, 255)
    #frameR[:, :largura//4] = (255, 255, 255)
    
    grayL = cv2.cvtColor(frameL, cv2.COLOR_RGB2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Blur to smooth the image
    blurredL = cv2.GaussianBlur(grayL, (5, 5), 0)
    blurredR = cv2.GaussianBlur(grayR, (5, 5), 0)

    # Perform edge detection using Canny
    edgesL = cv2.Canny(blurredL, 10, 255)
    edgesR = cv2.Canny(blurredR, 10, 255)

    # Apply Hough Line Transform
    # cv2.HoughLines for standard Hough Line Transform
    linesL = cv2.HoughLines(edgesL, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=10)
    linesR = cv2.HoughLines(edgesR, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=10)

    # Optionally, use cv2.HoughLinesP for probabilistic Hough Line Transform
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=10)

    # Draw the lines on the original image
    boxL=[]
    if linesL is not None:
        for rho, theta in linesL[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            boxL.append([(x1, y1), (x2, y2)])
            cv2.line(frameL, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    boxR=[]
    if linesR is not None:
        for rho, theta in linesR[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            boxR.append([(x1, y1), (x2, y2)])
            cv2.line(frameR, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Optionally, use cv2.HoughLinesP for probabilistic lines
    # if lines is not None:
    #     for x1, y1, x2, y2 in lines[:, 0]:
    #         cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the result
    #cv2.imshow('Detected LinesLeft', frameL)
    #cv2.imshow('Detected LinesRight', frameR)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    centerL=[0]
    centerR=centerL
    count=0

    return frameL, centerL, np.array(boxL, dtype=object), frameR, centerR, np.array(boxR, dtype=object), edgesL, edgesR, count

if __name__ == "__main__":
    folder_path = project_folder = os.path.join(os.getcwd(), 'dataFolder_240803', 'stereoVisionPictures_TrainningSet')
    imgList = os.listdir(folder_path)

    for name in imgList[:]:
        if name.startswith('L') and name.endswith('.jpg'):
            continue
        else:
            imgList.remove(name)

    for i in imgList:
        frameL, centerL, boxL, frameR, centerR, boxR, bwL, bwR, count = find_object(folder_path + '/' + i, 
                                                                            folder_path + '/' + i.replace('L', 'R'),
                                                                            tresholdValueL = 50, tresholdValueR = 50, 
                                                                            minArea = 10000, maxArea = 500000, 
                                                                            histogram = False, drawContours = True)
        print(i)

        
        """buff = 20
        frameL[:max(0, min(boxL[0][1], boxL[1][1])-buff),:] = (255, 255, 255)
        frameL[min(720, max(boxL[2][1], boxL[3][1])+buff):,:] = (255, 255, 255)
        frameL[:,:max(0, min(boxL[2][0], boxL[1][0])-buff)] = (255, 255, 255)
        frameL[:,min(1280, max(boxL[0][0], boxL[3][0])+buff):] = (255, 255, 255)

        frameR[:max(0, min(boxR[0][1], boxR[1][1])-buff),:] = (255, 255, 255)
        frameR[min(720, max(boxR[2][1], boxR[3][1])+buff):,:] = (255, 255, 255)
        frameR[:,:max(0, min(boxR[2][0], boxR[1][0])-buff)] = (255, 255, 255)
        frameR[:,min(1280, max(boxR[0][0], boxR[3][0])+buff):] = (255, 255, 255)"""

        frame = cv2.hconcat([cv2.resize(frameL, (640, 420)), cv2.resize(frameR, (640, 420))])
        cv2.imshow('Frame', frame)
        cv2.waitKey(1000)
        if count != 2:
            input("Press Enter to Continue...")