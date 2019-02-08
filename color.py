import cv2
import numpy as np
import imutils 

#cap = cv2.VideoCapture('aibo_ers7_2.mp4')
#colors = {'rojo':(0,0,255), 'azul':(255,0,0)}

#KNOWN_DISTANCE = 24.0
#KNOWN_WIDTH = 11.0



def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)     

def distance_to_camera(knownWidth, focalLength, perWidth):
# compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth

cap = cv2.VideoCapture('aibo_ers7_3.mp4')
colors = {'rojo':(0,0,255), 'azul':(255,0,0)}

KNOWN_DISTANCE = 24.0
KNOWN_WIDTH = 11.0 


while True:
    _, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
 
    #azul
    lower_blue = np.array([120,50,50], dtype=np.uint8)
    upper_blue = np.array([130,255,255], dtype=np.uint8)

    #rojo
    lower_rojo = np.array([170,50,50], dtype=np.uint8)
    upper_rojo = np.array([180,255,255], dtype=np.uint8)  

    #amarillo
    
    lower_yellow = np.array([5, 110, 200], dtype=np.uint8)
    upper_yellow = np.array([50, 255, 255], dtype=np.uint8)


    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_rojo = cv2.inRange(hsv, lower_rojo, upper_rojo)
    mask_amarillo = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((6,6),np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

    mask_rojo = cv2.morphologyEx(mask_rojo, cv2.MORPH_CLOSE, kernel)
    mask_rojo = cv2.morphologyEx(mask_rojo, cv2.MORPH_OPEN, kernel)

    mask_amarillo = cv2.morphologyEx(mask_amarillo, cv2.MORPH_CLOSE, kernel)
    mask_amarillo = cv2.morphologyEx(mask_amarillo, cv2.MORPH_OPEN, kernel)
    mask = cv2.add(mask_blue,mask_rojo,mask_amarillo)


    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #pixelsPerMetric = None

    for contour in contours:

        area = cv2.contourArea(contour)
        c = max(contours, key=cv2.contourArea)
        (x, y) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        #ret = cv2.minAreaRect(c)
 
        if area > 5000:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
            cv2.putText(frame, " porteria", (300,250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif area < 100:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
            cv2.putText(frame, " Landmark_1", (520,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #cv2.putText(frame, " Landmark_2", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif area < 200:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
            cv2.putText(frame, " Landmark_2", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
    
    focalLength = (c * KNOWN_DISTANCE) / KNOWN_WIDTH
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, center)
    print (inches)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
 
    key = cv2.waitKey(1)
    if key == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
