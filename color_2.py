import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import argparse
  
cap = cv2.VideoCapture('aibo_ers7_2.mp4')
colors = {'rojo':(0,0,255),'blue':(117,255,255)}


	
#def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
#    return (knownWidth * focalLength) / perWidth
#KNOWN_DISTANCE = 24.0
#KNOWN_WIDTH = 11.0
#focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH)

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5) 
 
while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #azul
    lower_blue = np.array([120,50,50], dtype=np.uint8)
    upper_blue = np.array([130,255,255], dtype=np.uint8)

    #rojo
    lower_rojo = np.array([170,50,50], dtype=np.uint8)
    upper_rojo = np.array([180,255,255], dtype=np.uint8)  



    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_rojo = cv2.inRange(hsv, lower_rojo, upper_rojo)  


    kernel = np.ones((6,6),np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

    mask_rojo = cv2.morphologyEx(mask_rojo, cv2.MORPH_CLOSE, kernel)
    mask_rojo = cv2.morphologyEx(mask_rojo, cv2.MORPH_OPEN, kernel)
    mask = cv2.add(mask_blue,mask_rojo)

    #cnts = cv2.findContours(mask_blue.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts,hierarchy = cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    for c in cnts:
        if cv2.contourArea(c)<100:
            continue

    orig = frame.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
 
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
    
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    cv2.putText(orig, "{:.1f}mts".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}mts".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)

    #frame_2 = cv2.putText(frame, "Landmark_1".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    #cv2.putText(frame,c + " porteria", (int(tltrX - 15),int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[c],2)

    #mask = cv2.add(mask_blue,mask_rojo)
    cv2.imshow("Image", orig)
    #cv2.imshow("label",frame_2)
    cv2.imshow('mask', mask)
    cv2.imshow('realtime', frame)
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
cv2.destroyAllWindows()
