# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import argparse
import imutils
import numpy as np


#inititialize the camera and parameters

camera = cv2.VideoCapture(1)
time.sleep(2.0)
#fps = FPS().start()

useful_image = np.matrix('1 2; 3 4')
ref_frame = None

# capture frames from the camera
while True:
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    (grabbed, frame1) = camera.read()
    frame1 = imutils.resize(frame1, width=320)
    image_gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) #change the image for a greyscale
    image_gray1 = cv2.equalizeHist(image_gray1)
    image_gray1 = cv2.GaussianBlur(image_gray1, (21, 21), 0)
    
    if ref_frame is None:
        ref_frame = image_gray1
        continue
   
    frameDelta = cv2.absdiff(ref_frame, image_gray1)
    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]#cv2.threshold(img, threshold, max value, type of binarization)
    thresh = cv2.dilate(thresh, None, iterations=2)
    (test, contour, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contour:
        if cv2.contourArea(c) < 700:#define a minimun area to contour
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        if x + h > 25 and y + w > 25:
            (grabbed, frame2) = camera.read()
            frame2 = imutils.resize(frame2, width=320)
            ref_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) #change the image for a greyscale
            ref_frame = cv2.equalizeHist(ref_frame)
            ref_frame = cv2.GaussianBlur(ref_frame, (21, 21), 0)
            frameDelta = cv2.absdiff(image_gray1, ref_frame)


            useful_image = frame1[y:y+h, x:x+w]
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (100, 255, 100), 2)
            (h, w) = useful_image.shape[:2]
            cv2.imshow("test", useful_image)
            
            #if x+40+w<320 or y+40+h<240:
                #cv2.imshow("test", useful_image)
                #continue
                # show the frame

    
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Camera", frame1)
    cv2.imshow("Frame Delta binarisée", thresh)
    #cv2.imshow("Relevant part", relevant)
    key = cv2.waitKey(1) & 0xFF                                       
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break