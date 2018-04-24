#t is also a Gaussian Mixture-based Background/Foreground Segmentation Algorithm. It is based on two papers by Z.Zivkovic,
#"Improved adaptive Gausian mixture model for background subtraction" in 2004 and "Efficient Adaptive Density Estimation
#per Image Pixel for the Task of Background Subtraction" in 2006. One important feature of this algorithm is that it selects
#the appropriate number of gaussian distribution for each pixel. (Remember, in last case, we took a K gaussian distributions
#throughout the algorithm). It provides better adaptibility to varying scenes due illumination changes etc.

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import argparse
import imutils
import numpy as np



cap = cv2.VideoCapture(1)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

while(1):
    (ret, frame) = cap.read()
    fgmask = fgbg.apply(frame)
    #fgmask = cv2.dilate(fgmask, kernel, iterations = 2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('fgmask',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break
    
    cap.release()
    cv2.destroyAllWindows()
    
    key = cv2.waitKey(1) & 0xFF                                       
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break

cap.relese()
cv2.destroyAllWindows()