#t is also a Gaussian Mixture-based Background/Foreground Segmentation Algorithm. It is based on two papers by Z.Zivkovic,
#"Improved adaptive Gausian mixture model for background subtraction" in 2004 and "Efficient Adaptive Density Estimation
#per Image Pixel for the Task of Background Subtraction" in 2006. One important feature of this algorithm is that it selects
#the appropriate number of gaussian distribution for each pixel. (Remember, in last case, we took a K gaussian distributions
#throughout the algorithm). It provides better adaptibility to varying scenes due illumination changes etc.

# import the necessary packages
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import time
import cv2
import argparse
import imutils
import numpy as np

cap = cv2.VideoCapture(1)

fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg.setBackgroundRatio(0.2)
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
while(1):
    ret, frame = cap.read() #ret = indica se houve uma capura, frame = frma do video
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame)
    
    cv2.imshow('frame', fgmask)
    
    key = cv2.waitKey(1) & 0xFF                                       
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break

cap.relese()
cv2.destroyAllWindows()