# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import argparse
import imutils
import numpy as np

#inititialize the camera and parameters

camera =cv2.VideoCapture(1)
time.sleep(0.25)


#if using pi camera:
#camera = PiCamera()
#rawCapture = PiRGBArray(camera, size=(320, 240))
#time.sleep(0.1)# warmup the camera
#camera.resolution = (320,240)
#camera.framerate = 32
#camera.brightness = 50# values between 0 and 100
#camera.sharpness = 0
#camera.contrast = 0
#camera.saturation = 0
#camera.iso = 800
#camera.video_stabilization = False
#camera.exposure_compensation = 0
#camera.exposure_mode = 'auto'
#camera.meter_mode = 'average'
#camera.awb_mode = 'auto'
#camera.image_effect = 'none'
#camera.color_effects = None
#camera.rotation = 0
#camera.hflip = False
#camera.vflip = False
#camera.crop = (0.0, 0.0, 1.0, 1.0)


ref_frame = None
test = True
# capture frames from the camera
while True:
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	(grabbed, image) = camera.read()
	image = imutils.resize(image, width=320)
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #change the image for a greyscale
	image_gray = cv2.GaussianBlur(image_gray, (21, 21), 0)
	base = np.zeros([240,320,3])
	if ref_frame is None:
            ref_frame = image_gray
            continue
        
	frameDelta = cv2.absdiff(ref_frame, image_gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	
	(test, contour, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	for c in contour:
            if cv2.contourArea(c) < 700:#define a minimun area to contour
                continue

            
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x-50, y-50), (x+w+50, y+h+50), (0, 255, 0), 2)
                    
            
	# show the frame
	cv2.imshow("Frame Delta", frameDelta)
	cv2.imshow("Camera", image)
	cv2.imshow("Frame Delta binarisée", thresh)
	cv2.imshow("relevant", base)
        
	#cv2.imshow("Relevant part", relevant)
	key = cv2.waitKey(1) & 0xFF
        
 
	
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
