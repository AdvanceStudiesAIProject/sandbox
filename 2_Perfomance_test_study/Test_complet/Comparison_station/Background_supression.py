import csv
import numpy as np
from imutils.video import WebcamVideoStream
from imutils.video import FileVideoStream
import argparse
import time
import cv2
import os
import sys
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from xml.dom import minidom
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
from pandas import ExcelWriter
from pandas import ExcelFile
from collections import namedtuple
#----------------------------------------------------------------------
#Function for write the CSV file
def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

#Function for count the nu,ber of files in the folder
def filecount(dir_name):
	return len([f for f in os.listdir(dir_name) if os.path.isfile(f)])

#----------------------------------------------------------------------
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


vs = cv2.VideoCapture("./in00%04d.jpg")

fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg.setBackgroundRatio(0.01)
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()


while(1):
	squares = []
	ret, frame = vs.read() #ret = indica se houve uma capura, frame = frma do video
	traited_frame = frame
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	fgmask = fgbg.apply(traited_frame)
	thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh,None,iterations = 1)
	thresh = cv2.dilate(thresh, None, iterations=1)
	(test, contour, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for c in contour:
		if cv2.contourArea(c) > 700:
			(x, y, w, h) = cv2.boundingRect(c)
			rects = Rectangle(x, y, w, h)
			squares.append(rects)
			#cv2.rectangle(traited_frame, (x, y), (x+w, y+h), 255, 2)
	
	if len(squares) > 0:


		xfinal = 999
		yfinal = 999
		wfinal = 0
		hfinal = 0									

		for i in range(0, len(squares)):

			x, y, w, h = squares[i]
			w = w + x
			h = h + y
			# print(x, y, w, h)
			if xfinal > x:
				xfinal = x
			if yfinal > y:
				yfinal = y
			if wfinal < w:
				wfinal = w
			if hfinal < h:
				hfinal = h
			
		#cv2.rectangle(traited_frame, (xfinal, yfinal), (wfinal, hfinal), (0, 0, 255), 2)
		useful_image = frame[yfinal:hfinal, xfinal:wfinal]
		#cv2.imshow("DETCTION", useful_image)
		(htest, wtest) = useful_image.shape[:2]
		blob = cv2.dnn.blobFromImage(useful_image,	0.007843, (300, 300), 127.5) 
		net.setInput(blob)
		detections = net.forward()
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the prediction
			confidence = detections[0, 0, i, 2]
			if confidence > args["confidence"]:
				# extract the index of the class label from the `detections`, then compute the (x, y)-coordinates of the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				if idx == 15:
					box = detections[0, 0, i, 3:7] * np.array([wtest, htest, wtest, htest])
					(startX, startY, endX, endY) = box.astype("int")
					
					startX = startX + xfinal
					startY = startY + yfinal
					#print(startX, startY, endX, endY)
					endX = endX + xfinal
					endY = endY + yfinal
               		# draw the prediction on the frame
					label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
					cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
					cv2.imshow("final", frame)

	else:
		
		cv2.destroyAllWindows()
	#input("Press enter to continue")


	key = cv2.waitKey(1) & 0xFF                                       
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
