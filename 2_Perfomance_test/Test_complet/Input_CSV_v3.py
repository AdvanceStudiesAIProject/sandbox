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
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

#Definition of the rectangles
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def write_CSV(input_to_write, csv_file):
	with open(csv_file, "a") as fp:
		wr = csv.writer(fp, dialect='excel')
		wr.writerow(input_to_write)
		input_to_write = []
	return input_to_write 


def Create_row_for_CSV(row_name, frame_name, object_number, cordinates, 
 detection_time, time_spent_on_frame, detection_total_time, accumulated_elapsed_time):
	row_name.append(frame_name)
	row_name.append(object_number)
	row_name.append(cordinates)
	row_name.append(detection_time)
	row_name.append(time_spent_on_frame)
	row_name.append(detection_total_time)
	row_name.append(accumulated_elapsed_time)
	return row_name


def Cut_selector(input_frame, found_contours):
	(hlimit, wlimit) = frame.shape[:2]

	for c in found_contours:
		if cv2.contourArea(c) > 700:
			(x, y, w, h) = cv2.boundingRect(c)
			rects = Rectangle(x, y, w, h)
			squares.append(rects)

	xfinal = wlimit
	yfinal = hlimit
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
	
	useful_image = frame[yfinal:hfinal, xfinal:wfinal]
	return useful_image, xfinal, yfinal, squares

#----------------------------------------------------------------------


#Initialization of variables
accumulated_elapsed_time = 0
detection_total_time = 0
object_number = 0
frame_number = 0

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-i", "--interface", type=str2bool, nargs='?', 
	const=True, default=False)
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

#Creation of the CSV file
csvfile = "Input_Cordonnes.csv"
open(csvfile, "w")

#Defines the Dataset for reading
vs = cv2.VideoCapture("./in00%04d.jpg")

#Number of files on the folder
n = filecount("./")
n = n / 100

#Create the menu of the CSV file
menu = []

menu = Create_row_for_CSV(menu, '#filename', 'region_count', 'region_shape_attributes', 
	'Detection_time_spent_on_frame (seconds)', 'Total_time_spent_on_frame (seconds)',
	'Accumulated_detection_time_spent_on_video(seconds)', 'Accumulated_elapsed_time (seconds)')

#Write the menu on the CSV file
write_CSV(menu,csvfile)
menu = []

# loop over the frames
while (1):
	detection_time = 0
	start_time = time.time()
	no_human = 0
	signal, frame = vs.read() 

	if signal is False:#verify if there is signal on the input
		break


	#DETECTION--------------------------------------------------------------------------------
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame,	0.007843, (300, 300), 127.5) 
	net.setInput(blob)
	detections = net.forward()
	detection_time = time.time() - start_time

	detection_total_time = detection_total_time + detection_time
	detection = False
	frame_name = "frame"+str(frame_number)+".jpg"
	object_number = 0
	cordinates = []
	test = []
	
	if detections.shape[2] == 0:
		object_number = 0
		time_spent_on_frame = time.time() - start_time
		accumulated_elapsed_time = accumulated_elapsed_time + time_spent_on_frame
		cordinates = Create_row_for_CSV(cordinates, frame_name, object_number, None, 
			detection_time, time_spent_on_frame, detection_total_time, accumulated_elapsed_time)
		write_CSV(cordinates,csvfile)
		cordinates = []
		frame_number = frame_number + 1

		#cv2.imshow("Frame", frame)# show the output frame
		
		sys.stdout.write("Progress: %d%%   \r" % (frame_number/n) )
		sys.stdout.flush()
		continue

	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			# extract the index of the class label from the `detections`, then compute the (x, y)-coordinates of the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			if idx == 15:
				no_human = 1
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
               	# Draw the prediction on the frame
				label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
               		cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				
				object_number = object_number + 1
				test.append("name:rect,x: %d ,y: %d,width: %d, height: %d" %(startX, startY, endX-startX, endY-startY))

	if no_human == 0:
		object_number = 0
		time_spent_on_frame = time.time()-start_time
		accumulated_elapsed_time = accumulated_elapsed_time + time_spent_on_frame

		cordinates = Create_row_for_CSV(cordinates, frame_name, object_number, None, 
			detection_time, time_spent_on_frame, detection_total_time, accumulated_elapsed_time)
		
		write_CSV(cordinates,csvfile)
		cordinates = []

	for i in range(0, object_number):
		time_spent_on_frame = time.time() - start_time
		accumulated_elapsed_time = accumulated_elapsed_time + time_spent_on_frame

		cordinates = Create_row_for_CSV(cordinates, frame_name, object_number, test[i], 
			detection_time, time_spent_on_frame, detection_total_time, accumulated_elapsed_time)
		
		write_CSV(cordinates,csvfile)
		cordinates = []

	frame_number = frame_number + 1
	test = []
	
	sys.stdout.write("Progress: %d%%   \r" % (frame_number/n) )
	sys.stdout.flush()

	key = cv2.waitKey(1) & 0xFF	
	# if the "q" key was pressed, break from the loop
	if key == ord("q"):
		break
	
	if args["interface"] == True:
		cv2.imshow("Frame", frame)# show the output frame

with open('Input_Cordonnes.csv','r') as csvinput:
    with open('Input_Cordonnes_FPS.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('FPS')
        all.append(row)

        for row in reader:
            row.append(frame_number/accumulated_elapsed_time)
            all.append(row)

        writer.writerows(all)

print("INFOS:")
print("Total time spend on detection : ", detection_total_time)
print("Total elapsed time: ", accumulated_elapsed_time)
print("Approx. FPS: ", frame_number/accumulated_elapsed_time)
print("Number of frames: ", frame_number)	

cv2.destroyAllWindows()    