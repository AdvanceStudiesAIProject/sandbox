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

csvfile = "Input_Cordonnes.csv"
open(csvfile, "w")

vs = cv2.VideoCapture("./in00%04d.jpg")

accumulated_elapsed_time = 0

detection_total_time = 0
n = filecount("./")#Number of files on the folder
n = n / 100
object_number = 0
frame_number = 0
menu = []
menu.append('#filename')
menu.append('region_count')
menu.append('region_shape_attributes')
menu.append('Detection_time_spent_on_frame (seconds)')
menu.append('Total_time_spent_on_frame (seconds)')
menu.append('Accumulated_detection_time_spent_on_video(seconds)')
menu.append('Accumulated_elapsed_time (seconds)')
with open(csvfile, "a") as fp:
	wr = csv.writer(fp, dialect='excel')
	wr.writerow(menu)
	cordinates = []

# loop over the frames
while True:
	start_time = time.time()
	no_human = 0
	signal, frame = vs.read() #o ".read" do VideoCapture retorna duas variaveis, a primeira pode ser usada para verificar se ha uma imagem e a segunda eh a propria image

	if signal is False:#verify if there is signal on the input
		#print("All files analised")
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
	#cordinates = [None]*2
	cordinates = []
	test = []
	
	if detections.shape[2] == 0:
		object_number = 0
		time_spent_on_frame = time.time() - start_time
		accumulated_elapsed_time = accumulated_elapsed_time + time_spent_on_frame
		cordinates.append(frame_name)
		cordinates.append(object_number)
		cordinates.append(None)
		cordinates.append(detection_time)
		cordinates.append(time_spent_on_frame)
		cordinates.append(detection_total_time)
		cordinates.append(accumulated_elapsed_time)
		
		with open(csvfile, "a") as fp:
			wr = csv.writer(fp, dialect='excel')
			wr.writerow(cordinates)
			cordinates = []
		frame_number = frame_number + 1
		cv2.imshow("Frame", frame)# show the output frame
		
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
               	# draw the prediction on the frame
				label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
               		cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				
				object_number = object_number + 1
				# cordinates.append(frame_name)
				# cordinates.append(object_number)
				#cordinates.append("name:rect,x: %d ,y: %d,width: %d, height: %d" %(startX, startY, endX-startX, endY-startY))
				test.append("name:rect,x: %d ,y: %d,width: %d, height: %d" %(startX, startY, endX-startX, endY-startY))

				#with open(csvfile, "a") as fp: #grava os novos dados no mesmo arquivo, mantendo os dados gravados anteriormente. graças ao "a"
				# with open(csvfile, "a") as fp:
				# 	wr = csv.writer(fp, dialect='excel')
				# 	wr.writerow(cordinates)
				# cordinates = []

	if no_human == 0:
		object_number = 0
		time_spent_on_frame = time.time()-start_time
		accumulated_elapsed_time = accumulated_elapsed_time + time_spent_on_frame
		cordinates.append(frame_name)
		cordinates.append(object_number)
		cordinates.append(None)
		cordinates.append(detection_time)
		cordinates.append(time_spent_on_frame)
		cordinates.append(detection_total_time)
		cordinates.append(accumulated_elapsed_time)
		
		with open(csvfile, "a") as fp:
			wr = csv.writer(fp, dialect='excel')
			wr.writerow(cordinates)
			cordinates = [] 

	for i in range(0, object_number):
		time_spent_on_frame = time.time() - start_time
		accumulated_elapsed_time = accumulated_elapsed_time + time_spent_on_frame
		cordinates.append(frame_name)
		cordinates.append(object_number)
		cordinates.append(test[i])
		cordinates.append(detection_time)
		cordinates.append(time_spent_on_frame)
		cordinates.append(detection_total_time)
		cordinates.append(accumulated_elapsed_time)
		
		
		with open(csvfile, "a") as fp:
			wr = csv.writer(fp, dialect='excel')
			wr.writerow(cordinates)
			cordinates = []

	frame_number = frame_number + 1
	test = []
	cv2.imshow("Frame", frame)# show the output frame
	
	sys.stdout.write("Progress: %d%%   \r" % (frame_number/n) )
	sys.stdout.flush()


	key = cv2.waitKey(1) & 0xFF
	# if the "q" key was pressed, break from the loop
	if key == ord("q"):
		break
# FPS = []
# for i in range(0, frame_number):
# 	FPS.append(frame_number/accumulated_elapsed_time)


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
#vs.stop()#necessary if using "imutils"     

