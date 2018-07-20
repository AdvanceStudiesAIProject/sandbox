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
#vs = cv2.VideoCapture("./test_video.mp4")
#vs = cv2.VideoCapture("./gt000%03d.png")#"in%06d" the "gt" it's the fixe part of the image and the "%06d" its the part that will change from a range of 000001 to 999999.
#vs = cv2.VideoCapture("Change_Detection/dataset2014/dataset/baseline/pedestrians/groundtruth/gt%06d.png")
fps = FPS().start()
start_time = time.time()
detection_total_time = 0
n = filecount("./")#Number of files on the folder
#print("number of files:  "+ str(n))
n = n / 100
object_number = 0
frame_number = 1
# loop over the frames
while True:


	signal, frame = vs.read() #o ".read" do VideoCapture retorna duas variaveis, a primeira pode ser usada para verificar se ha uma imagem e a segunda eh a propria image
	if signal is False:#verify if there is signal on the input
		#print("All files analised")
		break

	#frame = imutils.resize(frame, width=320) # resize the image for 320, 240
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame,	0.007843, (300, 300), 127.5) 
	net.setInput(blob)
	detections = net.forward()
	detection_time = time.time() - start_time
	# print("--- %s seconds ---" % (detection_time))
	start_time = time.time()

	detection_total_time = detection_total_time + detection_time
	detection = False
	frame_name = "frame"+str(frame_number)+".jpg"
	object_number = 0
	#cordinates = [None]*2
	cordinates = []
	test = []

	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			# extract the index of the class label from the `detections`, then compute the (x, y)-coordinates of the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			if idx == 15:
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

	for i in range(0, object_number):
		cordinates.append(frame_name)
		cordinates.append(object_number)
		cordinates.append(test[i])
		with open(csvfile, "a") as fp:
			wr = csv.writer(fp, dialect='excel')
			wr.writerow(cordinates)
			cordinates = []

	frame_number = frame_number + 1
	test = []
	cv2.imshow("Frame", frame)# show the output frame
	fps.update()
	sys.stdout.write("Progress: %d%%   \r" % (frame_number/n) )
	sys.stdout.flush()

	key = cv2.waitKey(1) & 0xFF
	# if the "q" key was pressed, break from the loop
	if key == ord("q"):
		break
			

fps.stop()
print("INFOS:")
print("Total time spend on detection : ", detection_total_time)
# print("Total elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))
	

cv2.destroyAllWindows()
#vs.stop()#necessary if using "imutils"     

