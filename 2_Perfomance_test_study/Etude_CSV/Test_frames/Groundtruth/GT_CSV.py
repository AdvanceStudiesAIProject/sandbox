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
import time
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from xml.dom import minidom



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
csvfile = "GT_Cordonnes.csv"
open(csvfile, "w")




vs = cv2.VideoCapture("./gt00%04d.png")
#vs = cv2.VideoCapture("./gt000%03d.png")#"in%06d" the "gt" it's the fixe part of the image and the "%06d" its the part that will change from a range of 000001 to 999999.
#vs = cv2.VideoCapture("Change_Detection/dataset2014/dataset/baseline/pedestrians/groundtruth/gt%06d.png")

n = filecount("./")#Number of files on the folder
print("number of files:  "+ str(n))



n = n / 100


object_number = 0
frame_number = 1
# loop over the frames
while True:

	signal, frame = vs.read() #o ".read" do VideoCapture retorna duas variaveis, a primeira pode ser usada para verificar se ha uma imagem e a segunda eh a propria image
	if signal is False:#verify if there is signal on the input
		#print("All files analised")
		break
	frame = imutils.resize(frame, width=320) # resize the image for 320, 240
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)[1]
	
	(test, contour, _) = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	detection = False
	frame_name = "Frame"+str(frame_number)
	object_number = 0
	cordinates = [None]*2
	cordinates[0] = frame_number
	cordinates[1] = object_number

	i = 0
	for c in contour:
		detection = True
		#if cv2.contourArea(c) < 5:
		(x, y, w, h) = cv2.boundingRect(c)

		if h > 15 and w >15:
			cordinates.append(x)
			i = i + 1
			cordinates.append(y)
			i = i + 1
			cordinates.append(w)
			i = i + 1
			cordinates.append(h)
			i = i + 1
			object_number = object_number + 1

		# if h > 25 and w > 25:
		# 	cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        #   cv2.imshow("test", useful_image)
    
    #with open(csvfile, "a") as fp: #grava os novos dados no mesmo arquivo, mantendo os dados gravados anteriormente. graças ao "a"
	if detection == True:	
		object_name = "Object"+str(object_number)
		cordinates[1] = object_number
		with open(csvfile, "a") as fp:
			wr = csv.writer(fp, dialect='excel')
			wr.writerow(cordinates)
			

	frame_number = frame_number + 1
	
	
	#cv2.imshow("Frame", frame)# show the output frame
	key = cv2.waitKey(1) & 0xFF

	if frame_number == n:
		break

	# if the "q" key was pressed, break from the loop
	if key == ord("q"):
		break
	
	sys.stdout.write("Progress: %d%%   \r" % (frame_number/n) )
	sys.stdout.flush()

	

cv2.destroyAllWindows()
#vs.stop()#necessary if using "imutils"     

