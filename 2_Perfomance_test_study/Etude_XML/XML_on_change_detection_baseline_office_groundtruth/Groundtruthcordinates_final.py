# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# import the necessary packages
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.video import WebcamVideoStream
from imutils.video import FileVideoStream
import numpy as np
import argparse
import time
import cv2
import os
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from xml.dom import minidom
import sys


def prettify(elem):#create the function to organize the xml file
    """Return a pretty-printedXML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent = " ")

def filecount(dir_name):
	return len([f for f in os.listdir(dir_name) if os.path.isfile(f)])

#For image sequences
myfile = open("GT_Cordinates.xml", "w")# create a new XML file with the results
vs = cv2.VideoCapture("./gt%06d.png")#"in%06d" the "gt" it's the fixe part of the image and the "%06d" its the part that will change from a range of 000001 to 999999.
#vs = cv2.VideoCapture("Change_Detection/dataset2014/dataset/baseline/pedestrians/groundtruth/gt%06d.png")
object_number = 1
frame_number = 1
n = filecount("./")
print("number of files:  "+ str(n))
n = n / 100

data = ET.Element("Dataset")

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

	
	for c in contour:
		# if cv2.contourArea(c) < 700:
		# 	continue
		(x, y, w, h) = cv2.boundingRect(c)
		useful_image = frame[y:y+h, x:x+w]
		frame_name = "Frame"+str(frame_number)
		object_name = "Object"+str(object_number)


		 
		if object_number == 1:
			Frame = ET.SubElement(data, frame_name)
			#Frame.set('name',frame_name)
			Object = ET.SubElement(Frame, object_name)
			X = ET.SubElement(Object, 'x')
			Y = ET.SubElement(Object, 'y') 
			H = ET.SubElement(Object, 'h') 
			W = ET.SubElement(Object, 'w')

			X.text = str(x)
			Y.text = str(y)
			H.text = str(h)  
			W.text = str(w)        
			# X.set('text',str(x))
			# Y.set('text',str(y))
			# H.set('text',str(h))
			# W.set('text',str(w))  
			#myfile.write(prettify(Frame)) #write the data on the xml file
		else:
			Object = ET.SubElement(Frame, object_name)
			X = ET.SubElement(Object, 'x')
			Y = ET.SubElement(Object, 'y') 
			H = ET.SubElement(Object, 'h') 
			W = ET.SubElement(Object, 'w')

			X.text = str(x)
			Y.text = str(y)
			H.text = str(h)  
			W.text = str(w)    
			# X.set('text',str(x))
			# Y.set('text',str(y))
			# H.set('text',str(h))
			# W.set('text',str(w))  
			#myfile.write(prettify(Frame)) #write the data on the xml file

		object_number = object_number + 1
		


		# if h > 25 and w > 25:
		# 	cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
  #           #cv2.imshow("test", useful_image)


	
	frame_number = frame_number + 1
	object_number = 1
	
	#cv2.imshow("Frame", frame)# show the output frame
	key = cv2.waitKey(1) & 0xFF

	if frame_number == n:
		break

	# if the "q" key was pressed, break from the loop
	if key == ord("q"):
		break
	
	sys.stdout.write("Progress: %d%%   \r" % (frame_number/n) )
	sys.stdout.flush()

myfile.write(prettify(data)) #write the data on the xml file
# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()#necessary if using "imutils"      