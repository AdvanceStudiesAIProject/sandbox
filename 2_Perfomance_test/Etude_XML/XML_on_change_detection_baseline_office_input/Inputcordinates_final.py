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

#Load the data 
vs = cv2.VideoCapture("./in%06d.jpg")#"in%06d" the "in" it's the fixe part of the image and the "%06d" its the part that will change from a range of 000001 to 999999.
print("Loading video...")
fps = FPS().start()




#For image sequences
myfile = open("Input_Cordinates.xml", "w")# create a new XML file with the results
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
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame,	0.007843, (300, 300), 127.5) # FPS melhorou consideravelmente quando diminui as dimensoes do blob
	# pass the blob through the network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward()

	

	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		frame_name = "frame"+str(frame_number)
		object_name = "object"+str(object_number)


		# filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
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

				if object_number == 1:
					Frame = ET.SubElement(data, frame_name)
					Object = ET.SubElement(Frame, object_name)
					X = ET.SubElement(Object, 'x')
					Y = ET.SubElement(Object, 'y') 
					H = ET.SubElement(Object, 'h') 
					W = ET.SubElement(Object, 'w')

					X.text = str(startX)
					Y.text = str(startY)
					H.text = str(endY)  
					W.text = str(endX)  
			
				else:
					Object = ET.SubElement(Frame, object_name)
					X = ET.SubElement(Object, 'x')
					Y = ET.SubElement(Object, 'y') 
					H = ET.SubElement(Object, 'h') 
					W = ET.SubElement(Object, 'w')

					X.text = str(startX)
					Y.text = str(startY)
					H.text = str(endY)  
					W.text = str(endX) 
			

					object_number = object_number + 1

	
	frame_number = frame_number + 1
	object_number = 1

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the "q" key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

	
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

# stop the timer and display FPS information
fps.stop()
print("INFOS:")
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))


# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()#necessary if using "imutils"      