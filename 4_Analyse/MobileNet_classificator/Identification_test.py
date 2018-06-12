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

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

vs = cv2.VideoCapture("./in00%04d.jpg")
while True:
	signal, frame = vs.read() 
	resized_frame = cv2.resize(frame, (224, 224))
	(h, w) = frame.shape[:2]
	#blob = cv2.dnn.blobFromImage(frame,	0.007843, (w, h), (104, 117, 123)) 
	blob = cv2.dnn.blobFromImage(frame, 0.017, (300, 300), (104, 117, 123), swapRB = True)
	net.setInput(blob)
	prob = net.forward()
	prob = np.squeeze(prob)
	idx = np.argsort(-prob)
	label_names = np.loadtxt('synset.txt', str, delimiter='\t')
	for i in np.arange(0, 1):
		label = idx[i]
		print('%.2f - %s' % (prob[label], label_names[label]))




	cv2.imshow("Frame", resized_frame)
	key = cv2.waitKey(1) & 0xFF	
	# if the "q" key was pressed, break from the loop
	if key == ord("q"):
		break

















######--------------------test with only 1 image-----------------
# #Defines the Dataset for reading
# # vs = cv2.VideoCapture("./in00%04d.jpg")
# vs = cv2.VideoCapture("./homem.jpg")

# signal, frame = vs.read() 
# resized_frame = cv2.resize(frame, (224, 224))
# (h, w) = frame.shape[:2]
# #blob = cv2.dnn.blobFromImage(frame,	0.007843, (w, h), (104, 117, 123)) 
# blob = cv2.dnn.blobFromImage(frame, 0.017, (300, 300), (104, 117, 123), swapRB = True)
# net.setInput(blob)
# prob = net.forward()

# prob = np.squeeze(prob)
# idx = np.argsort(-prob)
# label_names = np.loadtxt('synset.txt', str, delimiter='\t')
# for i in np.arange(0, 1):
# 	label = idx[i]
# 	print('%.2f - %s' % (prob[label], label_names[label]))

# while True:
# 	cv2.imshow("Frame", resized_frame)
# 	key = cv2.waitKey(1) & 0xFF	
# 	# if the "q" key was pressed, break from the loop
# 	if key == ord("q"):
# 		break













































# while True:
# 	signal, frame = vs.read() 

# 	if signal is False:#verify if there is signal on the input
# 		break
# 	(h, w) = frame.shape[:2]
# 	blob = cv2.dnn.blobFromImage(frame,	0.007843, (w, h), (104, 117, 123)) 
# 	net.setInput(blob)
# 	detections = net.forward()
# 	for i in np.arange(0, 5):
# 		# extract the confidence (i.e., probability) associated with the prediction
# 		# confidence = detections[0, 0, i, 2]
# 		# if confidence > args["confidence"]:
# 		# extract the index of the class label from the `detections`, then compute the (x, y)-coordinates of the bounding box for the object
# 		detections = np.squeeze(detections)
# 		idx = np.argsort(-detections)
# 		#print(idx[i])
# 		label_names = np.loadtxt('synset.txt', str, delimiter='\t')
# 		for i in range(5):
# 			label = idx[i]
# 			print('%.2f - %s' % (detections[label], label_names[label]))


# 	key = cv2.waitKey(1) & 0xFF	
# 	# if the "q" key was pressed, break from the loop
# 	if key == ord("q"):
# 		break

# 	cv2.imshow("Frame", frame)




# our CNN requires fixed spatial dimensions for our input image(s)
# so we need to ensure it is resized to 227x227 pixels while
# performing mean subtraction (104, 117, 123) to normalize the input;
# after executing this command our "blob" now has the shape:
# (1, 3, 227, 227)
# blob = cv2.dnn.blobFromImage(image, 1, (227, 227), (104, 117, 123))
 
# # load our serialized model from disk
# print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# # set the blob as input to the network and perform a forward-pass to
# # obtain our output classification
# net.setInput(blob)
# start = time.time()
# preds = net.forward()
# end = time.time()
# print("[INFO] classification took {:.5} seconds".format(end - start))
 
# # sort the indexes of the probabilities in descending order (higher
# # probabilitiy first) and grab the top-5 predictions
# preds = preds.reshape((1, len(classes)))
# idxs = np.argsort(preds[0])[::-1][:5]


# # loop over the top-5 predictions and display them
# for (i, idx) in enumerate(idxs):
# 	# draw the top prediction on the input image
# 	if i == 0:
# 		text = "Label: {}, {:.2f}%".format(classes[idx],
# 			preds[0][idx] * 100)
# 		cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
# 			0.7, (0, 0, 255), 2)
 
# 	# display the predicted label + associated probability to the
# 	# console	
# 	print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
# 		classes[idx], preds[0][idx]))
 
# # display the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)