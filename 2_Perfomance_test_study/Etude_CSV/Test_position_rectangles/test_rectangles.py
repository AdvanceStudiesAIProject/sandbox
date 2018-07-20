#test para ajustar a posicao dos retangulos da detecçao

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

vs = cv2.VideoCapture("./gt000598.png")

#940,2,126,35,100,175,128,36,102,62


signal, frame = vs.read()
frame = imutils.resize(frame, width=320)
#598,1,30,49,49,117
x = 126
y = 35
w = 100
h = 175
# h = 165
# w = 83
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
labely = y - 15 if y - 15 > 15 else y + 15
cv2.putText(frame, "Input", (x, labely),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)




#598,1,78,51,27,116

x = 128
y = 36
w = 102
h = 62
cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
labely = y - 15 if y - 15 > 15 else y + 15
cv2.putText(frame, "GT", (x, labely),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

while True:
	cv2.imshow("Frame", frame)# show the output frame
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break