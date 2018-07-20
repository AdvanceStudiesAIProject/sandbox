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

#----------------------------------------------------------------------
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Input images")
ap.add_argument("-d", "--dataset", type=str2bool, nargs='?', 
	const=True, default=False)
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class

# load our serialized model from disk
print("Loading model...")

#vs = cv2.VideoCapture("./in00%04d.jpg")

vs = cv2.VideoCapture(args["image"] + "/in00%04d.jpg")
if args["dataset"] == True:
	vs = cv2.VideoCapture(args["image"] + "/frame%d.jpg")

#args["prototxt"]
#vs = cv2.VideoCapture("./test.jpg")

fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
fgbg.setBackgroundRatio(1)


def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 15 :
                return True
            elif i==row1-1 and j==row2-100:
                return False








# default_cut = [80,60,80*60]
# initialx = 0
# initialy = 0
# parts = []
# #read and process the image
# ret, frame = vs.read() 
# (imgy, imgx) = frame.shape[:2]
# image_area = imgx*imgy
# #Divide the images 
# number_of_squares = image_area/default_cut[2]
# xsquares = imgx/default_cut[0]
# ysquares = imgy/default_cut[1]
# xsquares = int(xsquares)
# ysquares = int(ysquares)

# image_part = [int(imgx/xsquares), int(imgy/ysquares)]
# print(imgy, imgx)
# print(image_part)
# print(int(number_of_squares))
# for i in range(0, int(number_of_squares)):
# 	parts[i] = frame[initialx:image_part[0] + i*image_part[0], initialy:image_part[1] + i*image_part[1]]
# 	initialx = image_part[0] + i*image_part[0]
# 	initialy = image_part[1] + i*image_part[1]

# while(True):
	
	
# 	key = cv2.waitKey(1) & 0xFF                                       
# 	# if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break





# thresh = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# thresh = cv2.GaussianBlur(thresh,(3,3),0)
# thresh = cv2.threshold(thresh, 50, 255, cv2.THRESH_BINARY)[1]
# #Identifies the contours 
# test,contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)

# LENGTH = len(contours)
# status = np.zeros((LENGTH,1))

# for i,cnt1 in enumerate(contours):
#     x = i    
#     if i != LENGTH-1:
#         for j,cnt2 in enumerate(contours[i+1:]):
#             x = x+1
#             dist = find_if_close(cnt1,cnt2)
#             if dist == True:
#                 val = min(status[i],status[x])
#                 status[x] = status[i] = val
#             else:
#                 if status[x]==status[i]:
#                     status[x] = i+1

# unified = []
# maximum = int(status.max())+1
# for i in range(maximum):
#     pos = np.where(status==i)[0]
#     if pos.size != 0:
#         cont = np.vstack(contours[i] for i in pos)
#         hull = cv2.convexHull(cont)
#         unified.append(hull)

# cv2.drawContours(frame,unified,-1,(0,255,0),2)
# cv2.drawContours(thresh,unified,-1,255,-1)

# while(True):
	
# 	cv2.imshow("contours", frame)
# 	#cv2.imshow("background supression", backgroundsupression)
	


# 	key = cv2.waitKey(1) & 0xFF                                       
# 	# if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break





#### ----------------------------------------Detecttion 15/05/2018 --------------------------------------
dilatation_kernel = np.ones((6,3), np.uint8)
erode_kernel = np.ones((2,2), np.uint8)
opening_kernel = np.ones((2,2), np.uint8)


while True:
	big_contours = []
	big_contours2 = []
	#Read the frame
	ret, frame = vs.read() 
	ret, frame2 = vs.read() 
	frame = cv2.resize(frame, (320,240))
	frame2 = cv2.resize(frame2, (320,240))
	(imgy, imgx) = frame.shape[:2]

	#Aplies the background substraction
	bgs_result = fgbg.apply(frame)
	bgs_result2 = fgbg.apply(frame2)
	#gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
	#Gaussian_result = cv2.GaussianBlur(bgs_result,(5,5),0)
	Opening_result = cv2.morphologyEx(bgs_result, cv2.MORPH_OPEN, opening_kernel)
	# Erode_result = cv2.erode(bgs_result,erode_kernel,iterations = 1)
	Dilatation_result = cv2.dilate(Opening_result, dilatation_kernel, iterations=5)
	Binarization_result = cv2.threshold(Dilatation_result, 50, 255, cv2.THRESH_BINARY)[1]

	Dilatation_result_without_opening = cv2.dilate(bgs_result2, dilatation_kernel, iterations=5)
	Binarization_result_without_opening = cv2.threshold(Dilatation_result_without_opening, 50, 255, cv2.THRESH_BINARY)[1]






	
	#Identifies the contours 
	test,contours,hier = cv2.findContours(Binarization_result,cv2.RETR_EXTERNAL,2)
	for c in range(0,len(contours)):
		if cv2.contourArea(contours[c]) > 800:
			big_contours.append(contours[c])

	for i in range(0, len(big_contours)):
		x, y, h, w = cv2.boundingRect(big_contours[i])

		cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 255, 0), 2)

	test,contours2,hier = cv2.findContours(Binarization_result_without_opening,cv2.RETR_EXTERNAL,2)
	for c in range(0,len(contours2)):
		if cv2.contourArea(contours2[c]) > 800:
			big_contours2.append(contours2[c])

	for i in range(0, len(big_contours2)):
		x2, y2, h2, w2 = cv2.boundingRect(big_contours2[i])

		cv2.rectangle(frame2, (x2, y2), (x2+h2, y2+w2), (0, 255, 0), 2)


	cv2.imshow("Background supression", bgs_result)
	cv2.moveWindow("Background supression", 0, 0)
	cv2.imshow("Opening", Opening_result)
	cv2.moveWindow("Opening", imgx, 0)
	cv2.imshow("Dilatation", Dilatation_result)
	cv2.moveWindow("Dilatation", 2*imgx, 0)
	cv2.imshow("Binarization", Binarization_result)
	cv2.moveWindow("Binarization", 3*imgx, 0)
	cv2.imshow("Contours identification", frame)
	cv2.moveWindow("Contours identification", 4*imgx, 0)


	cv2.imshow("Dilatation_without_opening", Dilatation_result_without_opening)
	cv2.moveWindow("Dilatation_without_opening", 2*imgx, 2*imgy)
	cv2.imshow("Binarization_without_opening", Binarization_result_without_opening)
	cv2.moveWindow("Binarization_without_opening", 3*imgx, 2*imgy)
	cv2.imshow("Contours identification_without_opening", frame2)
	cv2.moveWindow("Contours identification_without_opening", 4*imgx, 2*imgy)
	

	key = cv2.waitKey(1) & 0xFF                                       
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		cv2.destroyAllWindows()
		break

	
###########################################Detection an image using boundingRect##################################################################

# ret, frame = vs.read() 
# #gray = fgbg.apply(frame)
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# thresh = cv2.GaussianBlur(gray,(3,3),0)
# #thresh = cv2.erode(gray,None,iterations = 1)
# thresh = cv2.dilate(thresh, None, iterations=5)
# thresh = cv2.threshold(thresh, 50, 255, cv2.THRESH_BINARY)[1]
# #Identifies the contours 
# test,contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)

# for i in range(0, len(contours)):
# 	x, y, h, w = cv2.boundingRect(contours[i])
# 	print(x, y, h, w)
# 	cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 255, 0), 2)

# while(True):
	
# 	cv2.imshow("contours", frame)
# 	cv2.imshow("test", thresh)
# 	#cv2.imshow("background supression", backgroundsupression)

# 	key = cv2.waitKey(1) & 0xFF                                       
# 	# if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break

###########################################Detection for a sequence of images using covexhull##################################################################

# while True:
# 	ret, frame = vs.read() 
# 	gray = fgbg.apply(frame)
# 	#thresh = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
# 	thresh = cv2.GaussianBlur(gray,(3,3),0)
# 	thresh = cv2.threshold(thresh, 50, 255, cv2.THRESH_BINARY)[1]
# 	#Identifies the contours 
# 	test,contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)

# 	LENGTH = len(contours)
# 	status = np.zeros((LENGTH,1))

# 	for i,cnt1 in enumerate(contours):
# 		x = i    
# 		if i != LENGTH-1:
# 			for j,cnt2 in enumerate(contours[i+1:]):
# 				x = x+1
# 				dist = find_if_close(cnt1,cnt2)
# 				if dist == True:
# 					val = min(status[i],status[x])
# 					status[x] = status[i] = val
# 				else:
# 					if status[x]==status[i]:
# 						status[x] = i+1
# 	if len(status) == 0:
# 		cv2.imshow("contours", frame)
# 		continue

# 	unified = []
# 	maximum = int(status.max())+1
# 	for i in range(maximum):
# 		pos = np.where(status==i)[0]
# 		if pos.size != 0:
# 			cont = np.vstack(contours[i] for i in pos)
# 			hull = cv2.convexHull(cont)
# 			unified.append(hull)

# 	cv2.drawContours(frame,unified,-1,(0,255,0),2)
# 	cv2.drawContours(thresh,unified,-1,255,-1)

# 	cv2.imshow("contours", frame)
# 	#cv2.imshow("background supression", backgroundsupression)
	
# 	key = cv2.waitKey(1) & 0xFF                                       
# 	# if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break

###########################################Detection on one image using covexhull##################################################################
# #read and process the image
# ret, frame = vs.read() 
# thresh = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# thresh = cv2.GaussianBlur(thresh,(3,3),0)
# thresh = cv2.threshold(thresh, 50, 255, cv2.THRESH_BINARY)[1]
# #Identifies the contours 
# test,contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)

# LENGTH = len(contours)
# status = np.zeros((LENGTH,1))

# for i,cnt1 in enumerate(contours):
#     x = i    
#     if i != LENGTH-1:
#         for j,cnt2 in enumerate(contours[i+1:]):
#             x = x+1
#             dist = find_if_close(cnt1,cnt2)
#             if dist == True:
#                 val = min(status[i],status[x])
#                 status[x] = status[i] = val
#             else:
#                 if status[x]==status[i]:
#                     status[x] = i+1

# unified = []
# maximum = int(status.max())+1
# for i in range(maximum):
#     pos = np.where(status==i)[0]
#     if pos.size != 0:
#         cont = np.vstack(contours[i] for i in pos)
#         hull = cv2.convexHull(cont)
#         unified.append(hull)

# cv2.drawContours(frame,unified,-1,(0,255,0),2)
# cv2.drawContours(thresh,unified,-1,255,-1)

# while(True):
	
# 	cv2.imshow("contours", frame)
# 	#cv2.imshow("background supression", backgroundsupression)

# 	key = cv2.waitKey(1) & 0xFF                                       
# 	# if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break

##########################################################################################################################################

# #####TEST#################""
# minLineLength = 1
# maxLineGap = 10

# ret, frame = vs.read() #ret = indica se houve uma capura, frame = frma do video
# #gray = fgbg.apply(frame)
# #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# edges = cv2.Canny(frame,50,150, apertureSize = 3)
# lines = cv2.HoughLinesP(edges,1,np.pi/180,0,minLineLength,maxLineGap)
# print(lines)

# # for i in range(0, len(lines)):
# # 	for x1,y1,x2,y2 in lines[i]:
# # 		cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

# while(True):
	
# 	cv2.imshow("contours", frame)
# 	#cv2.imshow("background supression", backgroundsupression)
# 	cv2.imshow("canys", edges)

# 	key = cv2.waitKey(1) & 0xFF                                       
# 	# if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break










# while(True):
# 	squares = []
# 	ret, frame = vs.read() #ret = indica se houve uma capura, frame = frma do video
# 	(h, w) = frame.shape[:2]
# 	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	backgroundsupression = fgbg.apply(frame)

	# frame = cv2.GaussianBlur(frame,(3,3),0)
	# frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)[1]
	# frame = cv2.erode(frame,None,iterations = 1)
	# frame = cv2.dilate(frame, None, iterations=1)
	# frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# minLineLength = 1
	# maxLineGap = 10
	# lines = cv2.HoughLinesP(frame,1,np.pi/180,100,minLineLength,maxLineGap)
	# for x1,y1,x2,y2 in lines[0]:
	# 	cv2.line(frame,(x1,y1),(x2,y2),(0,255, 0),2)



	# cv2.imshow("contours", frame)
	# cv2.imshow("background supression", backgroundsupression)
	# # cv2.imshow("canys", edges)

	# key = cv2.waitKey(1) & 0xFF                                       
	# # if the `q` key was pressed, break from the loop
	# if key == ord("q"):
	# 	break

	#input("test")

	
	
	
	# (_,contours,_) = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# if len(contours) > 0:
	# 	edges = cv2.Canny(frame,50,150,apertureSize = 3)
	

	# cv2.drawContours(frame, contour_list,  -1, (255,0,0), 2)
	
		
		
		#print(approx)
		





	#frame = cv2.threshold(frame, 25, 255, cv2.THRESH_BINARY)[1]
	#frame = cv2.GaussianBlur(frame,(3,3),0)
	#frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)[1]
	# frame = cv2.erode(frame,kernel,iterations = 1)
	#frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
	#frame = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY)[1]
	#frame = cv2.dilate(frame, kernel, iterations=1)

	# (test, contour, _) = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# for c in contour:
	# 	if cv2.contourArea(c) > 700:
	# 		(x, y, w, h) = cv2.boundingRect(c)
	#input("test")

