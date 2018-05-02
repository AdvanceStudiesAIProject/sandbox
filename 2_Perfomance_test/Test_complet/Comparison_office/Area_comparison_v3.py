from collections import namedtuple
import csv
import math
import numpy as np
from pandas import DataFrame, read_csv
import re
import matplotlib.pyplot as plt
import pandas as pd 
from pandas import ExcelWriter
from pandas import ExcelFile
import cv2
import imutils
import time

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def Area(rect):
	if rect == None:
		return 0
	else:
		a, b, c, d = rect
		return c * d

def IntersecArea(recta, rectb):  # returns None if rectangles don't intersect  (enter the cornones of initial point and the height ang width)
	if recta == None:
		return 0
	elif rectb == None:
		return 0
	else:
		xa, ya, wa, ha = recta
		a = Rectangle(xa, ya, xa+wa, ya+ha)
		xb, yb, wb, hb = rectb
		b = Rectangle(xb, yb, xb+wb, yb+hb)
		dx = int(min(a.xmax, b.xmax)) - int(max(a.xmin, b.xmin))
		dy = int(min(a.ymax, b.ymax)) - int(max(a.ymin, b.ymin))
		if (dx>=0) and (dy>=0):
			return int(dx*dy)
		else:
			return 0

def UnionAreas(rect1, rect2):
	intersection = IntersecArea(rect1, rect2)
	if intersection == None:
		return Area(rect1) + Area(rect2)
	elif intersection == 0:
		return Area(rect1) + Area(rect2)
	else:	
		somareas = Area(rect1) + Area(rect2)
		return somareas - intersection

def Precision(rect1, rect2):
	porcetages = []
	for i in range(0, len(rect1)):
		for j in range(0, len(rect2)):
			if UnionAreas(rect1[i], rect2[j]) == 0:
				return [100] #case where there is no detectin in both, Input and GT
			porcetages.append((IntersecArea(rect1[i], rect2[j])/UnionAreas(rect1[i], rect2[j]))*100)
	return porcetages

def Frame_Compter(Number_of_objects_column):
	number_of_frames = 0
	frame_conter = True
	l = 0
	while frame_conter == True:
		if l == len(Number_of_objects_column):
			frame_conter = False

		elif int(Number_of_objects_column[l]) > 1:
			number_of_frames = number_of_frames + 1
			l = l + int(Number_of_objects_column[l])	
		else:
			number_of_frames = number_of_frames + 1
			l = l + 1	 
	return number_of_frames

def Frame_Number_Reader(enter_lecture):
	lecture = enter_lecture.split('.')
	match = re.match(r"([a-z]+)([0-9]+)", lecture[0], re.I)
	if match:
	 	items = match.groups()
	return int(items[1])

def Cordonnes_extractor(enter_cordonnes):
	if isinstance(enter_cordonnes, float):
		return None
	elif enter_cordonnes == "{}":
		return None

	else:
		splited = enter_cordonnes.split(',')
		cord_valor = splited[1].split(':')
		X = cord_valor[1]
		cord_valor = splited[2].split(':')
		Y = cord_valor[1]
		cord_valor = splited[3].split(':')
		W = cord_valor[1]
		cord_valor = splited[4].split(':')
		H = cord_valor[1].replace("}", " ")
		rect = Rectangle(int(X), int(Y), int(W), int(H))
		return rect


def Maximun_selector(valeurs_list):
	maximun_index = 0
	maximun = valeurs_list[0]
	if len(valeurs_list) == 1:
		return maximun ,valeurs_list
	for i in range(0, len(valeurs_list)):
		if maximun < valeurs_list[i]:
			maximun = valeurs_list[i]
			maximun_index = i
	new_valeurs_list = []
	for j in range(0, len(valeurs_list)):
		if j == maximun_index:
			continue
		new_valeurs_list.append(valeurs_list[j])  	
	return maximun, new_valeurs_list


vs = cv2.VideoCapture("./in00%4d.jpg")

Inputfile = r'Input_Cordonnes_FPS.csv'
GTfile = r'GT_cordonnes.csv'
Input = pd.read_csv(Inputfile)
GT = pd.read_csv(GTfile)

Input_frame_name = Input['#filename']
GT_frame_name = GT['#filename']

Input_n_objects = Input['region_count']
GT_n_objects = GT['region_count']

cordones_Input = Input['region_shape_attributes']
cordones_GT = GT['region_shape_attributes']


#Traitement de la entree:

Input_frame_number = []
for i in range(0, len(Input_frame_name)):
	Input_frame_number.append(Frame_Number_Reader(Input_frame_name[i]))
Input_cordonnes_traites = []
for i in range(0, len(cordones_Input)):
	Input_cordonnes_traites.append(Cordonnes_extractor(cordones_Input[i]))
Input_valeurs = np.column_stack((Input_frame_number, Input_cordonnes_traites))

#Traitement du GT:

GT_frame_number = []
for i in range(0, len(GT_frame_name)):
	GT_frame_number.append(Frame_Number_Reader(GT_frame_name[i]))
GT_cordonnes_traites = []
for i in range(0, len(cordones_GT)):
	GT_cordonnes_traites.append(Cordonnes_extractor(cordones_GT[i]))
GT_valeurs = np.column_stack((GT_frame_number, GT_cordonnes_traites))

#Comparation
input_objects_per_frame = []
gt_objects_per_frame = []
precision_final = []
final_porcentage = 0
objects_counter = 0
l = 1
CSV_precision = []
#selecteur du frame correct (faire la selection du frame e prend les cordones des objects identifies sur cette frame)
for i in range(1, Frame_Compter(Input_n_objects)):
	for j in range(0, len(GT_valeurs)):
		if GT_valeurs[j][0] == i:
			gt_objects_per_frame.append(GT_valeurs[j][1])
	for k in range(0, len(Input_valeurs)):
		if Input_valeurs[k][0] == i:
			input_objects_per_frame.append(Input_valeurs[k][1])
	# Realise tout les comparison posibles entre les objects du frame
	resultad = Precision(gt_objects_per_frame, input_objects_per_frame)

	#selectione les detections avec le plus grand valeur entre les comparisons des detections
	if len(input_objects_per_frame) >= len(gt_objects_per_frame):
		for i in range(0,len(gt_objects_per_frame)):
			precision_final.append(Maximun_selector(resultad)[0])
			resultad = Maximun_selector(resultad)[1]
	elif len(input_objects_per_frame) < len(gt_objects_per_frame):
		for i in range(0,len(input_objects_per_frame)):
			precision_final.append(Maximun_selector(resultad)[0])
			resultad = Maximun_selector(resultad)[1]


	# if there is a difference between the number of detections on the input and on the GT, add the zeros to the vector of comparisons
	if len(input_objects_per_frame) != len(gt_objects_per_frame):
		for i in range(0, abs(len(input_objects_per_frame) - len(gt_objects_per_frame))):

			precision_final.append(0)
	
	#print(l, precision_final)

	CSV_precision.append(precision_final)

	#make the sum of all the comparisons 
	for i in range(0, len(precision_final)):
		final_porcentage = final_porcentage + precision_final[i]
		objects_counter = objects_counter + 1

	#Show the detections on the respective frame to compare thes results
	signal, frame = vs.read()
	
	if input_objects_per_frame != [None]:
		for i in range(0, len(input_objects_per_frame)):
			x, y, w, h = input_objects_per_frame[i]
			cv2.rectangle(frame, (x, y), (x+w, y+h), 1, 2)
			Ylabel = y - 15 if y - 15 > 15 else y + 15
			cv2.putText(frame, "INPUT", (x, Ylabel),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 2)
	
	if gt_objects_per_frame != [None]:
		for i in range(0, len(gt_objects_per_frame)):
			x, y, w, h = gt_objects_per_frame[i]
			cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
			Ylabel = y - 15 if y - 15 > 15 else y + 15
			cv2.putText(frame, "GT", (x, Ylabel),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
	

	cv2.imshow("Frame", frame)




	#input("Press Enter to continue...")


	resultad = []
	precision_final = []
	gt_objects_per_frame = []
	input_objects_per_frame = []

	l = l + 1



	key = cv2.waitKey(1) & 0xFF
	# if the "q" key was pressed, break from the loop
	if key == ord("q"):
		break

CSV_precision_list = []
#print(CSV_precision)
for i in range(0, len(CSV_precision)):
	for j in range(0, len(CSV_precision[i])):
		CSV_precision_list.append(CSV_precision[i][j])

#print(CSV_precision_list[2005])

#print(len(CSV_precision_list))

i = 0
with open('Input_Cordonnes_FPS.csv','r') as csvinput:
    with open('Resultad_comparison.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('Precision (%)')
        row.append('Mean Precision (%)')
        all.append(row)

        for row in reader:
            row.append(CSV_precision_list[i-1])
            row.append(final_porcentage/objects_counter)
            all.append(row)
            i = i + 1
            

        writer.writerows(all)


print(final_porcentage/objects_counter)






































