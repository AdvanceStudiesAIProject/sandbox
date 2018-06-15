from collections import namedtuple
import csv
import math
from pandas import DataFrame, read_csv
import pandas as pd 
from pandas import ExcelWriter
from pandas import ExcelFile
import cv2
import time
import argparse
import Precision_module as PM
import Data_Processing as DP

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def Show_comparison(input_objects_per_frame, gt_objects_per_frame):
	signal, frame = vs.read()
	if input_objects_per_frame[0] != None:
		for i in range(0, len(input_objects_per_frame)):
			x, y, w, h = input_objects_per_frame[i]
			cv2.rectangle(frame, (x, y), (x+w, y+h), 1, 2)
			Ylabel = y - 15 if y - 15 > 15 else y + 15
			cv2.putText(frame, "INPUT", (x, Ylabel),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 2)
	if gt_objects_per_frame[0] != None:
		for i in range(0, len(gt_objects_per_frame)):
			x, y, w, h = gt_objects_per_frame[i]
			cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
			Ylabel = y - 15 if y - 15 > 15 else y + 15
			cv2.putText(frame, "GT", (x, Ylabel),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
	return frame

def write_CSV(input_to_write, csv_file):
	with open(csv_file, "a") as fp:
		wr = csv.writer(fp, dialect='excel')
		wr.writerow(input_to_write)
		input_to_write = []
	return input_to_write 

#Atribuition des arguments d'entrée
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
	help="path to data")
ap.add_argument("-g", "--groundtruth", required=True,
	help="path to groundtruth")
ap.add_argument("-i", "--input", required=True,
	help="path to input")
args = vars(ap.parse_args())

#Verification to correct the name of the images on dataset, (must put all the dataset with the same nomenclature)
a = "../Datasets_Test/Corridor/"
b = args["data"]
if a == b:
	vs = cv2.VideoCapture(args["data"] + "frame%d.jpg")
else:
	vs = cv2.VideoCapture(args["data"] + "in00%04d.jpg")

with open(args["input"],'r') as Inputfile:
	Input = pd.read_csv(Inputfile)
with open(args["groundtruth"],'r') as GTfile:
	GT = pd.read_csv(GTfile)

#Reads the Input CSV and save the relevant data
Input_frame_name = Input['#filename']
GT_frame_name = GT['#filename']
Input_n_objects = Input['region_count']
GT_n_objects = GT['region_count']
cordones_Input = Input['region_shape_attributes']
cordones_GT = GT['region_shape_attributes']

#Input tratment, associates the frame number and the cordinates to simplify the calculations.

Input_valeurs = DP.Data_treatment(Input_frame_name, cordones_Input)
GT_valeurs = DP.Data_treatment(GT_frame_name, cordones_GT)

#Comparation
input_objects_per_frame = [None]
gt_objects_per_frame = [None]
precision_final = []
final_porcentage = 0
objects_counter = 0
l = 1
CSV_precision = []
TP = []
FP = []
FN = []
TP_list = []
FP_list = []
FN_list = []
TP_val = 0

# makes sure to take the grounstruth correspondent to the detection 
for i in range(0, DP.Frame_Compter(Input_n_objects)):
	# Will create a list with all the objects on the frame
	for j in range(0, len(GT_valeurs)):
		if GT_valeurs[j][0] == i:
			gt_objects_per_frame.append(GT_valeurs[j][1])
	for k in range(0, len(Input_valeurs)):
		if Input_valeurs[k][0] == i:
			input_objects_per_frame.append(Input_valeurs[k][1])

	#Make all possible matchs between the detection and the groundtruth 
	resultad = PM.Precision(input_objects_per_frame, gt_objects_per_frame)

	#Takes the detections with best precision
	if input_objects_per_frame[0] == None:
		input_objects_per_frame_number = 0
	else:
		input_objects_per_frame_number = len(input_objects_per_frame)
	if gt_objects_per_frame[0] == None:
		gt_objects_per_frame_number = 0
	else:
		gt_objects_per_frame_number = len(gt_objects_per_frame)

	if input_objects_per_frame_number >= gt_objects_per_frame_number:
		for i in range(0,gt_objects_per_frame_number):
			precision_final.append(DP.Maximun_selector(resultad)[0])
			resultad = DP.Maximun_selector(resultad)[1]
	elif input_objects_per_frame_number < gt_objects_per_frame_number:
		for i in range(0,input_objects_per_frame_number):
			precision_final.append(DP.Maximun_selector(resultad)[0])
			resultad = DP.Maximun_selector(resultad)[1]

	if input_objects_per_frame_number != gt_objects_per_frame_number:
		for i in range(0, abs(input_objects_per_frame_number - gt_objects_per_frame_number)):
			precision_final.append(0)	

	if input_objects_per_frame[0] == None:
		#TP_list.append(0)
		TP_val = 0
		number_of_input_detections = 0

	else:
		k = 0
		l = 0
		
		for i in range(0, len(input_objects_per_frame)):
			if precision_final[i] > 40:
				k = k + 1
			else: 
				l = l + 1
		if k > 0:
			TP_list.append(k)
			TP.append(k)
			TP_val = k
		else:
			TP.append(0)
			#TP_list.append(0)
		if l > 0:
			FP_list.append(l)
		number_of_input_detections = len(input_objects_per_frame)

	if gt_objects_per_frame[0] == None:
		number_of_gt_detections = 0
	else:
		number_of_gt_detections = len(gt_objects_per_frame)
	# print("TP" + str(TP_list))
	# print("FP" + str(FP_list))
	# print("FN" + str(FN_list))
	# input("top")

	#FP_list.append(number_of_input_detections - TP_val)
	FN_list.append(number_of_gt_detections - TP_val)
	TP_val = 0

	# if there is a difference between the number of detections on the input and on the GT, add the zeros to the vector of comparisons
	if gt_objects_per_frame[0] == None:
			gt_objects_per_frame_number = 0
	else:
		gt_objects_per_frame_number = len(gt_objects_per_frame)
	
	CSV_precision.append(precision_final)
	
	#make the sum of all the comparisons 
	for i in range(0, len(precision_final)):
		final_porcentage = final_porcentage + precision_final[i]
		objects_counter = objects_counter + 1

	#Show the detections on the respective frame to compare thes results
	cv2.imshow("Frame", Show_comparison(input_objects_per_frame, gt_objects_per_frame))

	resultad = []
	precision_final = []
	gt_objects_per_frame = []
	input_objects_per_frame = []

	key = cv2.waitKey(1) & 0xFF
	# if the "q" key was pressed, break from the loop
	if key == ord("q"):
		break

CSV_precision_list = []
#print(CSV_precision)
for i in range(0, len(CSV_precision)):
	for j in range(0, len(CSV_precision[i])):
		CSV_precision_list.append(CSV_precision[i][j])

TP_total = 0
FP_total = 0
FN_total = 0

for i in range(0, len(TP_list)):
	TP_total = TP_total + TP_list[i]
for i in range(0, len(FP_list)):
	FP_total = FP_total + FP_list[i]
for i in range(0, len(FN_list)):
	FN_total = FN_total + FN_list[i]

TP_mean = TP_total / DP.Frame_Compter(Input_n_objects)
FP_mean = FP_total / DP.Frame_Compter(Input_n_objects)
FN_mean = FN_total / DP.Frame_Compter(Input_n_objects)

i = 0

with open(args["input"],'r') as csvinput:
    with open('Resultad_comparison.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        #row.append('Precision (%)')
        row.append('Mean Precision (%)')
        row.append('TP_MEAN')
        row.append('FN_MEAN')
        row.append('FP_MEAN')
        row.append('TP_TOTAL')
        row.append('FP_TOTAL')
        row.append('FN_TOTAL')
        all.append(row)

        for row in reader:
            #row.append(CSV_precision_list[i-1])
            row.append(final_porcentage/objects_counter)
            row.append(TP_mean)
            row.append(FN_mean)
            row.append(FP_mean)
            row.append(TP_total)
            row.append(FP_total)
            row.append(FN_total)
            all.append(row)
            i = i + 1

        writer.writerows(all)

print(final_porcentage/objects_counter)






































