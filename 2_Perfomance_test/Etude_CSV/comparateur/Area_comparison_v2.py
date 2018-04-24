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
	maximun = valeurs_list[0]
	for i in range(0, len(valeurs_list)):
		if maximun < valeurs_list[i]:
			maximun = valeurs_list[i]
	new_valeurs_list = []
	for j in range(0, len(valeurs_list)):
		
		if valeurs_list[j] == maximun:
			continue
		new_valeurs_list.append(valeurs_list[j])  	
	return maximun, new_valeurs_list



Inputfile = r'Input_Cordonnes.csv'
GTfile = r'GT_cordones.csv'
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


input_objects_per_frame = []
gt_objects_per_frame = []
precision_final = []
for i in range(1, Frame_Compter(Input_n_objects)):
	for j in range(0, len(GT_valeurs)):
		if GT_valeurs[j][0] == i:
			gt_objects_per_frame.append(GT_valeurs[j][1])
	for k in range(0, len(Input_valeurs)):
		if Input_valeurs[k][0] == i:
			input_objects_per_frame.append(Input_valeurs[k][1])
	
	resultad = Precision(gt_objects_per_frame, input_objects_per_frame)
	
	for i in range(0,len(input_objects_per_frame)):
		precision_final.append(Maximun_selector(resultad)[0])
		print(resultad)
		#resultad = Maximun_selector(resultad)[1]

	#print(precision_final)
	#print(precision_final)
	resultad = []
	precision_final = []
	gt_objects_per_frame = []
	input_objects_per_frame = []

	






# test = [2, 5, 8, 25, 1, 0]
# print(Maximun_selector(test))
# test = Maximun_selector(test)[1]
# print(test)















































# splited_Input_cordonnes = cordones_Input.str.split(',', expand = True)
# splited_GT_cordonnes = cordones_GT.str.split(',', expand = True)

# input_temp_cordonnes = []
# gt_temp_cordonnes = []
# precision_porcentage = []
# objects_input = []
# objects_gt = []

# Xin = []
# Yin = []
# Win = []
# Hin = []

# Xgt = []
# Ygt = []
# Wgt = []
# Hgt = []





# # print(splited_GT_cordonnes[1][i] + splited_GT_cordonnes[2][i] + splited_GT_cordonnes[3][i] + splited_GT_cordonnes[4][i])
# # print(splited_Input_cordonnes[1][j] + splited_Input_cordonnes[2][j] + splited_Input_cordonnes[3][j] + splited_Input_cordonnes[4][j])
# i = 0
# k = 0
# l = 0
# reading = True
# while reading == True:

# 	if Input_n_objects[i] == 0:
# 		i = i + 1
# 		Xin.append(None)
# 		Yin.append(None)
# 		Win.append(None)
# 		Hin.append(None)
# 		continue
# 	else:

# 		for k in range(0, Input_n_objects[i]):
# 			input_temp_cordonnes.append(splited_Input_cordonnes[1][i])
# 			input_temp_cordonnes.append(splited_Input_cordonnes[2][i])
# 			input_temp_cordonnes.append(splited_Input_cordonnes[3][i])
# 			input_temp_cordonnes.append(splited_Input_cordonnes[4][i])

# 		for l in range(0, Input_n_objects[i]):
# 			cord_valor = input_temp_cordonnes[0+(4*l)].split(':')
# 			Xin.append(cord_valor[1])
# 			cord_valor = input_temp_cordonnes[1+(4*l)].split(':')
# 			Yin.append(cord_valor[1])
# 			cord_valor = input_temp_cordonnes[2+(4*l)].split(':')
# 			Win.append(cord_valor[1])
# 			cord_valor = input_temp_cordonnes[3+(4*l)].split(':')
# 			Hin.append(cord_valor[1].replace("}", " "))

# 		i = i + Input_n_objects[i]
# 		if i == len(Input_n_objects):
#  			reading = False
#  			continue

# 	input_temp_cordonnes = []


# j = 0
# k = 0
# l = 0
# reading = True
# while reading == True:

# 	if GT_n_objects[j] == 0:
# 		j = j + 1
# 		Xgt.append(None)
# 		Ygt.append(None)
# 		Wgt.append(None)
# 		Hgt.append(None)
# 		continue
# 	else:

# 		for m in range(0, GT_n_objects[j]):
# 			gt_temp_cordonnes.append(splited_GT_cordonnes[1][j])
# 			gt_temp_cordonnes.append(splited_GT_cordonnes[2][j])
# 			gt_temp_cordonnes.append(splited_GT_cordonnes[3][j])
# 			gt_temp_cordonnes.append(splited_GT_cordonnes[4][j])

# 		print(gt_temp_cordonnes)
# 		for n in range(0, GT_n_objects[j]):
# 			cord_valor = gt_temp_cordonnes[0+(4*n)].split(':')
# 			Xgt.append(cord_valor[1])
# 			cord_valor = gt_temp_cordonnes[1+(4*n)].split(':')
# 			Ygt.append(cord_valor[1])
# 			cord_valor = gt_temp_cordonnes[2+(4*n)].split(':')
# 			Wgt.append(cord_valor[1])
# 			cord_valor = gt_temp_cordonnes[3+(4*n)].split(':')
# 			Hgt.append(cord_valor[1].replace("}", " "))

# 		j = j + GT_n_objects[j]
# 		if j == len(GT_n_objects):
#  			reading = False
#  			continue

# 	gt_temp_cordonnes = []



# #fazer arquivo de saida com os dados brutos
# #usar o nome (numero dos frames) para calcular a precisao











































# while reading == True:
# 	if Input_n_objects[i] == GT_n_objects[j]:
# 		if Input_n_objects[i] == 0:# case 1: Input and GT number of objects equals to zero. 
# 			i = i + 1
# 			j = j + 1
# 			Xin.append(None)
# 			Yin.append(None)
# 			Win.append(None)
# 			Hin.append(None)

# 			Xgt.append(None)
# 			Ygt.append(None)
# 			Wgt.append(None)
# 			Hgt.append(None)
# 			continue
# 		else:#case 2: Input and GT same number of objects, but not equal to zero.
# 			k=0
# 			for k in range(0, Input_n_objects[i]):
# 				input_temp_cordonnes.append(splited_Input_cordonnes[1][i])
# 				input_temp_cordonnes.append(splited_Input_cordonnes[2][i])
# 				input_temp_cordonnes.append(splited_Input_cordonnes[3][i])
# 				input_temp_cordonnes.append(splited_Input_cordonnes[4][i])
			
# 			for l in range(0, Input_n_objects[i]):
# 				cord_valor = input_temp_cordonnes[0+(4*l)].split(':')
# 				Xin.append(cord_valor[1])
# 				cord_valor = input_temp_cordonnes[1+(4*l)].split(':')
# 				Yin.append(cord_valor[1])
# 				cord_valor = input_temp_cordonnes[2+(4*l)].split(':')
# 				Win.append(cord_valor[1])
# 				cord_valor = input_temp_cordonnes[3+(4*l)].split(':')
# 				Hin.append(cord_valor[1].replace("}", " "))

# 				i = i + Input_n_objects[i]
# 				if i == len(Input_n_objects):
# 					reading = False
# 					continue
# 			print()
			
			
# 			k=0
# 			for k in range(0, GT_n_objects[j]):
# 				gt_temp_cordonnes.append(splited_GT_cordonnes[1][j])
# 				gt_temp_cordonnes.append(splited_GT_cordonnes[2][j])
# 				gt_temp_cordonnes.append(splited_GT_cordonnes[3][j])
# 				gt_temp_cordonnes.append(splited_GT_cordonnes[4][j])
				
# 			for m in range(0, GT_n_objects[j]):
# 				cord_valor = gt_temp_cordonnes[0+4*m].split(':')
# 				Xgt.append(cord_valor[1])
# 				cord_valor = gt_temp_cordonnes[1+4*m].split(':')
# 				Ygt.append(cord_valor[1])
# 				cord_valor = gt_temp_cordonnes[2+4*m].split(':')
# 				Wgt.append(cord_valor[1])
# 				cord_valor = gt_temp_cordonnes[3+4*m].split(':')
# 				Hgt.append(cord_valor[1].replace("}", " "))

# 				j = j + GT_n_objects[j]
# 				if j == len(GT_n_objects):
# 					reading = False
# 					continue


	
# 	else:#case 3: Input and GT diffe(rents numbers of objects

# 		i = i + 1
# 		j = j + 1
# 		print("ERRO")

# 	input_temp_cordonnes = []
# 	gt_temp_cordonnes = []


# while True:









# 	if Input_n_objects[i] == GT_n_objects[j]:

# 		if Input_n_objects[i] == 0:
# 			i = i + 1
# 			j = j + 1
# 			continue
# 		else:
# 			k=0
# 			for k in range(0, Input_n_objects[i]):
# 				input_temp_cordonnes.append(splited_Input_cordonnes[1][j])
# 				input_temp_cordonnes.append(splited_Input_cordonnes[2][j])
# 				input_temp_cordonnes.append(splited_Input_cordonnes[3][j])
# 				input_temp_cordonnes.append(splited_Input_cordonnes[4][j])
# 				i = i + 1
			
# 			k=0
# 			for k in range(0, GT_n_objects[j]):
# 				gt_temp_cordonnes.append(splited_GT_cordonnes[1][i])
# 				gt_temp_cordonnes.append(splited_GT_cordonnes[2][i])
# 				gt_temp_cordonnes.append(splited_GT_cordonnes[3][i])
# 				gt_temp_cordonnes.append(splited_GT_cordonnes[4][i])
# 				j = j + 1
			
# 			cord_valor = input_temp_cordonnes[0].split(':')
# 			Xin = cord_valor[1]
# 			cord_valor = input_temp_cordonnes[1].split(':')
# 			Yin = cord_valor[1]
# 			cord_valor = input_temp_cordonnes[2].split(':')
# 			Win = cord_valor[1]
# 			cord_valor = input_temp_cordonnes[3].split(':')
# 			Hin = cord_valor[1].replace("}", " ")
# 			#print(Xin, Yin, Win, Hin)

# 			cord_valor = gt_temp_cordonnes[0].split(':')
# 			Xgt = cord_valor[1]
# 			cord_valor = gt_temp_cordonnes[1].split(':')
# 			Ygt = cord_valor[1]
# 			cord_valor = gt_temp_cordonnes[2].split(':')
# 			Wgt = cord_valor[1]
# 			cord_valor = gt_temp_cordonnes[3].split(':')
# 			Hgt = cord_valor[1].replace("}", " ")


# 			#print(Xin, Yin, Win,Hin)


# 			# print(int(Xin)+int(Yin)+int(Win)+int(Hin))
# 			# print(int(Xgt)+int(Ygt)+int(Wgt)+int(Hgt))

			

# 			rectA = Rectangle(int(Xin), int(Yin), int(Win), int(Hin))
# 			rectB = Rectangle(int(Xgt), int(Ygt), int(Wgt), int(Hgt))

# 			# print("Area rect A:  ", Area(rectA))
# 			# print("Area rect B:  ", Area(rectB))
# 			# print("Intersection rect A and B:  ", IntersecArea(rectA, rectB))
# 			# print("Union rect A and B:  ", UnionAreas(rectA, rectB))

# 			# print("Precision between rect A and B:  ", Precision(rectA, rectB), "%")

# 			precision_porcentage.append(Precision(rectA, rectB))






# 		i = i + Input_n_objects[i]
# 		j = j + GT_n_objects[j]

# 	else:
# 		i = i + 1
# 		j = j + 1



	# for i in range(0, len(region_shape_column)):
	# 	if n_objects[i] == 0:
	# 		continue
	# 	j = 1
	# 	cord_valor = cords[j][i].split(':')
	# 	Xin = cord_valor[1]
	# 	j = j + 1
	# 	cord_valor = cords[j][i].split(':')
	# 	Yin = cord_valor[1]
	# 	j = j + 1
	# 	cord_valor = cords[j][i].split(':')
	# 	Win = cord_valor[1]
	# 	j = j + 1
	# 	cord_valor = cords[j][i].split(':')
	# 	Hin = cord_valor[1]
	# 	j = j + 1


#cord_valor = cords[1][0].split(':')





# print(read[1])
# #print(len(read))
# #print(cords[1][0])
#print(Xin, " ", Yin, " ", Win, " ", Hin)



#####################################################################################################
############################## Precision porcentage calculation #####################################
#####################################################################################################

# def Area(rect):
# 	a, b, c, d = rect
# 	altura = int(d) - int(b)
# 	largura = int(c) - int(a)
# 	return altura * largura

# def IntersecArea(a, b):  # returns None if rectangles don't intersect
# 	dx = int(min(a.xmax, b.xmax)) - int(max(a.xmin, b.xmin))
# 	dy = int(min(a.ymax, b.ymax)) - int(max(a.ymin, b.ymin))
# 	if (dx>=0) and (dy>=0):
# 		return int(dx*dy)

# def UnionAreas(rect1, rect2):
# 	somareas = Area(rect1) + Area(rect2)
# 	return somareas - IntersecArea(rect1, rect2)

# def Precision(rect1, rect2):
# 	return (IntersecArea(rect1, rect2)/UnionAreas(rect1, rect2))*100

# Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

# Xin = 0
# Yin = 0
# Win = 50
# Hin = 50

# Xgt = 1
# Ygt = 1
# Wgt = 3
# Hgt = 3

# rectA = Rectangle(Xin, Yin, Win, Hin)
# rectB = Rectangle(Xgt, Ygt, Wgt, Hgt)

# print("Area rect A:  ", Area(rectA))
# print("Area rect A:  ", Area(rectB))
# print("Intersection rect A and B:  ", IntersecArea(rectA, rectB))
# print("Union rect A and B:  ", UnionAreas(rectA, rectB))
# print("Precision between rect A and B:  ", Precision(rectA, rectB), "%")




