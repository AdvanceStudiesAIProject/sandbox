import re
from collections import namedtuple
import numpy as np

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

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

def Data_treatment(frame_number, cordinates):
	Input_frame_number = []
	for i in range(0, len(frame_number)):
		Input_frame_number.append(Frame_Number_Reader(frame_number[i]))
	Input_cordonnes_traites = []
	for i in range(0, len(cordinates)):
		Input_cordonnes_traites.append(Cordonnes_extractor(cordinates[i]))
	Input_valeurs = np.column_stack((Input_frame_number, Input_cordonnes_traites))
	return Input_valeurs