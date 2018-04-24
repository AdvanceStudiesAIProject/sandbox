from collections import namedtuple
import csv

def Area(a, b, c, d):
	altura = int(d) - int(b)
	largura = int(c) - int(a)
	return altura * largura

def IntersecArea(a, b):  # returns None if rectangles don't intersect
	dx = int(min(a.xmax, b.xmax)) - int(max(a.xmin, b.xmin))
	dy = int(min(a.ymax, b.ymax)) - int(max(a.ymin, b.ymin))
	if (dx>=0) and (dy>=0):
		return int(dx*dy)

# def UnionAreas(a, b, c, d, e, f, g, h):
# 	somareas = Area(a, b, c, d) + Area(e, f, g, h)





CorrectAreaMean = []
TotalMean = 0

#READ INPUT DATA
with open('Input_Cordonnes.csv') as csvfile1:
    Inputdata1 = csv.reader(csvfile1)
    frame_number_input = []
    number_of_objects_input = []
    cordonnes_input = []

    for row in Inputdata1:
        frame_number_input.append(row[0])
        number_of_objects_input.append(row[1])
        cordonnes_input.append(row[2::])


#READ GT DATA
with open('GT_Cordonnes.csv') as csvfile2:
    Inputdata2 = csv.reader(csvfile2)
    frame_number_gt = []
    number_of_objects_gt = []
    cordonnes_gt = []

    for row in Inputdata2:
        frame_number_gt.append(row[0])
        number_of_objects_gt.append(row[1])
        cordonnes_gt.append(row[2::])

n = 0
if len(frame_number_gt) > len(frame_number_input):
	for i in range(0 ,len(frame_number_input)):
		if frame_number_input[0] == frame_number_gt[i]:
			for j in range(1, len(frame_number_input)):

				if frame_number_input[j+n] != frame_number_gt[j+i]:
					if frame_number_input[j+n] > frame_number_gt[j+i]:
						i = i + 1
						CorrectAreaMean.append(0)
					else:
						n = n + 1
						CorrectAreaMean.append(0)

					MaxDetectionNumber = number_of_objects_input[j+n]
					MaxDetectionGT = number_of_objects_gt[j+i]

				if int(number_of_objects_gt[j+i]) > 1 and number_of_objects_input[j+n] == number_of_objects_gt[j+i]:
					
					correction = 0

					if number_of_objects_input[j+n] > MaxDetectionNumber:
						MaxDetectionNumber = number_of_objects_input[j+n]

					if number_of_objects_gt[j+i] > MaxDetectionGT:
						MaxDetectionGT = number_of_objects_gt[j+i]

					for k in range(0, number_of_objects_gt[j+i]):
						correction = 4*k
						Xin = int(cordonnes_input[j+n][0 + correction])
						Yin = int(cordonnes_input[j+n][1 + correction])
						Win = Xin + int(cordonnes_input[j+n][2 + correction])
						Hin = Yin + int(cordonnes_input[j+n][3 + correction])

						Xgt = int(cordonnes_gt[j+i][0 + correction])
						Ygt = int(cordonnes_gt[j+i][1 + correction])
						Wgt = Xgt + int(cordonnes_gt[j+i][2 + correction])
						Hgt = Ygt + int(cordonnes_gt[j+i][3 + correction])
						Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

						RecInput = Rectangle(Xin, Yin, Win, Hin)
						RecGT = Rectangle(Xgt, Ygt, Wgt, Hgt)
						areaintersec = 	IntersecArea(RecInput, RecGT)
						areanormal = Area(Xgt, Ygt, Wgt, Hgt)
						if areaintersec == None:
							#print(frame_number_input[j],"No intersection")
							Correctarea = 0
							CorrectAreaMean.append(Correctarea)
							continue
						Correctarea = (areaintersec/areanormal)*100
						CorrectAreaMean.append(Correctarea)

				elif int(number_of_objects_input[j+n]) > 1 and int(number_of_objects_gt[j+i]) < 2:
					CorrectAreaSelector = []
					correction = 0

					if number_of_objects_input[j+n] > MaxDetectionNumber:
						MaxDetectionNumber = number_of_objects_input[j+n]

					if number_of_objects_gt[j+i] > MaxDetectionGT:
						MaxDetectionGT = number_of_objects_gt[j+i]

					for k in range(0, int(number_of_objects_input[j+n])):
						correction = 4*k
						Xin = int(cordonnes_input[j+n][0 + correction])
						Yin = int(cordonnes_input[j+n][1 + correction])
						Win = Xin + int(cordonnes_input[j+n][2 + correction])
						Hin = Yin + int(cordonnes_input[j+n][3 + correction])

						Xgt = int(cordonnes_gt[j+i][0])
						Ygt = int(cordonnes_gt[j+i][1])
						Wgt = Xgt + int(cordonnes_gt[j+i][2])
						Hgt = Ygt + int(cordonnes_gt[j+i][3])
						Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

						RecInput = Rectangle(Xin, Yin, Win, Hin)
						RecGT = Rectangle(Xgt, Ygt, Wgt, Hgt)
						areaintersec = 	IntersecArea(RecInput, RecGT)
						areanormal = Area(Xgt, Ygt, Wgt, Hgt)
						if areaintersec == None:
							#print(frame_number_input[j],"No intersection")
							Correctarea = 0
							CorrectAreaMean.append(Correctarea)
							continue
						Correctarea = (areaintersec/areanormal)*100
						CorrectAreaSelector.append(Correctarea)
					BetteCorrectArea = CorrectAreaSelector[0]
					for l in range(0, len(CorrectAreaSelector)):
						if CorrectAreaSelector[l] > BetteCorrectArea:
							BetteCorrectArea = CorrectAreaSelector[l]
					CorrectAreaMean.append(BetteCorrectArea)
					for m in range(0, int(len(CorrectAreaSelector)) - 1):
						CorrectAreaMean.append(0)

				elif int(number_of_objects_gt[j+i]) > 1 and int(number_of_objects_input[j+n]) < 2:
					CorrectAreaSelector = []
					correction = 0

					if number_of_objects_input[j+n] > MaxDetectionNumber:
						MaxDetectionNumber = number_of_objects_input[j+n]

					if number_of_objects_gt[j+i] > MaxDetectionGT:
						MaxDetectionGT = number_of_objects_gt[j+i]

					for k in range(0, int(number_of_objects_gt[j+i])):
						correction = 4*k
						# print(frame_number_input[j+n])
						# print(frame_number_gt[j+i])
						# print(correction)
						Xin = int(cordonnes_input[j+n][0])
						Yin = int(cordonnes_input[j+n][1])
						Win = Xin + int(cordonnes_input[j+n][2])
						Hin = Yin + int(cordonnes_input[j+n][3])

						Xgt = int(cordonnes_gt[j+i][0 + correction])
						Ygt = int(cordonnes_gt[j+i][1 + correction])
						Wgt = Xgt + int(cordonnes_gt[j+i][2 + correction])
						Hgt = Ygt + int(cordonnes_gt[j+i][3 + correction])
						Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

						RecInput = Rectangle(Xin, Yin, Win, Hin)
						RecGT = Rectangle(Xgt, Ygt, Wgt, Hgt)
						areaintersec = 	IntersecArea(RecInput, RecGT)
						areanormal = Area(Xgt, Ygt, Wgt, Hgt)
						if areaintersec == None:
							#print(frame_number_input[j],"No intersection")
							Correctarea = 0
							CorrectAreaMean.append(Correctarea)
							continue
						Correctarea = (areaintersec/areanormal)*100
						CorrectAreaSelector.append(Correctarea)
					BetteCorrectArea = CorrectAreaSelector[0]
					for l in range(0, len(CorrectAreaSelector)):
						if CorrectAreaSelector[l] > BetteCorrectArea:
							BetteCorrectArea = CorrectAreaSelector[l]
					CorrectAreaMean.append(BetteCorrectArea)
					for m in range(0, int(len(CorrectAreaSelector)) - 1):
						CorrectAreaMean.append(0)
				
				else:

					if number_of_objects_input[j+n] > MaxDetectionNumber:
						MaxDetectionNumber = number_of_objects_input[j+n]

					if number_of_objects_gt[j+i] > MaxDetectionGT:
						MaxDetectionGT = number_of_objects_gt[j+i]

					Xin = int(cordonnes_input[j+n][0])
					Yin = int(cordonnes_input[j+n][1])
					Win = Xin + int(cordonnes_input[j+n][2])
					Hin = Yin + int(cordonnes_input[j+n][3])

					Xgt = int(cordonnes_gt[j+i][0])
					Ygt = int(cordonnes_gt[j+i][1])
					Wgt = Xgt + int(cordonnes_gt[j+i][2])
					Hgt = Ygt + int(cordonnes_gt[j+i][3])
					# print(frame_number_input[j+n])
					# print(Xin, Yin, Win, Hin)
					# #rectangle area comparison
					Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
					RecInput = Rectangle(Xin, Yin, Win, Hin)
					RecGT = Rectangle(Xgt, Ygt, Wgt, Hgt)
					areaintersec = 	IntersecArea(RecInput, RecGT)
					areanormal = Area(Xgt, Ygt, Wgt, Hgt)
					if areaintersec == None:
						#print(frame_number_input[j],"No intersection")
						Correctarea = 0
						CorrectAreaMean.append(Correctarea)
						continue
					Correctarea = (areaintersec/areanormal)*100
					CorrectAreaMean.append(Correctarea) 
				


			for i in range(0, len(CorrectAreaMean)):
				TotalMean = TotalMean + CorrectAreaMean[i]

			print("The correctness rate is :  ",TotalMean/len(frame_number_input), "%")
			print("The maximun number of simultaneous detection is: ", MaxDetectionNumber)
			print("The maximun number of simultaneous detection on GT is: ", MaxDetectionGT)



else:
	for i in range(0 ,len(frame_number_gt)):
		if frame_number_input[i] == frame_number_gt[0]:
			for j in range(1, len(frame_number_input)):

				if frame_number_input[j+i] != frame_number_gt[j+n]:
					if frame_number_input[j+i] > frame_number_gt[j+n]:
						n = n + 1
						CorrectAreaMean.append(0)
					else:
						i = i + 1
						CorrectAreaMean.append(0)

					MaxDetectionNumber = number_of_objects_input[j+i]
					MaxDetectionGT = number_of_objects_gt[j+n]

				if int(number_of_objects_gt[j+n]) > 1 and number_of_objects_input[j+i] == number_of_objects_gt[j+n]:
					
					correction = 0

					if number_of_objects_input[j+i] > MaxDetectionNumber:
						MaxDetectionNumber = number_of_objects_input[j+i]

					if number_of_objects_gt[j+n] > MaxDetectionGT:
						MaxDetectionGT = number_of_objects_gt[j+n]

					for k in range(0, number_of_objects_gt[j+n]):
						correction = 4*k
						Xin = int(cordonnes_input[j+i][0 + correction])
						Yin = int(cordonnes_input[j+i][1 + correction])
						Win = Xin + int(cordonnes_input[j+i][2 + correction])
						Hin = Yin + int(cordonnes_input[j+i][3 + correction])

						Xgt = int(cordonnes_gt[j+n][0 + correction])
						Ygt = int(cordonnes_gt[j+n][1 + correction])
						Wgt = Xgt + int(cordonnes_gt[j+n][2 + correction])
						Hgt = Ygt + int(cordonnes_gt[j+n][3 + correction])
						Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

						RecInput = Rectangle(Xin, Yin, Win, Hin)
						RecGT = Rectangle(Xgt, Ygt, Wgt, Hgt)
						areaintersec = 	IntersecArea(RecInput, RecGT)
						areanormal = Area(Xgt, Ygt, Wgt, Hgt)
						if areaintersec == None:
							#print(frame_number_input[j],"No intersection")
							Correctarea = 0
							CorrectAreaMean.append(Correctarea)
							continue
						Correctarea = (areaintersec/areanormal)*100
						CorrectAreaMean.append(Correctarea)

				elif int(number_of_objects_input[j+i]) > 1 and int(number_of_objects_gt[j+n]) < 2:
					CorrectAreaSelector = []
					correction = 0

					if number_of_objects_input[j+i] > MaxDetectionNumber:
						MaxDetectionNumber = number_of_objects_input[j+i]

					if number_of_objects_gt[j+n] > MaxDetectionGT:
						MaxDetectionGT = number_of_objects_gt[j+n]

					for k in range(0, int(number_of_objects_input[j+i])):
						correction = 4*k
						Xin = int(cordonnes_input[j+i][0 + correction])
						Yin = int(cordonnes_input[j+i][1 + correction])
						Win = Xin + int(cordonnes_input[j+i][2 + correction])
						Hin = Yin + int(cordonnes_input[j+i][3 + correction])

						Xgt = int(cordonnes_gt[j+n][0])
						Ygt = int(cordonnes_gt[j+n][1])
						Wgt = Xgt + int(cordonnes_gt[j+n][2])
						Hgt = Ygt + int(cordonnes_gt[j+n][3])
						Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

						RecInput = Rectangle(Xin, Yin, Win, Hin)
						RecGT = Rectangle(Xgt, Ygt, Wgt, Hgt)
						areaintersec = 	IntersecArea(RecInput, RecGT)
						areanormal = Area(Xgt, Ygt, Wgt, Hgt)
						if areaintersec == None:
							#print(frame_number_input[j],"No intersection")
							Correctarea = 0
							CorrectAreaMean.append(Correctarea)
							continue
						Correctarea = (areaintersec/areanormal)*100
						CorrectAreaSelector.append(Correctarea)
					BetteCorrectArea = CorrectAreaSelector[0]
					for l in range(0, len(CorrectAreaSelector)):
						if CorrectAreaSelector[l] > BetteCorrectArea:
							BetteCorrectArea = CorrectAreaSelector[l]
					CorrectAreaMean.append(BetteCorrectArea)
					for m in range(0, int(len(CorrectAreaSelector)) - 1):
						CorrectAreaMean.append(0)

				elif int(number_of_objects_gt[j+n]) > 1 and int(number_of_objects_input[j+i]) < 2:
					CorrectAreaSelector = []
					correction = 0

					if number_of_objects_input[j+i] > MaxDetectionNumber:
						MaxDetectionNumber = number_of_objects_input[j+i]

					if number_of_objects_gt[j+n] > MaxDetectionGT:
						MaxDetectionGT = number_of_objects_gt[j+n]

					for k in range(0, int(number_of_objects_gt[j+n])):
						correction = 4*k
						# print(frame_number_input[j+n])
						# print(frame_number_gt[j+i])
						# print(correction)
						Xin = int(cordonnes_input[j+i][0])
						Yin = int(cordonnes_input[j+i][1])
						Win = Xin + int(cordonnes_input[j+i][2])
						Hin = Yin + int(cordonnes_input[j+i][3])

						Xgt = int(cordonnes_gt[j+n][0 + correction])
						Ygt = int(cordonnes_gt[j+n][1 + correction])
						Wgt = Xgt + int(cordonnes_gt[j+n][2 + correction])
						Hgt = Ygt + int(cordonnes_gt[j+n][3 + correction])
						Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

						RecInput = Rectangle(Xin, Yin, Win, Hin)
						RecGT = Rectangle(Xgt, Ygt, Wgt, Hgt)
						areaintersec = 	IntersecArea(RecInput, RecGT)
						areanormal = Area(Xgt, Ygt, Wgt, Hgt)
						if areaintersec == None:
							#print(frame_number_input[j],"No intersection")
							Correctarea = 0
							CorrectAreaMean.append(Correctarea)
							continue
						Correctarea = (areaintersec/areanormal)*100
						CorrectAreaSelector.append(Correctarea)
					BetteCorrectArea = CorrectAreaSelector[0]
					for l in range(0, len(CorrectAreaSelector)):
						if CorrectAreaSelector[l] > BetteCorrectArea:
							BetteCorrectArea = CorrectAreaSelector[l]
					CorrectAreaMean.append(BetteCorrectArea)
					for m in range(0, int(len(CorrectAreaSelector)) - 1):
						CorrectAreaMean.append(0)
				
				else:

					if number_of_objects_input[j+i] > MaxDetectionNumber:
						MaxDetectionNumber = number_of_objects_input[j+i]

					if number_of_objects_gt[j+n] > MaxDetectionGT:
						MaxDetectionGT = number_of_objects_gt[j+n]

					Xin = int(cordonnes_input[j+i][0])
					Yin = int(cordonnes_input[j+i][1])
					Win = Xin + int(cordonnes_input[j+i][2])
					Hin = Yin + int(cordonnes_input[j+i][3])

					Xgt = int(cordonnes_gt[j+n][0])
					Ygt = int(cordonnes_gt[j+n][1])
					Wgt = Xgt + int(cordonnes_gt[j+n][2])
					Hgt = Ygt + int(cordonnes_gt[j+n][3])
					# print(frame_number_input[j+n])
					# print(Xin, Yin, Win, Hin)
					# #rectangle area comparison
					Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
					RecInput = Rectangle(Xin, Yin, Win, Hin)
					RecGT = Rectangle(Xgt, Ygt, Wgt, Hgt)
					areaintersec = 	IntersecArea(RecInput, RecGT)
					areanormal = Area(Xgt, Ygt, Wgt, Hgt)
					if areaintersec == None:
						#print(frame_number_input[j],"No intersection")
						Correctarea = 0
						CorrectAreaMean.append(Correctarea)
						continue
					Correctarea = (areaintersec/areanormal)*100
					CorrectAreaMean.append(Correctarea) 
				


			for i in range(0, len(CorrectAreaMean)):
				TotalMean = TotalMean + CorrectAreaMean[i]

			print("The correctness rate is :  ",TotalMean/len(frame_number_input), "%")
			print("The maximun number of simultaneous detection is: ", MaxDetectionNumber)
			print("The maximun number of simultaneous detection on GT is: ", MaxDetectionGT)
			
#print(cordonnes_input)










