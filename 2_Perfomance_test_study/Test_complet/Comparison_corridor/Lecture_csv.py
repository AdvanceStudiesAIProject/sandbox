from collections import namedtuple
import csv


with open('cordonees.csv') as csvfile2:
    Inputdata2 = csv.reader(csvfile2)
    frame_number_gt = []
    number_of_objects_gt = []
    cordonnes_gt = []

    for row in Inputdata2:
       frame_number_gt.append(row[0])
       number_of_objects_gt.append(row[1])
       cordonnes_gt.append(row[2::])


print(frame_number_gt)
print(number_of_objects_gt)
print(cordonnes_gt)