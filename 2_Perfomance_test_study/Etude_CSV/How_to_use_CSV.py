import csv
import numpy as np
from imutils.video import WebcamVideoStream
from imutils.video import FileVideoStream
import argparse
import time
import cv2
import os
import sys



#----------------------------------------------------------------------
def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)
#----------------------------------------------------------------------

frame_number = str(65)
X = str(2)
Y = str(3)
H = str(88)
W = str(84552)
obj = str(1)





if __name__ == "__main__":
    data = [frame_number, obj, X, Y, H, W,
    		"1,1,10,12,16,28".split(","),
    		"2,1,12,16,45,82,2,7,8,9,1".split(",")
            ]
    path = "Test.csv"
    csv_writer(data, path)