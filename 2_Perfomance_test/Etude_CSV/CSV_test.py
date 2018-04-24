import csv
import numpy as np
from imutils.video import WebcamVideoStream
from imutils.video import FileVideoStream
import argparse
import time
#programa usado para escrever os seguintes dados em uma linha.



import cv2
import os
import sys



x = 2
y = 3 
h = 4
w = 5



csvRow = [x, y, h, w]
csvfile = "data.csv"
with open(csvfile, "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(csvRow)