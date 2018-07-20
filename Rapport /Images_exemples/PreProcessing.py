import cv2
import numpy as np

img = cv2.imread('corredor.jpg')
b,g,r = cv2.split(img)
#channel swap
img_incorrect = cv2.merge((r,g,b))
img_correct = cv2.merge((b,g,r))


#sobtraction of average from each channel

r = r - 124.96
g = g - 115.97
b = b - 106.13
img = cv2.merge((b,g,r))

#scaling
img_scaled = cv2.resize(img,(300, 300), interpolation = cv2.INTER_CUBIC)


#Detections


while (1):

	cv2.imshow("Red1", r)
	cv2.imshow("Green1", g)
	cv2.imshow("Blue1", b)
	cv2.imshow("Station", img)

	cv2.imshow("Before swap", img_incorrect)
	cv2.imshow("After swap", img_correct)

	cv2.imshow("after scaling", img_scaled)


	key = cv2.waitKey(1) & 0xFF
	# if the "q" key was pressed, break from the loopq
	if key == ord("q"):
		break
	
