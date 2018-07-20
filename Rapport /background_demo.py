import cv2
import imutils


vs = cv2.VideoCapture("./frame0.jpg")
vs2 = cv2.VideoCapture("./frame2283.jpg")

# vs = cv2.VideoCapture("./frame3.jpg")
# vs2 = cv2.VideoCapture("./frame1486.jpg")

signal, base = vs.read()
signal, frame = vs2.read()
base = imutils.resize(base, width = 200)
frame = imutils.resize(frame, width = 200)
base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
result = cv2.absdiff(base, frame)
result = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)[1]





while(True):
	cv2.imshow("Base", base)
	cv2.imshow("Input frame", frame)
	cv2.imshow("Diference", result)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break