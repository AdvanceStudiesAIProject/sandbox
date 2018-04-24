# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import argparse
import imutils
import numpy as np


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# load our serialized model from disk
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#inititialize the camera and parameters

camera = cv2.VideoCapture(1)
time.sleep(2.0)
#fps = FPS().start()



useful_image = np.matrix('1 2; 3 4')
ref_frame = None

# capture frames from the camera
while True:
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    (grabbed, image) = camera.read()
    image = imutils.resize(image, width=320)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #change the image for a greyscale
    image_gray = cv2.equalizeHist(image_gray)
    #image_gray = cv2.GaussianBlur(image_gray, (21, 21), 0)
    base = np.zeros([240, 320, 3])
    if ref_frame is None:
        ref_frame = image_gray
        continue
    frameDelta = cv2.absdiff(ref_frame, image_gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (test, contour, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contour:
        if cv2.contourArea(c) < 700:#define a minimun area to contour
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        useful_image = image[y:y+h, x:x+w]
        (h, w) = useful_image.shape[:2]
        if h > 25 and w > 25:
            cv2.imshow("test", useful_image)
            # grab the frame dimensions and convert it to a blob
            blob = cv2.dnn.blobFromImage(useful_image,	0.007843, (150, 150), 127.5)
            #pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            for i in np.arange(0, detections.shape[2]):
                cv2.rectangle(frameDelta, (x-50, y-50), (x+w+50, y+h+50), (100, 255, 100), 2)
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]		
                if confidence > args["confidence"]:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    if idx == 15:
                        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # show the frame
    
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Camera", image)
    cv2.imshow("Frame Delta binarisée", thresh)
    #cv2.imshow("Relevant part", relevant)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break