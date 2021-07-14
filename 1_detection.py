# Hand detection script
# USAGE: python 1_detection.py --mode 0

# import the necessary packages
import time
import numpy as np
import cv2
import argparse
import imutils
from imutils.video import VideoStream, FPS

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", type=int, default=0, help="Camera mode")
args = vars(ap.parse_args())

# Load the trained Tensorflow detection graph
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] Starting video stream...")
if args["mode"] == 0:
	print("[INFO] Camera mode : Desktop/Laptop")
	vs = VideoStream(src=0).start()
elif args["mode"] == 1:
	print("[INFO] Camera mode : Raspberry Pi")
	vs = VideoStream(usePiCamera=True).start()
else:
	print("[INFO] Camera mode : Jetson Nano")
	vs = VideoStream(src="nvarguscamerasrc ! video/x-raw(memory:NVMM), " \
		"width=(int)1280, height=(int)720,format=(string)NV12, " \
		"framerate=(fraction)6/1 ! nvvidconv ! video/x-raw, " \
		"format=(string)BGRx ! videoconvert ! video/x-raw, " \
		"format=(string)BGR ! appsink").start()
time.sleep(2.0)

# initialize the FPS throughput estimator
fps = FPS().start()

print("[INFO] Press 'q' to exit")
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # Number of hands
    num_hands = 0

    # loop over the top 2 detections
    for i in range(2):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.8:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        num_hands += 1
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the hand along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # update the FPS counter
    fps.update()
    fps.stop()

    # initialize the set of information we'll be displaying on the frame
    info = [
    ("FPS", "{:.2f}".format(fps.fps())),
    ("Time", "{:.2f}".format(fps.elapsed())),
    ("Hands", num_hands)
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, h - ((i * 20) + 20)), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Hand Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
vs.stop()
cv2.destroyAllWindows()