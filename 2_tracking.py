# Hand tracking script
# USAGE: python 2_tracking.py -m 2 --tracker mosse --confidence 0.8 --skip_frames 15 --width 500

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
ap.add_argument("-t", "--tracker", type=str, default="mosse", 
	help="OpenCV object tracker type")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=15,
    help="# of skip frames between detections")
ap.add_argument("-w", "--width", type=int, default=500,
    help="Resize to this width")
args = vars(ap.parse_args())

# Load the trained Tensorflow detection graph
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy_TrackerCSRT.create,
    "kcf": cv2.legacy_TrackerKCF.create,
    "mosse": cv2.legacy_TrackerMOSSE.create
}
print("[INFO] Tracker :", args["tracker"])

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

# initialize trackers
trackers = None

# total number of frames processed thus far
totalFrames = 0

# initialize the FPS throughput estimator
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=args["width"])
    (H, W) = frame.shape[:2]

    # waiting to detect a hand
    status = "Waiting"

    # find the midpoint of hands
    mid_X = 0
    mid_Y = 0

    # Run the object detector
    if totalFrames % args["skip_frames"] == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = cv2.legacy.MultiTracker_create()
        
        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()
        
        # loop over the top 2 detections
        for i in range(2):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < args["confidence"]:
                continue

            # compute the (x, y)-coordinates of the bounding box
            # for the object
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            rect = (startX, startY, endX-startX, endY-startY)

            # Create a new OpenCV tracker and add it to the Multitracker
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            trackers.add(tracker, frame, tuple(rect))
            print("Hand detected :", confidence)

    # grab the updated bounding box coordinates
    (success, boxes) = trackers.update(frame)
    
    # loop over the bounding boxes and draw then on the frame
    for box in boxes:
        status = "Tracking"
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the midpoint
        mid_X += x + w/2
        mid_Y += y + h/2

    # draw the midpoint
    num_hands = len(trackers.getObjects())
    if num_hands > 0:
        mid_X /= num_hands
        mid_Y /= num_hands
        cv2.circle(frame, (int(mid_X), int(mid_Y)), 10, (0, 0, 255), -1)

    # increment the total frames and then update the FPS counter
    totalFrames += 1
    fps.update()
    fps.stop()

    # initialize the set of information we'll be displaying on the frame
    info = [
    ("FPS", "{:.2f}".format(fps.fps())),
    ("Time", "{:.2f}".format(fps.elapsed())),
    ("Status", status),
    ("Hands", num_hands)
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Hand Tracking", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
vs.stop()
cv2.destroyAllWindows()