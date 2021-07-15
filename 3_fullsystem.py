# Tracking program with pan-tilt mechanism
# USAGE: python 3_fullsystem.py -m 1
# Credits: Adrian Rosebrock from PyImageSearch.com
# https://www.pyimagesearch.com/2019/04/01/pan-tilt-face-tracking-with-a-raspberry-pi-and-opencv/

# import the necessary packages
import time
import numpy as np
import cv2
import pantilthat as pth
import argparse
import signal
import sys
import imutils
from imutils.video import VideoStream, FPS
from multiprocessing import Manager
from multiprocessing import Process
from pid import PID

# define the range for the motors
servoRange = (-90, 90)

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy_TrackerCSRT.create,
    "kcf": cv2.legacy_TrackerKCF.create,
    "mosse": cv2.legacy_TrackerMOSSE.create
}

# function to handle keyboard interrupt
def signal_handler(sig, frame):
	# print a status message
	print("[INFO] You pressed `ctrl + c`! Exiting...")

	# disable the servos
	pth.servo_enable(1, False)
	pth.servo_enable(2, False)

	# exit
	sys.exit()

def obj_center(args, objX, objY, centerX, centerY):
	# signal trap to handle keyboard interrupt
	signal.signal(signal.SIGINT, signal_handler)
	
	# Load the trained Tensorflow detection graph
	print("[INFO] Loading model...")
	net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

	print("[INFO] Tracker :", args["tracker"])
	# initialize the video stream and allow the cammera sensor to warmup
	print("[INFO] Starting video stream...")
	if args["mode"] == 0:
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

	# loop indefinitely
	while True:
		# grab the frame from the threaded video stream and resize it
		frame = vs.read()
		frame = imutils.resize(frame, width=args["width"])

		# calculate the center of the frame as this is where we will
		# try to keep the object
		(H, W) = frame.shape[:2]
		centerX.value = W // 2
		centerY.value = H // 2

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

				# compute the (x, y)-coordinates of the bounding box for the
				# object
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
			cv2.circle(frame, (int(mid_X), int(mid_Y)), 5, (0, 0, 255), -1)

			# Update object center
			(objX.value, objY.value) = (int(mid_X), int(mid_Y))
		else:
			(objX.value, objY.value) = (centerX.value, centerY.value)

		# update the FPS counter
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

		# display the frame to the screen
		cv2.imshow("Pan-Tilt Tracking", frame)
		cv2.waitKey(1)

def pid_process(output, p, i, d, objCoord, centerCoord):
	# signal trap to handle keyboard interrupt
	signal.signal(signal.SIGINT, signal_handler)

	# create a PID and initialize it
	p = PID(p.value, i.value, d.value)
	p.initialize()

	# loop indefinitely
	while True:
		# calculate the error
		error = centerCoord.value - objCoord.value

		# update the value
		output.value = p.update(error)

def in_range(val, start, end):
	# determine the input vale is in the supplied range
	return (val >= start and val <= end)

def set_servos(pan, tlt):
	# signal trap to handle keyboard interrupt
	signal.signal(signal.SIGINT, signal_handler)

	# loop indefinitely
	while True:
		# the tilt angle is reversed
		panAngle = pan.value
		tltAngle = -1 * tlt.value

		# if the pan angle is within the range, pan
		if in_range(panAngle, servoRange[0], servoRange[1]):
			pth.pan(panAngle)

		# if the tilt angle is within the range, tilt
		if in_range(tltAngle, servoRange[0], servoRange[1]):
			pth.tilt(tltAngle)

# check to see if this is the main body of execution
if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--mode", type=int, default=1, help="Camera mode")
	ap.add_argument("-t", "--tracker", type=str, default="mosse", 
		help="OpenCV object tracker type")
	ap.add_argument("-c", "--confidence", type=float, default=0.8,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=15,
		help="# of skip frames between detections")
	ap.add_argument("-w", "--width", type=int, default=500,
		help="Resize to this width")
	args = vars(ap.parse_args())

	# start a manager for managing process-safe variables
	with Manager() as manager:
		# enable the servos
		pth.servo_enable(1, True)
		pth.servo_enable(2, True)

		# set integer values for the object center (x, y)-coordinates
		centerX = manager.Value("i", 0)
		centerY = manager.Value("i", 0)

		# set integer values for the object's (x, y)-coordinates
		objX = manager.Value("i", 0)
		objY = manager.Value("i", 0)

		# pan and tilt values will be managed by independed PIDs
		pan = manager.Value("i", 0)
		tlt = manager.Value("i", 0)

		# set PID values for panning
		panP = manager.Value("f", 0.06)
		panI = manager.Value("f", 0.20)
		panD = manager.Value("f", 0.001)

		# set PID values for tilting
		tiltP = manager.Value("f", 0.08)
		tiltI = manager.Value("f", 0.30)
		tiltD = manager.Value("f", 0.003)

		# we have 4 independent processes
		# 1. objectCenter  - finds/localizes the object
		# 2. panning       - PID control loop determines panning angle
		# 3. tilting       - PID control loop determines tilting angle
		# 4. setServos     - drives the servos to proper angles based
		#                    on PID feedback to keep object in center
		processObjectCenter = Process(target=obj_center,
			args=(args, objX, objY, centerX, centerY))
		processPanning = Process(target=pid_process,
			args=(pan, panP, panI, panD, objX, centerX))
		processTilting = Process(target=pid_process,
			args=(tlt, tiltP, tiltI, tiltD, objY, centerY))
		processSetServos = Process(target=set_servos, args=(pan, tlt))

		# start all 4 processes
		processObjectCenter.start()
		processPanning.start()
		processTilting.start()
		processSetServos.start()

		# join all 4 processes
		processObjectCenter.join()
		processPanning.join()
		processTilting.join()
		processSetServos.join()

		# disable the servos
		pth.servo_enable(1, False)
		pth.servo_enable(2, False)