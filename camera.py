# Camera script
# Usage: python camera.py --mode 0

# import the necessary packages
import time
import cv2
import argparse
import imutils
from imutils.video import VideoStream, FPS

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", type=int, default=0, help="Camera mode")
args = vars(ap.parse_args())

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
    (h, w) = frame.shape[:2]

    # update the FPS counter
    fps.update()
    fps.stop()

    # initialize the set of information we'll be displaying on the frame
    info = [
    ("FPS", "{:.2f}".format(fps.fps())),
    ("Time", "{:.2f}".format(fps.elapsed())),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, h - ((i * 20) + 20)), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
	
# release the video stream and close open windows
vs.stop()
cv2.destroyAllWindows()