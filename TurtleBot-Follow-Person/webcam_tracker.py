# This script demonstrates the object tracking portion of the TurtleBot-Follow-Person project.
# Running this script with the 'MobileNetSSD' files in the same directory will start the tracker
# and use the webcam for the camera feed.

import cv2
import numpy as np
from pynput import keyboard
import imutils
import rospy
from geometry_msgs.msg import Twist

KILL = False

# function for registering key presses
def on_press(key):
	if key.char == 'q':
		global KILL
		KILL = True
		#print('Killing now')

def main():
	rospy.init_node('nav_control', anonymous=True)
	cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=0)
	cmd = Twist()
	# load the serialized machine learning model for person detection
	net = cv2.dnn.readNetFromCaffe('/home/robot/natkin/src/TurtleBot-Follow-Person/MobileNetSSD_deploy.prototxt.txt', '/home/robot/natkin/src/TurtleBot-Follow-Person/MobileNetSSD_deploy.caffemodel')
	# initialize the list of class labels MobileNet SSD was trained to
	# detect (person is 15), then generate a set of bounding box colors for each class
	global counter
	global count_max
	counter = 0
	count_max = 400
	
	# initialize webcam
	cap = cv2.VideoCapture(0)

	while 1:
		global out
		counter += 1		

		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]
		COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

		# Take first frame
		ret, frame = cap.read()
		(h, w) = frame.shape[:2]

		# video writer setup
		fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #Define the codec and create VideoWriter object
		out = cv2.VideoWriter('webcam_tracker.avi',fourcc, 20.0, (w,h))




		# capture next frame and convert to grayscale
		ret, frame = cap.read()
		frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

		# convert frame to a blob for object detection
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()
		big_area = 0
		big_center = 320
		detected = 0
		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			object_type = detections[0,0,i,1]
			confidence = detections[0, 0, i, 2]
			if object_type == 15 and confidence > 0.2: # execute if confidence is high for person detection
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				label = "{}: {:.2f}%".format('person',confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),[0,0,255], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0,0,255], 2)


				rect_center = int((startX+endX)/2)
				rect_area = (endX-startX)*(endY-startY)
				detected = 1
				if rect_area > big_area: # big_area and big_center are used so that the TurtleBot always tracks the closest person
					big_area = rect_area
					big_center = rect_center			
		if detected:
			#print(rect_center)
			#print(rect_area)
			if big_area > 10000: # Execute if the person is within a reasonable range
				target_center = 320
				target_area = 150000
				# proportional controller for the TurtleBot based on the persons position in the frame and how far they are away
				kr = .002
				w = -kr*(big_center - target_center)
				kt = 0.0000045
				v = -kt*(big_area - target_area)
				maxv = 0.25 # Set a maximum velocity for the Turtlebot
				v = np.max([-maxv, v])
				v = np.min([maxv, v])
				#print(v)
				# Send Velocity command to turtlebot
				cmd.linear.x = v
				cmd.angular.z = w
				cmd_pub.publish(cmd)



		# # write frame to video file and display			
		# out.write(frame)
		# cv2.imshow('Webcam Tracking', frame)
		# Write frames to a video file for up to count_max frames
		if counter < count_max:
			out.write(frame)
			print(counter)
		if counter == count_max:
			out.release()
			print('made video')
		cv2.imshow("Image window", frame)
		cv2.waitKey(3)


			
		
		if KILL:
			print("\nFinished")
			out.release()
			cv2.destroyAllWindows()
			exit()

if __name__ == '__main__':
	listener = keyboard.Listener(on_press=on_press)
	listener.start()
	main()
	exit()
