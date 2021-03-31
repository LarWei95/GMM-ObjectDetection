#!/usr/bin/env python
import rospy
import roslib
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from lgsvl_msgs.msg import Detection2DArray, Detection3DArray
import os
import numpy as np
import datetime as dt
import message_filters

orig_width, orig_height = 1920.0, 1080.0
new_width, new_height = 512.0, 512.0
height_factor = new_height / orig_height
width_crop = orig_width * height_factor - new_width

def resize(x,y,w,h):
	x -= w / 2.0
	y -= h / 2.0
	
	mx = x+w
	my = y+h
	
	y *= height_factor
	my *= height_factor
	
	x = x * height_factor - width_crop/2.0
	mx = mx * height_factor - width_crop/2.0
	
	return x,y,mx,my
	

def ground_truth_threeD(data, id, detection_ids):
	if len(data.detections) == 0:
		return False
	
	filename = str(id)+".jpg"

	string = ""

	for box in data.detections:
		#filename,width,height,class,pos_x,pos_y,pos_z,size_x,size_y,size_z,o_x,o_y,o_z,o_w
		
		if box.id not in detection_ids:
			continue
		
		pos = box.bbox.position
		p = pos.position
		s = box.bbox.size
		o = pos.orientation
		string += "/home/rosvm/data/img/"+filename+",512,512,"+box.label+","+str(p.x)+","+str(p.y)+","+str(p.z)+","+str(s.x)+","+str(s.y)+","+str(s.z)+","+str(o.x)+","+str(o.y)+","+str(o.z)+","+str(o.w)+","+str(box.id)+"\n"
	
	with open("/home/rosvm/data/3D/train.csv","a") as f:
		f.write(string)
		
	return True
	

def ground_truth_twoD(data):
	if len(data.detections) == 0:
		return False, None, None
	
	#print(data)
	
	filename = str(data.header.seq)+".jpg"
	
	string = ""
	detection_ids = set()
	
	c = 0
	for box in data.detections:
		x,y,mx,my = resize(box.bbox.x, box.bbox.y, box.bbox.width, box.bbox.height)
		if mx < 0 or my < 0:
			continue
			
		detection_ids.add(box.id)
			
		c+=1
		string += "/home/rosvm/data/3D/img/"+filename+",512,512," + box.label+","+str(x)+","+str(y)+","+str(mx)+","+str(my)+","+str(box.id)+"\n"
		
	if c == 0:
		return False, None, None
	
	with open("/home/rosvm/data/train.csv","a") as f:
		f.write(string)	
	return True, data.header.seq, detection_ids

	
def main_camera(data, id):
	filename = str(id)+".jpg"
	np_arr = np.fromstring(data.data, np.uint8)
	img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	cv2.imwrite('/home/rosvm/data/img/'+filename, img)
	#check_usage()

def depth_camera(data, id):
	filename = str(id)+".jpg"
	np_arr = np.fromstring(data.data, np.uint8)
	img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	cv2.imwrite('/home/rosvm/data/3D/img/'+filename, img)
	#check_usage_3D()

	
def synchCallback (gt3d, gt2d, mc, dc):
	
	
	
	x, id, detection_ids = ground_truth_twoD(gt2d)
	if x == True:
		# print("Screened!")
		# print("2D:\n"+str(gt2d))
		# print("3D:\n"+str(gt3d))
		
		main_camera(mc, id)
		ground_truth_threeD(gt3d, id, detection_ids)
		depth_camera(dc, id)
	
	
def listener():
	rospy.init_node('ros_listener', anonymous=True)

	gt3d = message_filters.Subscriber("/simulator/ground_truth/3d_detections", Detection3DArray)
	gt2d = message_filters.Subscriber("/simulator/ground_truth/2d_detections", Detection2DArray)
	mc = message_filters.Subscriber("/simulator/main_camera/compressed", CompressedImage)
	dc = message_filters.Subscriber("/simulator/depth_camera/compressed", CompressedImage)
	
	ts = message_filters.ApproximateTimeSynchronizer([gt3d, gt2d, mc, dc], 100, 0.01)
	ts.registerCallback(synchCallback)
	
	rospy.spin()

if __name__ == '__main__':
	with open("/home/rosvm/data/train.csv","w") as f:
		f.write("filename,width,height,class,xmin,ymin,xmax,ymax,id\n")

	with open("/home/rosvm/data/3D/train.csv","w") as f:
		f.write("filename,width,height,class,pos_x,pos_y,pos_z,size_x,size_y,size_z,o_x,o_y,o_z,o_w,id\n")

	os.system("rm /home/rosvm/data/img/*")
	os.system("rm /home/rosvm/data/3D/img/*")
	listener()
