import argparse
import os
import sys
from pathlib import Path
import time
import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from geometry_msgs.msg import PointStamped
from collections import OrderedDict

class Timing:

	def __init__(self, last_itrs_means=0):

		# stamp_names: List of strings
		self.timing_details = OrderedDict()
		self.timing_details_total = OrderedDict()
		self.timing_details_last_itrs_means = OrderedDict()

		self.total_iters = 0
		self.last_itrs_means = last_itrs_means

		self.start_time = None
		self.end_time = None

		self.is_first_loop = True

		self.fps_curr = 0.0
		self.fps_mean = 0.0


	def start_loop(self):
		self.timing_details['start_time'] = time.time()

	def end_loop(self, print_report=False):
		self.timing_details['end_time'] = time.time()
		self.total_iters = self.total_iters + 1

		interval_dict= OrderedDict()
		stamp_names = list(self.timing_details.keys())

		for i in range(1, len(stamp_names)):
			interval = self.timing_details[stamp_names[i]]-self.timing_details[stamp_names[i-1]]
			interval_dict[stamp_names[i]] = interval
		interval_dict['loop_time'] = self.timing_details['end_time'] - self.timing_details['start_time']
		interval_dict['lps'] = 1/interval_dict['loop_time']

		if self.is_first_loop:
			self.is_first_loop = False
			self.timing_details_total = interval_dict.copy()
		else:
			for key, value in interval_dict.items():
				self.timing_details_total[key] = self.timing_details_total[key] + value

		if print_report:
			print('\n--------- Timing Report Start ------------')
			print(f'')
			print(f'iter: {self.total_iters}\t\tcurr (ms)\t|\tmean (ms)\t|\ttotal (s)')
			print('-------------------------------------------------------')
			for key, val in interval_dict.items():
				total_val = self.timing_details_total[key]
				mean_val = total_val/self.total_iters

				space_str = '\t\t'
				milisec = 1000
				if key=='lps':
					milisec=1
					space_str = '\t\t\t'

				print('{}:{}{:.2f}\t\t|\t{:.2f}\t\t|\t{:.2f}'.format(key, space_str, val*milisec, mean_val*milisec, total_val))
				
			print('\n--------- Timing Report End ------------')

		self.fps_curr = int(interval_dict['lps'])
		self.fps_mean = int(self.timing_details_total['lps']/self.total_iters)

	def stamp(self, name):
		self.timing_details[name] = time.time()
		

def draw(img, imgpts):
	imgpts = np.int32(imgpts).reshape(-1,2)
	origin = tuple(imgpts[0].ravel())
	img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (0,0,255), 2)
	img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 2)
	img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (255,0,0), 2)
	if len(imgpts)==5:
		img = cv2.line(img, origin, tuple(imgpts[4].ravel()), (50,255,255), 2)
	return img


def world_2_img(xyz_w_pt_lst, rvec, tvec, mtx):
	imgpts_axs = []
	for i, xyz_w in enumerate(xyz_w_pt_lst):
		xyz_w = np.expand_dims(np.array(xyz_w), axis=1)
		r_c_w ,_ = cv2.Rodrigues(rvec)
		xyz_c = np.dot(r_c_w, xyz_w) + tvec
		xyz_img = np.matmul(mtx, xyz_c)
		xy_img = (xyz_img/xyz_img[2])[0:2]
		imgpts_axs.append(tuple(xy_img.squeeze()))
	return imgpts_axs

def img_2_world(img_pt_lst, rvec, tvec, mtx, depth_lst): 
	# img_pt_lst : list of tuples/np_array (1-D) ([(x,y), (x,y), (x,y)])
	# depth_lst : list of depth of above points 
	world_pt_lst = []
	for img_pt, depth in zip(img_pt_lst, depth_lst):
		xyz_img = depth*np.array([[img_pt[0]],[img_pt[1]],[1]])
		xyz_c = np.matmul(np.linalg.inv(mtx), xyz_img)
		r_c_w ,_  = cv2.Rodrigues(rvec)
		r_w_c = np.linalg.inv(r_c_w)
		xyz_w = np.matmul(r_w_c, xyz_c-tvec)
		world_pt_lst.append(tuple(xyz_w.squeeze()))
	return world_pt_lst



def detectObject(frame, crop_coord):

	#[(x1,x2)(y1,y2)] == [(col1,clo2)(row1,row2)] therefore [row,column] requires reversal

	frame = frame[crop_coord[1][0]:crop_coord[1][1],crop_coord[0][0]:crop_coord[0][1]]

	frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)

	# ## Red Cube
	# low_red_bgr = np.array([0, 0, 120])
	# high_red_bgr = np.array([100, 100, 255])

	# low_red_hsv = np.array([0, 120, 100])
	# high_red_hsv = np.array([255, 255, 255])

	## Pink Cylinder
	low_red_bgr = np.array([0, 0, 0])
	high_red_bgr = np.array([255, 255, 255])

	low_red_hsv = np.array([150, 60, 120])
	high_red_hsv = np.array([255, 255, 255])

	red_mask_bgr = cv2.inRange(frame, low_red_bgr, high_red_bgr)
	red_bgr = cv2.bitwise_and(frame, frame, mask=red_mask_bgr)
	hsv_frame = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2HSV)
	red_mask_hsv = cv2.inRange(hsv_frame, low_red_hsv, high_red_hsv)
	red_hsv = cv2.bitwise_and(hsv_frame, hsv_frame, mask=red_mask_hsv)

	red_object = cv2.cvtColor(red_hsv, cv2.COLOR_HSV2BGR)

	red_object_gray = cv2.cvtColor(red_object, cv2.COLOR_BGR2GRAY)

	ret, red_object_bin = cv2.threshold(red_object_gray, 50, 255, cv2.THRESH_BINARY)

	# cv2.imshow('object_detection_frame', frame)

	contours, hierarchies = cv2.findContours(red_object_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	obj_center = None

	if len(contours)>0:
		max_area = cv2.contourArea(contours[0])
		max_i = 0

		for i in range(len(contours)):
			area = cv2.contourArea(contours[i])
			if area>max_area:
				max_area = area
				max_i = i

		M = cv2.moments(contours[max_i])
		if M['m00'] != 0:
			cx = int(M['m10']/M['m00'])+crop_coord[0][0] # cx is in X direction i.e. columns
			cy = int(M['m01']/M['m00'])+crop_coord[1][0] # cy is in X direction i.e. rows
			obj_center = (cx, cy)

	if obj_center is not None:
		return True, obj_center
	else:
		return False, (-1, -1)


class ObjectDetector:
	
	def __init__(
		self,
		img_size=(1280, 720),  # inference size (width, height) (x, y)
		# imgsz=(640, 480),  # inference size (width, height) (x, y)
		view_img=False,  # show results
		print_log=False,
	):
		print('Initialising Object Detector')
		
		self.imgsz = img_size
		self.view_img = view_img
		self.print_log = print_log

		if self.imgsz[1]==480:
			self.img_crop_coord = [(120, 480), (200, 440)] #[(x1, x2), (y1,y2)] -- [360, 240]
			self.fps_disp_pos = (160, 380)
			self.text_disp_pos = (160, 400)
		elif self.imgsz[1]==720:
			self.img_crop_coord = [(300, 900), (300, 650)] #[(x1, x2), (y1, y2)]
			self.fps_disp_pos = (360, 610)
			self.text_disp_pos = (360, 630)
			



		self.depth_range = (0.1, 2.0)

		# Chess Board Configuration
		self.chess_scale = 0.045 #meters
		self.chessboardSize = (5, 3)
		self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		self.objp_chess = np.zeros((self.chessboardSize[0] * self.chessboardSize[1], 3), np.float32)
		self.objp_chess[:,:2] = np.mgrid[0:self.chessboardSize[0],0:self.chessboardSize[1]].T.reshape(-1,2)
		self.objp_chess = self.objp_chess*self.chess_scale

		ax_val = 3*self.chess_scale
		self.axis = [(0,0,0), (ax_val,0,0), (0,ax_val,0), (0,0,ax_val)]

		self.chess_board_detected = False

		self.r_w_r = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
		self.t_w_r = np.array([[-0.11, -0.01, 0.0]]).T

		self.rvec = None
		self.tvec = None

		# Configure depth and color streams
		self.pipeline = rs.pipeline()
		self.config = rs.config()

		self.config.enable_stream(rs.stream.depth, self.imgsz[0], self.imgsz[1], rs.format.z16, 30)
		self.config.enable_stream(rs.stream.color, self.imgsz[0], self.imgsz[1], rs.format.bgr8, 30)

		# Start streaming
		profile = self.pipeline.start(self.config)

		# Setup the 'High Accuracy'-mode
		depth_sensor = profile.get_device().first_depth_sensor()
		preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
		for i in range(int(preset_range.max)):
			visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
			self.log('%02d: %s'%(i,visulpreset))
			if visulpreset == "High Accuracy":
				depth_sensor.set_option(rs.option.visual_preset, i)

		# enable higher laser-power for better detection
		depth_sensor.set_option(rs.option.laser_power, 180)

		# lower the depth unit for better accuracy and shorter distance covered
		depth_sensor.set_option(rs.option.depth_units, 0.005)

		# Getting the depth sensor's depth scale (see rs-align example for explanation)
		self.depth_scale = depth_sensor.get_depth_scale()
		self.log("Depth Scale is: " , self.depth_scale)

		intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
		self.cam_mtx = np.array([[intrinsics.fx, 0.0, intrinsics.ppx],[0.0, intrinsics.fy, intrinsics.ppy],[0.0, 0.0, 1.0]])
		self.cam_dist = np.array(intrinsics.coeffs).reshape(1,5)

		# Create an align object
		# rs.align allows us to perform alignment of depth frames to others frames
		# The "align_to" is the stream type to which we plan to align depth frames.
		self.align = rs.align(rs.stream.color)

		for x in range(5):
			self.pipeline.wait_for_frames()


		self.obj_vec = [(0, 0, 0)]

		self.tm = Timing(last_itrs_means=0)

		print('Successfully Initialised')

		self.pub = rospy.Publisher('object_position', PointStamped, queue_size=10)

	def log(self, *args, **kwargs):
			if self.print_log:
				print(*args, **kwargs)

	def start_detection(self):
		try:
			while True:
				self.tm.start_loop()

				self.log('\n\n-------------Start---------\n')

				## ------ Getting Color and Depth Image ------- ##

				# Wait for a coherent pair of frames: depth and color
				frames = self.pipeline.wait_for_frames()

				# Align the depth frame to color frame
				aligned_frames = self.align.process(frames)

				# Get aligned frames
				depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
				color_frame = aligned_frames.get_color_frame()

				# Validate that both frames are valid
				if not depth_frame or not color_frame:
					self.log('\n-------Unable to get frames-------\n')

				# Convert images to numpy arrays
				depth_image = np.asanyarray(depth_frame.get_data())
				color_image = np.asanyarray(color_frame.get_data())

				# Undistort image (Uncomment for more accuracy but FPS will drop, this step takes 40ms for 1280x72)
				# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.cam_mtx, self.cam_dist, self.imgsz, 1, self.imgsz)	
				# color_image = cv2.undistort(color_image, self.cam_mtx, self.cam_dist, None, newcameramtx)
				## -------------------------------------------- ##

				self.tm.stamp('read_image')

				## --------------- Detecting Chess Board ------ ##
				if not self.chess_board_detected:
					self.log('Detecting chessboard')

					rvec_chess =  np.array([[-0.697],[ 0.692],[ 1.457]])
					tvec_chess = np.array([[-0.063],[ 0.047],[ 0.985]])
					r_c_w ,_ = cv2.Rodrigues(rvec_chess)
					T_c_w = np.vstack((np.hstack((r_c_w, tvec_chess)),np.array([[0.0, 0.0, 0.0, 1.0]])))
					T_w_r = np.vstack((np.hstack((self.r_w_r, self.t_w_r)),np.array([[0.0, 0.0, 0.0, 1.0]]))) 
					T_c_r = np.matmul(T_c_w, T_w_r)
					self.rvec = cv2.Rodrigues(T_c_r[0:3,0:3])[0]
					self.tvec = T_c_r[0:3,3:4]
					self.chess_board_detected = True


					
					# gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
					# ret, corners = cv2.findChessboardCorners(gray, self.chessboardSize, None)
					# if ret == True:
					# 	self.corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
					# 	ret, rvec_chess, tvec_chess = cv2.solvePnP(self.objp_chess, self.corners2, self.cam_mtx, self.cam_dist)

					# 	r_c_w ,_ = cv2.Rodrigues(rvec_chess)
					# 	T_c_w = np.vstack((np.hstack((r_c_w, tvec_chess)),np.array([[0.0, 0.0, 0.0, 1.0]])))
					# 	T_w_r = np.vstack((np.hstack((self.r_w_r, self.t_w_r)),np.array([[0.0, 0.0, 0.0, 1.0]]))) 
					# 	T_c_r = np.matmul(T_c_w, T_w_r)

					# 	self.rvec = cv2.Rodrigues(T_c_r[0:3,0:3])[0]
					# 	self.tvec = T_c_r[0:3,3:4]

					# 	# self.rvec = rvec_chess
					# 	# self.tvec = tvec_chess

					# 	self.chess_board_detected = True

					# 	self.log('chess_', rvec_chess, tvec_chess)
					# 	self.log('tvec:', self.tvec, self.tvec.shape, type(self.tvec))
					# 	self.log('rvec', self.rvec, self.rvec.shape)
					# 	self.log(self.corners2[0])
				else:
					self.log('Using previously Detected Chessboard')
				## -------------------------------------------- ##

				self.tm.stamp('detect_chess')


				## --------------- Detect Object -------------- ##
				object_detected = False
				if self.chess_board_detected:

					object_detected, obj_center_img = detectObject(color_image.copy(), self.img_crop_coord)

					if object_detected:
						self.log(f'Object Detected at: {obj_center_img}')
					else:
						self.log('Object Not Detected')
				## -------------------------------------------- ##

				self.tm.stamp('detect_object')


				## ------- Get 3D cords ---------- ##
				if object_detected:

					# Get 3D coordinates of Object in world from 2d image point

					## Find out why this is not giving result
					obj_radius = 3
					# obj_depth = np.median(depth_image[obj_center_img[0]-obj_size:obj_center_img[0]+obj_size,obj_center_img[1]-obj_size:obj_center_img[1]+obj_size])*self.depth_scale
					# obj_depth = depth_image[obj_center_img[0],obj_center_img[1]]*self.depth_scale

					obj_depths = []
					for x in range(obj_center_img[0]-obj_radius, obj_center_img[0]+obj_radius+1):
						for y in range(obj_center_img[1]-obj_radius, obj_center_img[1]+obj_radius+1):
							obj_depths.append(depth_frame.get_distance(x, y))
					obj_depth = np.median(obj_depths)
					# obj_depth = depth_frame.get_distance(obj_center_img[0], obj_center_img[1])
					
					self.log('object depth:', obj_depth)

					if (obj_depth >= self.depth_range[0] and obj_depth <= self.depth_range[1]):

						self.obj_vec = img_2_world([obj_center_img], self.rvec, self.tvec, self.cam_mtx, [obj_depth])

						self.log('object location (in world):', np.array(self.obj_vec))
					else:
						self.log("Detected object depth is outside range, range:", self.depth_range)

					point = PointStamped()
					point.header.stamp = rospy.Time.now()
					point.header.frame_id = "/"
					point.point.x = self.obj_vec[0][0]
					point.point.y = self.obj_vec[0][1]
					point.point.z = self.obj_vec[0][2]
					self.pub.publish(point)
				## -------------------------------------------- ##

				self.tm.stamp('project_3d')


				## ------- Annotating Images and Displaying---------- ##
				
				if self.view_img:

					# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
					depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

					disp_vectors = None

					if self.chess_board_detected:
						disp_vectors = self.axis
					
					if object_detected:
						disp_vectors = self.axis + self.obj_vec
						cv2.circle(color_image, obj_center_img, 7, (0, 255, 255), -1)
						cv2.circle(depth_colormap, obj_center_img, 7, (0, 255, 255), -1)

						disp_obj_vec = np.array([self.obj_vec[0][0], self.obj_vec[0][1], self.obj_vec[0][2], obj_depth])
						cv2.putText(color_image, f"pos: {disp_obj_vec}", self.text_disp_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

					if disp_vectors is not None:
						imgpts_vectors = world_2_img(disp_vectors, self.rvec, self.tvec, self.cam_mtx)
						color_image = draw(color_image, imgpts_vectors)

					cv2.putText(color_image, f"fps [mean, curr]: [{self.tm.fps_mean}, {self.tm.fps_curr}]", self.fps_disp_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
					
					disp_images = np.hstack((color_image, depth_colormap))

					cv2.imshow('RealSense', disp_images)
					# the 'esc' button is set as the quitting button
					if cv2.waitKey(1) & 0xFF == 27:
						break
				## -------------------------------------------- ##

				self.tm.stamp('annotate')
				
				self.log('\n---------- END -----------------')

				self.tm.end_loop(print_report=self.print_log)

		finally:
			# Stop streaming
			self.pipeline.stop()
			cv2.destroyAllWindows()

def main(args=None):

	parser = argparse.ArgumentParser()
	parser.add_argument('--img-size', nargs='+', type=int, default=[640, 480], help='inference size w,h')
	parser.add_argument('--view-img', action='store_true', help='display image')
	parser.add_argument('--print-log', action='store_true', help='print results')
	opt = parser.parse_args()

	rospy.init_node('object_detector_node', anonymous=True)

	np.set_printoptions(precision=3)

	object_detector = ObjectDetector(**vars(opt))
	object_detector.start_detection()


if __name__ == "__main__":
	main()

