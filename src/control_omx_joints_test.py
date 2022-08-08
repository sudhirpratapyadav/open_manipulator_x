import rospy
from threading import Thread

import time
from threading import Event

from sensor_msgs.msg import JointState
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest, SetJointPosition, SetJointPositionRequest
from open_manipulator_msgs.msg import KinematicsPose

##
import numpy as np
import math
import os
import pathlib
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from Policy_PickPlace import Policy_PickPlace
from util import rad2deg, deg2rad, collect_one_traj, print_format_list

LOOP_RATE = 10  # number of times per second
LOOP_WAIT_TIME = 6  # time between commands in seconds

SAVE_PATH = '/home/sudhir/robotics_ros_workspaces/results'
# if not os.path.exists(SAVE_PATH):
# 	os.makedirs(SAVE_PATH)

PRINT_FILE_PATH = os.path.join(SAVE_PATH , 'log.txt')


class OpenManipulatorControl():

	def __init__(self):

		# Joints
		self.joint_names = [
		'joint1',
		'joint2',
		'joint3',
		'joint4',
		'gripper',
		]

		self.init_success = False

		# self.HOME_POSE = deg2rad([0.0, -60.0, 21.0, 40.3])

		self.HOME_POSE = deg2rad([0.0, 0.0, 0.0, 0.0])

		self.RESET_POSE = deg2rad([0.0, 0.0, 0.0, 0.0])

		self.object_pos = np.asarray([0.683, 0.100, 0.190])
		self.action_position_limits = (-0.05, 0.05)

		self.OPEN_GRIPPER_VALUE = [0.012]
		self.CLOSE_GRIPPER_VALUE = [-0.012]

		self.OPEN_GRIPPER = False
		self.CLOSE_GRIPPER = True

		self.ee_pose = None
		self.ee_position = None
		self.ee_orientation = None
		self.current_joint_position = None
		self.gripper_closed = None
		# self.gripper_state = CLOSE_GRIPPER

		# Subscriber for joint positions
		self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
		# Subscriber for end-eff pose
		self.end_eff_pose_sub = rospy.Subscriber("/gripper/kinematics_pose", KinematicsPose, self.end_eff_pose_callback)

		# Service for joint position control
		rospy.wait_for_service('/goal_joint_space_path', 10)
		self.goal_joint_space_path_client = rospy.ServiceProxy("/goal_joint_space_path", SetJointPosition)

		# Service for end_eff_delta position control
		rospy.wait_for_service('/goal_task_space_path', 10)
		self.goal_task_space_path_client = rospy.ServiceProxy("/goal_task_space_path", SetKinematicsPose)

		# Service for gripper control
		rospy.wait_for_service('/goal_tool_control', 10)
		self.goal_tool_control_service = rospy.ServiceProxy("/goal_tool_control", SetJointPosition)

		# # The argument '0' gets the default webcam.
		# self.cap = cv2.VideoCapture(0)
		 
		# # Used to convert between ROS and OpenCV images
		# self.br = CvBridge()
		

	def init_robot(self):

		temp_successful = True
		rospy.loginfo(f'\n\n----------------- Initialising Robot ----------------\n')

		rospy.loginfo('Sending to Home Position')
		result = self.send_joint_pose_goal(self.HOME_POSE, action_name='go_to_home_pose', time_step=3.0)
		if result is True:
			rospy.loginfo('Sending to Home Position: successfull\n')
		else:
			rospy.loginfo('Sending to Home Position: Unsuccessfull\n')
			temp_successful = False

		rospy.loginfo('Checking Gripper: Opening and Closing\n')
		result = self.send_gripper_command(action_gripper=self.OPEN_GRIPPER, action_name='Opening Gripper', time_step=1.0)
		if result is True:
			rospy.loginfo('Openining Gripper: successfull\n')
		else:
			rospy.loginfo('Openining Gripper: Unsuccessfull\n')
			temp_successful = False

		result = self.send_gripper_command(action_gripper=self.CLOSE_GRIPPER, action_name='Closing Gripper', time_step=1.0)
		if result is True:
			rospy.loginfo('Closing Gripper: successfull\n')
		else:
			rospy.loginfo('Closing Gripper: Unsuccessfull\n')
			temp_successful = False

		rospy.loginfo('Checking Gripper: successfull\n')

		if temp_successful:
			rospy.loginfo(f'\n\n-------------- Robot Successfully Initialised----------\n\n\n')
			self.init_success = True
		else:
			rospy.loginfo(f'\n\n-------------- FAILED to Initialise Robot ----------\n\n\n')
			self.init_success = False



	def reset(self):
		rospy.loginfo('Sending to RESET Position')
		result = self.send_joint_pose_goal(self.RESET_POSE, action_name='RESET', time_step=3.0)
		if result is True:
			rospy.loginfo('Sending to RESET Position: successfull\n')
		else:
			rospy.loginfo('Sending to RESET Position: Unsuccessfull\n')



	def joint_state_callback(self, joint_state):

		joint_pos_ordered = [None]*len(self.joint_names)

		for i, name in enumerate(joint_state.name):
			if name==self.joint_names[0]:
				joint_pos_ordered[0] = joint_state.position[i]
			elif name==self.joint_names[1]:
				joint_pos_ordered[1] = joint_state.position[i]
			elif name==self.joint_names[2]:
				joint_pos_ordered[2] = joint_state.position[i]
			elif name==self.joint_names[3]:
				joint_pos_ordered[3] = joint_state.position[i]
			elif name==self.joint_names[4]:
				joint_pos_ordered[4] = joint_state.position[i]

		self.current_joint_position = joint_pos_ordered
		rospy.loginfo(f'joint_angles: {print_format_list(rad2deg(self.current_joint_position))}')
		rospy.loginfo(f'end_eff_pose: {self.ee_position}')
		

	def end_eff_pose_callback(self, kinematics_pose):
		self.ee_pose = kinematics_pose.pose
		self.ee_position = np.array([self.ee_pose.position.x, self.ee_pose.position.y, self.ee_pose.position.z])
		self.ee_orientation = np.array([self.ee_pose.orientation.x, self.ee_pose.orientation.y, self.ee_pose.orientation.z, self.ee_pose.orientation.w])

	def send_joint_pose_goal(self, joint_pose_goal, action_name='move', time_step=2.0):
		# This function sends a action request to action-server
		# to take a action -  in this case action is
		# joint-position-target i.e. move joints to reach given target-joint-positions

		set_joint_position = SetJointPositionRequest()

		set_joint_position.joint_position.joint_name = self.joint_names[:4]
		set_joint_position.joint_position.position = joint_pose_goal
		set_joint_position.path_time = time_step

		rospy.loginfo(f'Exec: {action_name}')

		result = self.goal_joint_space_path_client(set_joint_position)

		time.sleep(time_step+0.5)

		return result.is_planned

	def send_ee_pose_goal(self, ee_pose, action_name='move', time_step=2.0):

		## ee_pose: geometry_msgs/Pose

		set_end_eff_pose = SetKinematicsPoseRequest()

		set_end_eff_pose.end_effector_name = 'gripper'
		set_end_eff_pose.kinematics_pose.pose = ee_pose
		set_end_eff_pose.path_time = time_step

		rospy.loginfo(f'Exec: {action_name}')

		result = self.goal_task_space_path_client(set_end_eff_pose)

		time.sleep(time_step+0.5)

		return result.is_planned


	def send_gripper_command(self, action_gripper, action_name='move', time_step=2.0):
		# This function sends a action request to action-server
		# to take a action -  in this case action is
		# joint-position-target i.e. move joints to reach given target-joint-positions
		
		joint_gripper_position = self.CLOSE_GRIPPER_VALUE if action_gripper==self.CLOSE_GRIPPER else self.OPEN_GRIPPER_VALUE

		set_joint_position = SetJointPositionRequest()

		set_joint_position.joint_position.joint_name = ['gripper']
		set_joint_position.joint_position.position = joint_gripper_position
		set_joint_position.path_time = time_step

		rospy.loginfo(f'Exec: {action_name}')

		result = self.goal_tool_control_service(set_joint_position)

		self.gripper_closed = True if action_gripper==self.CLOSE_GRIPPER else False

		time.sleep(time_step+0.5)

		return result.is_planned


def main(args=None):

	rospy.init_node('open_manipulator_control_node', anonymous=True)

	robot_arm = OpenManipulatorControl()

	# time.sleep(1.0)
	# robot_arm.init_robot()

	# result = robot_arm.send_gripper_command(action_gripper=robot_arm.OPEN_GRIPPER, action_name='Opening Gripper', time_step=2.0)
	# result = robot_arm.send_gripper_command(action_gripper=robot_arm.CLOSE_GRIPPER, action_name='Closing Gripper', time_step=2.0)

	time.sleep(1.0)
	robot_arm.reset()

	rate = rospy.Rate(10) # 10hz


	print('--------------- Starting Data Collection ------------------')

	joint_pose_goal = deg2rad([0.0, 0.0, 0.0, 0.0])

	result = robot_arm.send_joint_pose_goal(joint_pose_goal, action_name=f'move', time_step=5.0)

	time_taken = 2.0

	joint_limits = [
					deg2rad([0.0, 30]),
					deg2rad([0.0, 30]),
					deg2rad([0.0, 30]),
					deg2rad([0.0, 30]),
				]

	for i in range(4):
		joint_pose_goal = [0.0, 0.0, 0.0, 0.0]
		joint_pose_goal[i] = joint_limits[i][1]
		result = robot_arm.send_joint_pose_goal(joint_pose_goal, action_name=f'move joint_{i+1} +ve', time_step=time_taken)
		result = robot_arm.send_joint_pose_goal(robot_arm.RESET_POSE, action_name=f'move joint_{i+1} -ve', time_step=time_taken)

	result = robot_arm.send_gripper_command(action_gripper=robot_arm.OPEN_GRIPPER, action_name='Opening Gripper', time_step=2.0)
	result = robot_arm.send_gripper_command(action_gripper=robot_arm.CLOSE_GRIPPER, action_name='Closing Gripper', time_step=2.0)


if __name__ == "__main__":
	main()
