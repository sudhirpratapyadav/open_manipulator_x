import rospy
from threading import Thread

import time
from threading import Event

from sensor_msgs.msg import JointState
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest

##
import numpy as np
import math
import os
import pathlib
import sys

OPEN_GRIPPER = [-0.02]
CLOSE_GRIPER = [0.0]


def rad2deg(joint_pos_rad):
	if isinstance(joint_pos_rad, list):
		if isinstance(joint_pos_rad[0], list):
			joint_pos_deg = [[angle*(180/math.pi) for angle in lst] for lst in joint_pos_rad]
		else:
			joint_pos_deg = [angle*(180/math.pi) for angle in joint_pos_rad]
	else:
		joint_pos_deg = joint_pos_rad*(180/math.pi)
	return joint_pos_deg

def deg2rad(joint_pos_deg):
	if isinstance(joint_pos_deg, list):
		if isinstance(joint_pos_deg[0], list):
			joint_pos_rad = [ [angle*(math.pi/180) for angle in lst] for lst in joint_pos_deg]
		else:
			joint_pos_rad = [angle*(math.pi/180) for angle in joint_pos_deg]
	else:
		joint_pos_rad = joint_pos_deg*(math.pi/180)
	return joint_pos_rad

def print_format_list(lst):
	return ["{0:0.2f}".format(num) for num in lst]

np.set_printoptions(precision=3, suppress=True)

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

		self.HOME_POSE = deg2rad([0.0, -60.0, 21.0, 40.3])

		self.RESET_POSE = deg2rad([0.0, 0.0, 0.0, 0.0])

		# Subscriber for joint positions
		self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)

		# Service for joint position control
		rospy.wait_for_service('/goal_joint_space_path', 10)
		self.goal_joint_space_path_service = rospy.ServiceProxy("/goal_joint_space_path", SetJointPosition)

		# Service for gripper control
		rospy.wait_for_service('/goal_tool_control', 10)
		self.goal_tool_control_service = rospy.ServiceProxy("/goal_tool_control", SetJointPosition)
		

	def init_robot(self):

		temp_successful = True
		rospy.loginfo(f'\n\n----------------- Initialising Robot ----------------\n')

		rospy.loginfo('Sending to Home Position')
		result = self.send_joint_pose_goal(self.HOME_POSE, action_name='go_to_home_pose', time_step=2.0)
		if result is True:
			rospy.loginfo('Sending to Home Position: successfull\n')
		else:
			rospy.loginfo('Sending to Home Position: Unsuccessfull\n')
			temp_successful = False
		time.sleep(2.5)	

		rospy.loginfo('Checking Gripper: Opening and Closing\n')
		result = self.send_gripper_command(joint_pose_goal=OPEN_GRIPPER, action_name='Opening Gripper', time_step=1.0)
		if result is True:
			rospy.loginfo('Openining Gripper: successfull\n')
		else:
			rospy.loginfo('Openining Gripper: Unsuccessfull\n')
			temp_successful = False
		time.sleep(1.0)
		result = self.send_gripper_command(joint_pose_goal=CLOSE_GRIPER, action_name='Closing Gripper', time_step=1.0)
		if result is True:
			rospy.loginfo('Closing Gripper: successfull\n')
		else:
			rospy.loginfo('Closing Gripper: Unsuccessfull\n')
			temp_successful = False
		time.sleep(1.0)
		rospy.loginfo('Checking Gripper: successfull\n')

		if temp_successful:
			rospy.loginfo(f'\n\n-------------- Robot Successfully Initialised----------\n\n\n')
			self.init_success = True
		else:
			rospy.loginfo(f'\n\n-------------- FAILED to Initialise Robot ----------\n\n\n')
			self.init_success = False



	def reset(self):
		rospy.loginfo('Sending to RESET Position')
		result = self.send_joint_pose_goal(self.RESET_POSE, action_name='RESET', time_step=2.0)
		if result is True:
			rospy.loginfo('Sending to RESET Position: successfull\n')
		else:
			rospy.loginfo('Sending to RESET Position: Unsuccessfull\n')
			temp_successful = False
		time.sleep(2.5)	


	def joint_state_callback(self, joint_state):

		joint_pos_ordered = [None]*len(joint_state.position)

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
				# if self.init_success:
				rospy.loginfo(f'joint_angles: {rad2deg(self.current_joint_position)}')

	def send_joint_pose_goal(self, joint_pose_goal, action_name='move', time_step=2.0):
		# This function sends a action request to action-server
		# to take a action -  in this case action is
		# joint-position-target i.e. move joints to reach given target-joint-positions

		set_joint_position = SetJointPositionRequest()

		set_joint_position.joint_position.joint_name = self.joint_names[:4]
		set_joint_position.joint_position.position = joint_pose_goal
		set_joint_position.path_time = time_step

		rospy.loginfo(f'Exec :{action_name}')
		# rospy.loginfo(f'set_joint_position:\n{set_joint_position}')

		result = self.goal_joint_space_path_service(set_joint_position)


		return result.is_planned


	def send_gripper_command(self, joint_pose_goal, action_name='move', time_step=2.0):
		# This function sends a action request to action-server
		# to take a action -  in this case action is
		# joint-position-target i.e. move joints to reach given target-joint-positions

		set_joint_position = SetJointPositionRequest()

		set_joint_position.joint_position.joint_name = ['gripper']
		set_joint_position.joint_position.position = joint_pose_goal
		set_joint_position.path_time = time_step

		rospy.loginfo(f'Exec :{action_name}')
		# rospy.loginfo(f'set_joint_position :\n{set_joint_position}')

		result = self.goal_tool_control_service(set_joint_position)


		return result.is_planned


def main(args=None):

	rospy.init_node('open_manipulator_control_node', anonymous=True)

	robot_arm = OpenManipulatorControl()

	time.sleep(1.0)
	robot_arm.init_robot()
	time.sleep(4.0)
	robot_arm.reset()
	time.sleep(4.0)

	rate = rospy.Rate(10) # 10hz
	while not rospy.is_shutdown():
		rate.sleep()


if __name__ == "__main__":
	main()
