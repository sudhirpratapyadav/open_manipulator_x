import numpy as np

class Policy_PickPlace:

	def __init__(self, env, pick_height_thresh=0.10):

		self.env = env

		self.gripper_object_distance_threshold = 0.02 

		self.pick_height_thresh = pick_height_thresh

		self.pick_point = np.asarray([0.0, 0.0, 0.0])

		# self.pick_poses = dict(
		# 						first = dict(
		# 							joint=util.deg2rad([18, -120, -67, -85, 90, 18.0]),
		# 							cartesian=util.deg2rad([0.693, 0.111, 0.323]) 
		# 						),
		# 						second = dict(
		# 							joint=util.deg2rad([18, -120, -67, -85, 90, 18.0]),
		# 							cartesian=util.deg2rad([0.693, 0.111, 0.323]) 
		# 						)
		# 					)

		self.object_grasped = False
		self.object_grasped_once = False
		self.object_grasped_lifted = False

	def reset(self):
		self.object_grasped = False
		self.object_grasped_once = False
		self.object_grasped_lifted = False

		self.pick_point = self.env.object_pos

		# self.pick_point = np.array([self.env.object_pos[0], self.env.object_pos[1], 0.03])

		print('\n\nself.pick_point\n\n', self.pick_point)

	def get_action(self):

		# ee_pos = np.array([self.env.ee_pose.position.x, self.env.ee_pose.position.y, self.env.ee_pose.position.z])
		# ee_quat = np.array([self.env.ee_pose.orientation.x, self.env.ee_pose.orientation.y, self.env.ee_pose.orientation.z, self.env.ee_pose.orientation.w])

		if not self.object_grasped:
			self.pick_point = self.env.object_pos
			gripper_object_distance = np.linalg.norm(self.env.object_pos - self.env.ee_position)
			self.object_grasped = gripper_object_distance<self.gripper_object_distance_threshold and self.env.gripper_closed

			print('self.env.object_pos', self.env.object_pos, self.env.ee_position)

			print('gripper_object_distance', gripper_object_distance, self.gripper_object_distance_threshold)

		self.object_grasped_lifted = self.object_grasped and self.env.ee_position[2]>self.pick_height_thresh

		gripper_pickpoint_dist = np.linalg.norm(self.pick_point - self.env.ee_position)

		print('self.env.gripper_closed', self.env.gripper_closed)
		print('self.object_grasped', self.object_grasped)
		print('self.object_grasped_lifted', self.object_grasped_lifted, self.env.ee_position[2])
		print('gripper_pickpoint_dist', gripper_pickpoint_dist)

		done = False
		neutral_action = [0.]

		if gripper_pickpoint_dist > 0.01 and not self.object_grasped:
			action_xyz = (self.pick_point - self.env.ee_position)
			xy_diff = np.linalg.norm(action_xyz[:2])
			print('xy_diff', xy_diff)
			# if xy_diff > 0.08:
			# 	action_xyz[2] = 0.0
			action_angles = [0., 0., 0.]
			action_gripper = [0.]  
			print("Coming Closer")
		elif not self.object_grasped:
			action_xyz = (self.pick_point - self.env.ee_position)
			action_angles = [0., 0., 0.]
			action_gripper = [-1.0]
			print("Grasping")
		elif not self.object_grasped_lifted:
			print('self.env.RESET_END_EFF_POS',self.env.RESET_END_EFF_POSE)
			print('self.env.ee_position', self.env.ee_position)
			action_xyz = (self.env.RESET_END_EFF_POSE - self.env.ee_position)
			action_angles = [0., 0., 0.]
			action_gripper = [0.]
			print("Lifiting")
		else:	
			action_xyz = (0., 0., 0.)
			action_angles = [0., 0., 0.]
			action_gripper = [0.]
			print("Hold")


		agent_info = dict(done=done)

		action = np.concatenate((action_xyz, action_angles, action_gripper, neutral_action))

		# action = [0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]

		return action, agent_info