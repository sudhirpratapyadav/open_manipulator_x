import math
import numpy as np
from geometry_msgs.msg import Pose

EPSILON = 0.1
PI = 3.14159265359

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
	return [float("{0:0.2f}".format(num)) for num in lst]


def add_transition(traj, observation, action, reward, info, agent_info, done,
				   next_observation, img_dim):

	# observation["image"] = np.reshape(np.uint8(observation["image"] * 255.0), (img_dim, img_dim, 3))
	observation["image"] = np.uint8(observation["image"] * 255.0)
	traj["observations"].append(observation)
	next_observation["image"] = np.reshape(
		np.uint8(next_observation["image"] * 255.), (img_dim, img_dim, 3))
	traj["next_observations"].append(next_observation)
	traj["actions"].append(action)
	traj["rewards"].append(reward)
	traj["terminals"].append(done)
	traj["agent_infos"].append(agent_info)
	traj["env_infos"].append(info)
	return traj


def collect_one_traj(env, policy, num_timesteps=2, noise=0.0):
	num_steps = -1
	rewards = []
	success = False

	img_dim = 48
	env_action_dim = 8

	action_position_scale = 1.0
	action_orientation_scale = 1.0

	traj = dict(
		observations=[],
		actions=[],
		rewards=[],
		next_observations=[],
		terminals=[],
		agent_infos=[],
		env_infos=[],
	)

	env.reset()
	policy.reset()

	for j in range(num_timesteps):

		print('\n--------------------------------------')
		print(f"\nstep: {j+1}/{num_timesteps}")

		action, agent_info = policy.get_action()

		action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
		print('action:\t', action)

		# observation = env.get_observation()
		# ret, frame = env.cap.read()
		# if ret == True:
		# 	image = np.array(frame)
		# 	# image = np.array(frame)[0:48, 0:48, :]
		# 	print(image.shape)
		# else:
		# 	print('Unable to get Image')
		# 	exit()

		image = np.random.rand(6912)

		observation = {"image":image}

		
		# Take this Action
		##################
		action_position = np.clip(action[:3]+np.random.normal(scale=noise, size=(3,)), env.action_position_limits[0], env.action_position_limits[1])
		action_orientation = action[3:6]
		action_gripper = action[6]
		action_reset = action[7]
		
		target_ee_position = env.ee_position + action_position_scale * action_position
		print('target_ee_position.shape',target_ee_position.shape)

		target_ee_position = np.clip(target_ee_position, env.ee_pos_low, env.ee_pos_high)

		# target_ee_rad = tf_transformations.euler_from_quaternion(env.ee_orientation) + action_orientation_scale * action_orientation
		# target_ee_orientation = tf_transformations.euler_from_quaternion(target_ee_rad)

		target_ee_orientation = env.ee_orientation

		target_ee_pose = Pose()
		target_ee_pose.position.x = target_ee_position[0]
		target_ee_pose.position.y = target_ee_position[1]
		target_ee_pose.position.z = target_ee_position[2]
		target_ee_pose.orientation.x = target_ee_orientation[0]
		target_ee_pose.orientation.y = target_ee_orientation[1]
		target_ee_pose.orientation.z = target_ee_orientation[2]
		target_ee_pose.orientation.w = target_ee_orientation[3]
		
		print(f'\ncurrent_ee_pose:{env.ee_pose}')
		print(f'\ntarget_end_eff_pose:{target_ee_pose}')

		## Executing Action
		ret_val = env.send_ee_pose_goal(target_ee_pose, action_name='Moving end eff', time_step=1.0)

		if action_gripper<-0.5:
			print(f"gripper {action_gripper}<-0.5 == [Close Gripper]")
			ret_val = env.send_gripper_command(action_gripper=env.CLOSE_GRIPPER, action_name='Closing Gripper', time_step=1.0)
		elif action_gripper>0.5:
			print(f"gripper {action_gripper}>0.5 == [Open Gripper]")
			ret_val = env.send_gripper_command(action_gripper=env.OPEN_GRIPPER, action_name='Opening Gripper', time_step=1.0)

		
		

		##################

		next_observation = {"image":np.random.rand(6912)}
		reward = 0.1
		done = False
		info = None

		add_transition(traj, observation,  action, reward, info, agent_info,
					   done, next_observation, img_dim)

		rewards.append(reward)
		if done:
			break

	success = True

	return traj, success