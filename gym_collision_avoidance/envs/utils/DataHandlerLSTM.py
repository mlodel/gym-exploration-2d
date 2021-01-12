import numpy as np
import os
import random
import sys
import math
import cv2
import pickle as pkl
from pykalman import KalmanFilter
from copy import deepcopy
import matplotlib.pyplot as pl
import matplotlib.animation as animation
from time import sleep
import random
from scipy.stats import multivariate_normal
import json
from matplotlib.patches import Ellipse
from tqdm import tqdm
import glob
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.utils.Trajectory import *
from gym_collision_avoidance.envs.utils.AgentContainer import AgentContainer as ped_cont
from gym_collision_avoidance.envs.utils import Support as sup

class DataHandlerLSTM():
	"""
	Data handler for training an LSTM pedestrian prediction model
	"""
	def __init__(self,scenario):

		self.data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'datasets/'+scenario))
		self.scenario = scenario
		self.dt = Config.DT
		self.min_length_trajectory = 4
		self.trajectory_set = []
		# Normalization constants
		self.norm_const_x = 1.0
		self.norm_const_y = 1.0
		self.norm_const_heading = 1.0
		self.norm_const_vx = 1.0
		self.norm_const_vy = 1.0
		self.norm_const_omega = 1.0
		self.min_pos_x = 1000
		self.min_pos_y = 1000
		self.max_pos_x = -1000
		self.max_pos_y = -1000
		self.min_vel_x = 1000
		self.min_vel_y = 1000
		self.max_vel_x = -1000
		self.max_vel_y = -1000
		self.avg_traj_length = 0

		# Data structure containing all the information about agents
		self.agent_container = ped_cont()

		self.processData()

	def processData(self, **kwargs):
		"""
		Processes the simulation or real-world data, depending on the usage.
		"""
		data_pickle = self.data_path + "/" + self.scenario + ".pickle"
		if os.path.isfile(data_pickle):
			self.loadTrajectoryData(data_pickle)
		else:
			print("Processing real-world data.")
			self._process_real_data_()
			self.saveTrajectoryData(data_pickle)

	def _process_gym_data_(self, **kwargs):
		"""
		Process data generated with gym-collision-avoidance simulator
		"""
		print("Loading data from: '{}'".format(self.args.data_path + self.args.dataset))

		self.load_map(**kwargs)

		self.trajectory_set = []
		file_list = glob.glob(self.args.data_path + self.args.dataset)

		for file in file_list:

			print("Loading: " + file)

			self.file = open(file, 'rb')
			tmp_self = pkl.load(self.file , encoding='latin1')

			# Iterate through the data and fill the register
			if not self.trajectory_set:
				step = int(self.args.dt / 0.1)

			for traj_id in tqdm(range(len(tmp_self))):
				traj = tmp_self[traj_id]
				if len(traj)/step > self.args.truncated_backprop_length + self.args.prediction_horizon + self.args.prev_horizon:
					self.trajectory_set.append(
						(traj_id, Trajectory.Trajectory(goal=np.asarray(traj[0]["pedestrian_goal_position"]))))
					for t_id in range(0,len(traj),step):
						timestamp = traj[t_id]["time"]
						pose = np.zeros([1, 3])
						vel = np.zeros([1, 3])
						pose[:, 0:2] = traj[t_id]["pedestrian_state"]["position"]
						vel[:, 0:2] = traj[t_id]["pedestrian_state"]["velocity"]
						self.trajectory_set[-1][1].time_vec = np.insert(self.trajectory_set[-1][1].time_vec, int(t_id/step), timestamp)
						self.trajectory_set[-1][1].pose_vec = np.insert(self.trajectory_set[-1][1].pose_vec, int(t_id/step), pose, axis=0)
						self.trajectory_set[-1][1].vel_vec = np.insert(self.trajectory_set[-1][1].vel_vec, int(t_id/step), vel, axis=0)
						other_agents_pos = np.asarray(traj[t_id]["other_agents_pos"])
						other_agents_vel = np.asarray(traj[t_id]["other_agents_vel"])
						self.trajectory_set[-1][1].other_agents_positions.append(other_agents_pos)
						self.trajectory_set[-1][1].other_agents_velocities.append(other_agents_vel)

		# Dataset Statistics
		cnt = 0
		avg_len = 0
		for traj_id in tqdm(range(len(self.trajectory_set))):
			avg_len = (avg_len*cnt+self.trajectory_set[traj_id][1].pose_vec.shape[0])/(cnt+1)

		print("Avg. Trajectory Length: " + str(avg_len))
		print("Total number of trajectories: " + str(len(self.trajectory_set)))

		self.compute_min_max_values()

	def _process_simulation_data_(self, **kwargs):
		"""
		Import the data from the log file stored in the directory of data_path.
		This method brings all the data into a suitable format for training.
		"""
		self.load_map(**kwargs)
		# Pedestrian data
		# [id, timestep (s), timestep (ns), pos x, pos y, yaw, vel x, vel y, omega, goal x, goal y]
		pedestrian_data = np.genfromtxt(os.path.join(self.data_path+self.args.scenario, 'total_log.csv'), delimiter=",")[1:, :]

		# Iterate through the data and fill the register
		for sample_idx in range(pedestrian_data.shape[0]):
			#if pedestrian_data[sample_idx, 0] != -1:
			id = pedestrian_data[sample_idx, 0]
			timestamp = np.round(pedestrian_data[sample_idx, 1],1)# + pedestrian_data[sample_idx, 2] * 1e-9  # time in seconds
			pose = np.zeros([1,3])
			vel = np.zeros([1,3])
			pose[:,0:2] = np.true_divide(pedestrian_data[sample_idx, 3:5], np.array([self.norm_const_x, self.norm_const_y]))
			vel[:,0:2] = np.true_divide(pedestrian_data[sample_idx, 5:7], np.array([self.norm_const_vx, self.norm_const_vy]))
			goal = np.true_divide(pedestrian_data[sample_idx, 7:9], np.array([self.norm_const_x, self.norm_const_y]))

			self.agent_container.addDataSample(id, timestamp, pose, vel, goal)

		# Set the initial indices for agent trajectories (which trajectory will be returned when queried)
		self.agent_traj_idx = [0] * self.agent_container.getNumberOfAgents()

#     for id in self.agent_container.getAgentIDs():
#       for traj in self.agent_container.getAgentTrajectories(id):
#         if len(traj) > self.min_length_trajectory:
#           traj.smoothenTrajectory(dt=self.dt)

		# Subsample trajectories (longer discretization time) from dt=0.1 to dt=0.3
		for id in self.agent_container.getAgentIDs():
			for traj in self.agent_container.getAgentTrajectories(id):
				traj.subsample(int(self.args.dt*10))

		# Reconstruct interpolators since they were not pickled with the rest of the trajectory
		for id in self.agent_container.getAgentIDs():
			for traj_idx, traj in enumerate(self.agent_container.getAgentTrajectories(id)):
				if len(traj) > self.min_length_trajectory:
					traj.updateInterpolators()

		# Put all the trajectories in the trajectory set and randomize
		for id in self.agent_container.getAgentIDs():
			print("Processing agent {} / {}".format(id, self.agent_container.getNumberOfAgents()))
			# Adds trajectory if bigger than a minimum length and maximum size
			self.addAgentTrajectoriesToSet(self.agent_container,self.trajectory_set,id)

		self.compute_min_max_values()

	def shift_data(self):

		for traj_id in range(len(self.trajectory_set)):
			for t_id in range(1, self.trajectory_set[traj_id][1].pose_vec.shape[0]):
				self.trajectory_set[traj_id][1].pose_vec[t_id,0] -= (self.max_pos_x-self.min_pos_y)/2
				self.trajectory_set[traj_id][1].pose_vec[t_id, 1] -= (self.max_pos_y-self.min_pos_y)/2

	def compute_min_max_values(self):
		self.mean_pos_x = 0
		self.mean_pos_y = 0

		for traj_id in range(len(self.trajectory_set)):
			for t_id in range(1, self.trajectory_set[traj_id][1].pose_vec.shape[0]):
				self.min_pos_x = min(self.min_pos_x,self.trajectory_set[traj_id][1].pose_vec[t_id,0])
				self.min_pos_y = min(self.min_pos_y, self.trajectory_set[traj_id][1].pose_vec[t_id, 1])
				self.max_pos_x = max(self.max_pos_x, self.trajectory_set[traj_id][1].pose_vec[t_id, 0])
				self.max_pos_y = max(self.max_pos_y, self.trajectory_set[traj_id][1].pose_vec[t_id, 1])
				self.min_vel_x = min(self.min_vel_x,self.trajectory_set[traj_id][1].vel_vec[t_id,0])
				self.min_vel_y = min(self.min_vel_y, self.trajectory_set[traj_id][1].vel_vec[t_id, 1])
				self.max_vel_x = max(self.max_vel_x, self.trajectory_set[traj_id][1].vel_vec[t_id, 0])
				self.max_vel_y = max(self.max_vel_y, self.trajectory_set[traj_id][1].vel_vec[t_id, 1])

			self.mean_pos_x += np.mean(self.trajectory_set[traj_id][1].pose_vec[:, 0], axis=0)/len(self.trajectory_set)
			self.mean_pos_y += np.mean(self.trajectory_set[traj_id][1].pose_vec[:, 1], axis=0)/len(self.trajectory_set)

		self.calc_scale()

	def _process_real_data_(self):
		"""
		Import the real-world data from the log file stored in the directory of data_path.
		This method brings all the data into a suitable format for training.
		"""
		print("Extracting the occupancy grid ...")
		# Occupancy grid data
		self.agent_container.occupancy_grid.resolution = 0.1  # map resolution in [m / cell]
		self.agent_container.occupancy_grid.map_size = np.array([50., 50.])  # map size in [m]
		self.agent_container.occupancy_grid.gridmap = np.zeros([int(self.agent_container.occupancy_grid.map_size[0] / self.agent_container.occupancy_grid.resolution),
																														int(self.agent_container.occupancy_grid.map_size[1] / self.agent_container.occupancy_grid.resolution)])  # occupancy values of cells
		self.agent_container.occupancy_grid.center = self.agent_container.occupancy_grid.map_size / 2.0
		# Extract grid from real data
		# Homography matrix to transform from image to world coordinates
		H = np.genfromtxt(os.path.join(self.data_path, 'H.txt'), delimiter='  ', unpack=True).transpose()

		# Extract static obstacles
		obst_threshold = 200
		static_obst_img = cv2.imread(os.path.join(self.data_path, 'map.png'), 0)
		obstacles = np.zeros([0, 3])
		# pixel coordinates do cartesian coordinates
		for xx in range(static_obst_img.shape[0]):
			for yy in range(static_obst_img.shape[1]):
				if static_obst_img[xx, yy] > obst_threshold:
					obstacles = np.append(obstacles, np.dot(H, np.array([[xx], [yy], [1]])).transpose(), axis=0)

		# Compute obstacles in 2D
		self.obstacles_2d = np.zeros([obstacles.shape[0], 2])
		self.obstacles_2d[:, 0] = obstacles[:, 0] / obstacles[:, 2]
		self.obstacles_2d[:, 1] = obstacles[:, 1] / obstacles[:, 2]

		for obst_ii in range(self.obstacles_2d.shape[0]):
			obst_idx = self.agent_container.occupancy_grid.getIdx(self.obstacles_2d[obst_ii,0], self.obstacles_2d[obst_ii,1])
			self.agent_container.occupancy_grid.gridmap[obst_idx] = 1.0

		print("Extracting the pedestrian data ...")
		# Pedestrian data
		# [id, timestep (s), timestep (ns), pos x, pos y, yaw, vel x, vel y, omega, goal x, goal y]
		if os.path.exists(self.data_path +'/obsmat.txt'):
			pedestrian_data = np.genfromtxt(os.path.join(self.data_path , 'obsmat.txt'), delimiter="  ")[1:, :]
			pixel_data = False
		elif os.path.exists(self.data_path +'/obsmat_px.txt'):
			pedestrian_data = np.genfromtxt(os.path.join(self.data_path, 'obsmat_px.txt'), delimiter="  ")[1:, :]
			pixel_data = True
		else:
			print("Could not find obsmat.txt or obsmat_px.txt")

		idx_frame = 0
		idx_id = 1
		idx_posx = 2
		idx_posy = 4
		idx_posz = 3
		idx_vx = 5
		idx_vy = 7
		idx_vz = 6
		dt = 0.4 # seconds (equivalent to 2.5 fps)
		if os.path.split(self.data_path)[-1] == 'seq_eth':
			frames_between_annotation = 6.0
		else:
			frames_between_annotation = 10.0

		# Iterate through the data and fill the register
		for sample_idx in range(pedestrian_data.shape[0]):
			id = pedestrian_data[sample_idx, idx_id]
			timestamp = pedestrian_data[sample_idx, idx_frame] * dt / frames_between_annotation  # time in seconds
			pose = np.zeros([1,3])
			vel = np.zeros([1,3])
			pose[:,0] = pedestrian_data[sample_idx, idx_posx]
			if self.scenario == "zara_02":
				pose[:, 1] = pedestrian_data[sample_idx, idx_posy] + 14
			else:
				pose[:,1] = pedestrian_data[sample_idx, idx_posy]
			vel[:, 0] = pedestrian_data[sample_idx, idx_vx]
			vel[:, 1] = pedestrian_data[sample_idx, idx_vy]
			if pixel_data:
				converted_pose = sup.to_pos_frame(H, np.expand_dims(np.array((pedestrian_data[sample_idx, idx_posx], pedestrian_data[sample_idx, idx_posy])), axis=0).astype(float))
				pose[:, 0] = converted_pose[0,0]
				pose[:, 1] = converted_pose[0,1]
			goal = np.zeros([2])

			self.agent_container.addDataSample(id, timestamp, pose, vel, goal)

		# Set the initial indices for agent trajectories (which trajectory will be returned when queried)
		self.agent_traj_idx = [0] * self.agent_container.getNumberOfAgents()

		# Subsample trajectories (longer discretization time)
		if dt != self.dt:
			for id in self.agent_container.getAgentIDs():
				for traj in self.agent_container.getAgentTrajectories(id):
					if len(traj) > self.min_length_trajectory:
						traj.smoothenTrajectory(dt=self.dt) # before was 0.3
						traj.goal = np.expand_dims(traj.pose_vec[-1, :2], axis=0)
					else:
						self.agent_container.removeAgent(id)

		# Put all the trajectories in the trajectory set and randomize
		for cnt, id in enumerate(self.agent_container.getAgentIDs()):
			self.addAgentTrajectoriesToSet(self.agent_container,self.trajectory_set,id)

		#self.compute_min_max_values()

	def calc_scale(self, keep_ratio=False):

		self.sx_vel = 1 / (self.max_vel_x - self.min_vel_x)
		self.sy_vel = 1 / (self.max_vel_y - self.min_vel_y)
		if keep_ratio:
			if self.sx_vel > self.sy_vel:
				self.sx_vel = self.sy_vel
			else:
				self.sy_vel = self.sx_vel

		self.sx_pos = 1 / (self.max_pos_x - self.min_pos_x)
		self.sy_pos = 1 / (self.max_pos_y - self.min_pos_y)
		if keep_ratio:
			if self.sx_pos > self.sy_pos:
				self.sx_pos = self.sy_pos
			else:
				self.sy_pos = self.sx_pos

	def addAgentTrajectoriesToSet(self,agent_container,trajectory_set, id):
		"""
		Goes through all trajectories of agent and adds them to the member set if they fulfill the criteria.
		For all the time steps within the trajectory it also computes the positions of the other agents at that
		timestep in order to make training more efficient.
		"""
		for traj_idx, traj in enumerate(agent_container.getAgentTrajectories(id)):
			traj_with_collision = False
			if len(traj) > self.min_length_trajectory:
				#if traj.getMinTime() < 100:
				traj.updateInterpolators()
				# Find other agent's trajectories which overlap with each time step
				for time_idx in range(traj.time_vec.shape[0]):
					query_time = traj.time_vec[time_idx]
					other_agents_positions = agent_container.getAgentPositionsForTimeExclude(query_time, id)
					other_agents_velocities = agent_container.getAgentVelocitiesForTimeExclude(query_time, id)
					# Remove ego agent
					traj.other_agents_positions.append(other_agents_positions)
					traj.other_agents_velocities.append(other_agents_velocities)
				trajectory_set.append((id, traj))

	def saveTrajectoryData(self, save_path):
		print("Saving data to: '{}'".format(save_path))
		if not os.path.isdir(self.data_path ):
			os.makedirs(self.args.data_path )

		# Reconstruct interpolators since they were not pickled with the rest of the trajectory
		for id, traj in self.trajectory_set:
			traj.updateInterpolators()

		#if "test" not in self.args.scenario:
		random.shuffle(self.trajectory_set)

		self.compute_min_max_values()

		self.shift_data()

		data = {
			"trajectories" : self.trajectory_set,
			"agent_container" : self.agent_container,
			"min_pos_x" : self.min_pos_x,
			"min_pos_y" : self.min_pos_y,
			"max_pos_x" : self.max_pos_x,
			"max_pos_y" : self.max_pos_y,
			"min_vel_x" : self.min_vel_x,
			"min_vel_y" : self.min_vel_y,
			"max_vel_x" : self.max_vel_x,
			"max_vel_y" : self.max_vel_y,
			"mean_pos_x" : self.mean_pos_x,
			"mean_pos_y" : self.mean_pos_y,
		}
		pkl.dump(data, open(save_path, 'wb'),protocol=2)

	def loadTrajectoryData(self, load_path):
		print("Loading data from: '{}'".format(load_path))
		self.file = open(load_path, 'rb')
		if sys.version_info[0] < 3:
			tmp_self = pkl.loads(self.file,encoding='latin1')
		else:
			tmp_self = pkl.load(self.file , encoding='latin1')
		self.trajectory_set = tmp_self["trajectories"]
		self.agent_container = tmp_self["agent_container"]

		#self.compute_min_max_values()
		self.min_pos_x = tmp_self["min_pos_x"]
		self.min_pos_y = tmp_self["min_pos_y"]
		self.max_pos_x = tmp_self["max_pos_x"]
		self.max_pos_y = tmp_self["max_pos_y"]
		self.min_vel_x = tmp_self["min_vel_x"]
		self.min_vel_y = tmp_self["min_vel_y"]
		self.max_vel_x = tmp_self["max_vel_x"]
		self.max_vel_y = tmp_self["max_vel_y"]
		self.mean_pos_x = tmp_self["mean_pos_x"]
		self.mean_pos_y =tmp_self["mean_pos_y"]

		# Dataset Statistics
		cnt = 0
		avg_len = 0
		for traj_id in tqdm(range(len(self.trajectory_set))):
			avg_len = (avg_len*cnt+self.trajectory_set[traj_id][1].pose_vec.shape[0])/(cnt+1)

		print("Avg. Trajectory Length: " + str(avg_len))
		print("Total number of trajectories: " + str(len(self.trajectory_set)))

		# Reconstruct interpolators since they were not pickled with the rest of the trajectory
		for id, traj in self.trajectory_set:
			traj.updateInterpolators()

	def getAgentTrajectory(self, agent_id):
		"""
		Return the next agent trajectory in the queue for the agent with id agent_id.
		"""
		trajectory = self.agent_container.agent_data[agent_id].trajectories[self.agent_traj_idx[agent_id]]
		self.agent_traj_idx[agent_id] = (self.agent_traj_idx[agent_id] + 1)  % self.agent_container.getNumberOfTrajectoriesForAgent(agent_id)
		return trajectory

	def getRandomAgentTrajectory(self, agent_id):
		"""
		Return a totally random trajectory for the agent with id agent_id.
		"""
		random_traj_idx = np.random.randint(0, len(self.agent_container.agent_data[agent_id].trajectories))
		return self.agent_container.agent_data[agent_id].trajectories[random_traj_idx]

	def getRandomTrajectory(self):
		"""
		Return a totally random trajectory.
		"""
		random_traj_idx = np.random.randint(0, len(self.trajectory_set))
		return self.trajectory_set[random_traj_idx]


