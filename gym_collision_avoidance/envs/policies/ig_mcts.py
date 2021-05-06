import numpy as np
from copy import deepcopy

from gym_collision_avoidance.envs.policies.Policy import Policy
from gym_collision_avoidance.envs.information_models.edfMap import edfMap
from gym_collision_avoidance.envs.information_models.targetMap import targetMap

from gym_collision_avoidance.envs.policies.pydecmcts.DecMCTS import Tree

import os

class MCTS_state:
    def __init__(self, act_seq, pose_seq, visib_cells, obsvd_cells, cum_reward):
        self.action_seq = act_seq
        self.pose_seq = pose_seq
        self.cum_reward = cum_reward
        self.visib_cells = visib_cells
        self.obsvd_cells = obsvd_cells


class ig_mcts(Policy):
    def __init__(self):
        Policy.__init__(self, str="ig_mcts")

        self.cooperation_coef = 0
        self.edfMap = None
        self.targetMap = None
        self.detect_fov = None
        self.detect_range = None
        self.DT = None
        self.xDT = None

        self.ego_agent = None

        self.tree = None
        self.best_paths = None
        self.obsvd_targets = None

        self.parallize = False

        self.Ntree = 100
        self.Nsims = 10
        self.parallelize_sims = False
        self.mcts_cp = 1.
        self.mcts_horizon = 10

    def set_param(self, ego_agent, occ_map, map_size, map_res, detect_fov, detect_range, dt=0.1, xdt=1,
                  Ntree=100, Nsims=10, parallelize_sims=False, mcts_cp=1., mcts_horizon=10,
                  parallelize_agents=False):

        self.ego_agent = ego_agent

        self.detect_range = detect_range
        self.detect_fov = detect_fov
        self.DT = dt
        self.xDT = xdt

        # Init EDF and Target Map
        self.edfMap = edfMap(occ_map, map_res, map_size)
        self.targetMap = targetMap(self.edfMap, map_size, map_res * 5,
                                   sensFOV=self.detect_fov * np.pi / 180, sensRange=self.detect_range, rOcc=1.5,
                                   rEmp=0.66)
        self.Nsims = Nsims
        self.Ntree = Ntree
        self.parallelize_sims = parallelize_sims
        self.mcts_cp = mcts_cp
        self.mcts_horizon = mcts_horizon

        self.parallize = parallelize_agents

    def find_next_action(self, obs, agents, agent_id, obstacle):

        global_pose = np.append(obs['pos_global_frame'], obs['heading_global_frame'])

        dmcts_agents = [i for i in range(len(agents)) if "ig_" in str(type(agents[i].policy)) and i != agent_id]

        self.update_belief(dmcts_agents, global_pose, agents)

        comm_n = 5
        data = {"current_pose": global_pose}

        self.tree = None

        if self.tree is None:
            self.tree = Tree(data, self.mcts_reward, self.mcts_avail_actions, self.mcts_state_storer,
                             self.mcts_sim_selection_func, self.mcts_avail_actions, self.mcts_sim_state_storer, comm_n,
                             robot_id=agent_id, horizon=self.mcts_horizon, c_p=self.mcts_cp)
        else:
            self.tree.prune_tree()

        # collect communications
        for j in range(len(dmcts_agents)):
            if agents[j].policy.best_paths is not None:
                self.tree.receive_comms(agents[j].policy.best_paths, j)

        for i in range(self.Ntree):
            self.tree.grow(nsims=self.Nsims, gamma=1, parallelize=self.parallelize_sims)
            # #collect communications
            # for j in range(len(dmcts_agents)):
            #     self.tree.receive_comms(agents[j].policy.tree.send_comms(), j)

        self.best_paths = self.tree.send_comms()

        action = self.best_paths.X[0].action_seq[0]

        return action

    def parallel_next_action(self, obs, agents, agent_id, obstacle, send_end):
        action = self.find_next_action(obs, agents, agent_id, obstacle)
        print("This is the id of the agent", agent_id, "child process", os.getpid())
        send_end.send({"policy_obj": self, "action": action})
        send_end.close()

    def update_belief(self, dmcts_agents, global_pose, agents):
        targets = []
        poses = []
        # Find Targets in Range and FOV (Detector Emulation)
        self.obsvd_targets = self.find_targets_in_obs(obs)
        targets.append(self.obsvd_targets)
        poses.append(global_pose)
        # Get observations of other agens
        for j in dmcts_agents:
            other_agent_targets = agents[j].policy.obsvd_targets if agents[j].policy.obsvd_targets is not None else []
            targets.append(other_agent_targets)
            other_agent_pose = np.append(agents[j].pos_global_frame, agents[j].heading_global_frame)
            poses.append(other_agent_pose)

        # Update Target Map
        self.targetMap.update(poses, targets, frame='global')

    def find_targets_in_obs(self, obs):
        global_pose = np.append(obs['pos_global_frame'], obs['heading_global_frame'])

        # Find Targets in Range and FOV (Detector Emulation)
        targets = []
        for other_agent in obs['other_agents_states']:
            if other_agent[9] == 1.0:
                # Static Agent = Target
                # r = other_agent[0:2] - global_pose[0:2]
                r = other_agent[0:2]
                dphi = np.arctan2(r[1], r[0]) - global_pose[2]
                in_fov = abs(dphi) <= self.detect_fov / 2.0
                r_norm = np.sqrt(r[0] ** 2 + r[1] ** 2)
                in_range = r_norm <= self.detect_range
                if in_fov and in_range:
                    targets.append(other_agent[0:2] + global_pose[0:2])

        return targets

    def get_next_pose(self, pose, action):
        # Get velocity vector in world frame
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array(((c, -s), (s, c)))
        vel = np.dot(R, np.array([action[0], 0.0]))
        # First Order Dynamics for Next Pose
        dphi = action[1]
        u = np.append(vel, dphi)
        next_pose = pose
        for i in range(self.xDT):
            next_pose = next_pose + u * self.DT
            # Check if Next Pose is within map
            in_map = (self.targetMap.mapSize / 2 > next_pose[0:2]).all() and (next_pose[0:2] > -self.targetMap.mapSize / 2).all()
            if in_map:
                # Check if Next Pose is Obstacle
                edf_next_pose = self.targetMap.edfMapObj.get_edf_value_from_pose(next_pose)
                if edf_next_pose > self.ego_agent.radius + 0.1:
                    continue
                else:
                    return None
            else:
                return None

        return next_pose

    def mcts_state_storer(self, data, parent_state, action, robot_id, root_pose=None):
        # If first call create State object
        if parent_state is None:
            return MCTS_state([], [data["current_pose"]], set(), set(),
                              0)  # This state is also used Null action when calculating local reward

        # Compute state transition
        parent_pose = parent_state.pose_seq[-1]
        next_pose = self.get_next_pose(parent_pose, action)

        # Create new state if next pose feasible

        if next_pose is not None:
            state = deepcopy(parent_state)
            visible_cells = self.targetMap.getVisibleCells(next_pose)
            state.visib_cells = visible_cells
            state.obsvd_cells.update(visible_cells)
            state.action_seq.append(action)
            state.pose_seq.append(next_pose)
        else:
            state = None
        return state

    def mcts_sim_state_storer(self, data, parent_state, action, robot_id, root_pose=None):
        # If first call create State object
        if parent_state is None:
            return MCTS_state([], [data["current_pose"]], set(), set(),
                              0)  # This state is also used Null action when calculating local reward

        # Compute state transition
        parent_pose = parent_state.pose_seq[-1]
        next_pose = self.get_next_pose(parent_pose, action)

        # Create new state
        state = deepcopy(parent_state)
        if next_pose is not None:
            visible_cells = self.targetMap.getVisibleCells(next_pose)
            state.visib_cells = visible_cells
            state.obsvd_cells.update(visible_cells)
            state.action_seq.append(action)
            state.pose_seq.append(next_pose)
        else:
            visible_cells = set()
            state.visib_cells = visible_cells
            # state.obsvd_cells.update(visible_cells)
            state.action_seq.append(np.array([0.0, 0.0]))
            state.pose_seq.append(parent_pose)
        return state

    def mcts_reward(self, data, states, id):
        other_agents_obsvd_cells = set()
        for key in states:
            if key == id:
                continue
            other_agents_obsvd_cells.update(states[key].obsvd_cells)
        obsvd_cells = states[id].obsvd_cells.difference(other_agents_obsvd_cells)
        return self.targetMap.get_reward_from_cells(obsvd_cells)

    @staticmethod
    def mcts_sim_selection_func(data, options, temp_state):
        return options[np.random.choice(len(options))]

    @staticmethod
    def mcts_avail_actions(data, state, robot_id):
        vel_list = [0.0, 2.0, 4.0]
        dphi_list = [-np.pi, 0, np.pi]

        action_list = [np.array([vel, dphi]) for vel in vel_list for dphi in dphi_list]
        return action_list
