import numpy as np
from copy import deepcopy

from gym_collision_avoidance.envs.policies.Policy import Policy
from gym_collision_avoidance.envs.information_models.edfMap import edfMap
from gym_collision_avoidance.envs.information_models.targetMap import targetMap

from gym_collision_avoidance.envs.policies.pydecmcts.DecMCTS import Tree


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

        self.ego_agent = None

        self.tree = None
        self.best_paths = None
        self.obsvd_targets = None

    def init_maps(self, ego_agent, occ_map, map_size, map_res, detect_fov, detect_range, dt=0.1):

        self.ego_agent = ego_agent

        self.detect_range = detect_range
        self.detect_fov = detect_fov
        self.DT = dt

        # Init EDF and Target Map
        self.edfMap = edfMap(occ_map, map_res, map_size)
        self.targetMap = targetMap(self.edfMap, map_size, map_res * 5,
                                   sensFOV=self.detect_fov * np.pi / 180, sensRange=self.detect_range, rOcc=1.5,
                                   rEmp=0.66)

    def find_next_action(self, obs, agents, agent_id, obstacle):

        global_pose = np.append(obs['pos_global_frame'], obs['heading_global_frame'])

        dmcts_agents = [i for i in range(len(agents)) if "ig_" in str(type(agents[i].policy)) and i != agent_id]

        targets = []
        # Find Targets in Range and FOV (Detector Emulation)
        self.obsvd_targets = self.find_targets_in_obs(obs)
        targets.append(self.obsvd_targets)
        # Get observations of other agens
        for j in dmcts_agents:
            targets.append(agents[j].policy.obsvd_targets)

        # Update Target Map
        self.targetMap.update(global_pose, self.obsvd_targets, frame='global')

        comm_n = 5
        data = {"current_pose": global_pose}

        self.tree = None

        if self.tree is None:
            self.tree = Tree(data, self.mcts_reward, self.mcts_avail_actions, self.mcts_state_storer,
                             self.mcts_sim_selection_func, self.mcts_avail_actions, self.mcts_sim_state_storer, comm_n,
                             robot_id=agent_id, horizon=10, c_p=0.5)
        else:
            self.tree.prune_tree()

        for i in range(100):
            self.tree.grow()
            #collect communications
            for j in range(len(dmcts_agents)):
                self.tree.receive_comms(agents[j].policy.tree.send_comms(), j)

        self.best_paths = self.tree.send_comms()

        action = self.best_paths.X[0].action_seq[0]

        return action

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
        next_pose = pose + np.append(vel, dphi) * self.DT
        # Check if Next Pose is within map
        in_map = (self.targetMap.mapSize / 2 > next_pose[0:2]).all() and (next_pose[0:2] > -self.targetMap.mapSize / 2).all()
        # Check if Next Pose is Obstacle
        edf_next_pose = self.targetMap.edfMapObj.get_edf_value_from_pose(next_pose)
        if edf_next_pose > self.ego_agent.radius + 0.1 and in_map:
            return next_pose
        else:
            return None

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
        obsvd_cells = set()
        for key in states:
            obsvd_cells.update(states[key].obsvd_cells)
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
