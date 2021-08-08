import numpy as np
from copy import deepcopy

# from gym_collision_avoidance.envs.information_models.ig_agent import ig_agent
# from gym_collision_avoidance.envs.agent import Agent

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


class ig_mcts():
    def __init__(self, ig_model):

        self.ig_model = ig_model
        self.ego_agent = self.ig_model.host_agent

        self.tree = None
        self.best_paths = None
        self.obsvd_targets = None
        self.global_pose = None

        self.parallize = False
        self.parallelize_sims = False

        self.Ntree = 50
        self.Ncycles = 10
        self.Nsims = 10
        self.mcts_cp = 1.
        self.mcts_horizon = 4
        self.mcts_gamma = 0.99
        self.cooperation_coef = 0
        self.DT = 0.1
        self.xDT = 12

    def set_param(self, ego_agent, dt=0.1, xdt=1,
                  Ntree=100, Nsims=10, parallelize_sims=False, mcts_cp=1., mcts_horizon=10, mcts_gamma=0.99, Ncycles=5,
                  parallelize_agents=False):

        self.DT = dt
        self.xDT = xdt

        self.Nsims = Nsims
        self.Ntree = Ntree
        self.Ncycles = Ncycles
        self.parallelize_sims = parallelize_sims
        self.mcts_cp = mcts_cp
        self.mcts_horizon = mcts_horizon
        self.mcts_gamma = mcts_gamma

        self.parallize = parallelize_agents

    def get_expert_goal(self, new_step=True):

        self.global_pose = np.append(self.ig_model.host_agent.pos_global_frame,
                                     self.ig_model.host_agent.heading_global_frame)

        # dmcts_agents = [i for i in range(len(agents)) if "ig_mcts" in str(type(agents[i].ig_model.expert_policy))
        #                 and i != agent_id]
        dmcts_agents = []

        comm_n = 5
        data = {"current_pose": self.global_pose}
        if new_step:
            self.tree = Tree(data, self.mcts_reward, self.mcts_avail_actions, self.mcts_state_storer,
                             self.mcts_sim_selection_func, self.mcts_avail_actions, self.mcts_sim_state_storer, comm_n,
                             robot_id=self.ig_model.host_agent.id, horizon=self.mcts_horizon, c_p=self.mcts_cp)

        # collect communications
        # for j in range(len(dmcts_agents)):
        #     if agents[j].policy.best_paths is not None:
        #         self.tree.receive_comms(agents[j].policy.best_paths, j)

        for i in range(self.Ntree):
            self.tree.grow(nsims=self.Nsims, gamma=self.mcts_gamma, parallelize=self.parallelize_sims)
            # #collect communications
            # for j in range(len(dmcts_agents)):
            #     self.tree.receive_comms(agents[j].policy.tree.send_comms(), j)

        self.best_paths = self.tree.send_comms()

        if len(self.best_paths.X[0].pose_seq) > 0:
            goal = self.best_paths.X[0].pose_seq[1]
        else:
            goal = self.best_paths.X[0].pose_seq[0]

        return goal

    def parallel_next_action(self, obs, agents, agent_id, obstacle, send_end, new_step=True):
        action = self.get_expert_goal(new_step)
        # print("This is the id of the agent", agent_id, "child process", os.getpid())
        send_end.send({"policy_obj": self, "action": action})
        send_end.close()

    def get_next_pose(self, pose, action):

        next_pose = pose
        for i in range(self.xDT):

            # Get velocity vector in world frame
            c, s = np.cos(next_pose[2]), np.sin(next_pose[2])
            R = np.array(((c, -s), (s, c)))
            vel = np.dot(R, np.array([action[0], 0.0]))
            # First Order Dynamics for Next Pose
            dphi = action[1]
            u = np.append(vel, dphi)

            next_pose = next_pose + u * self.DT
            if action[0] == 0.0:
                continue
            else:
                # Check if Next Pose is within map
                in_map = (self.ig_model.targetMap.mapSize / 2 > next_pose[0:2]).all() \
                         and (next_pose[0:2] > -self.ig_model.targetMap.mapSize / 2).all()
                if in_map:
                    # Check if Next Pose is Obstacle
                    edf_next_pose = self.ig_model.targetMap.edfMapObj.get_edf_value_from_pose(next_pose)
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
            visible_cells = self.ig_model.targetMap.getVisibleCells(next_pose)
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
            visible_cells = self.ig_model.targetMap.getVisibleCells(next_pose)
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
        return self.ig_model.targetMap.get_reward_from_cells(obsvd_cells)

    @staticmethod
    def mcts_sim_selection_func(data, options, temp_state):
        return options[np.random.choice(len(options))]

    @staticmethod
    def mcts_avail_actions(data, state, robot_id):
        vel_list = [0., 1.0, 3.0]
        dphi_list = [-0.25*np.pi, -0.1*np.pi, 0, 0.1*np.pi, 0.25*np.pi]

        action_list = [np.array([vel, dphi]) for vel in vel_list for dphi in dphi_list]
        return action_list
