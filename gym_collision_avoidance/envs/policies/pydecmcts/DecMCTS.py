from __future__ import print_function

import networkx as nx
from copy import copy
from math import log
import numpy as np
import multiprocessing
from functools import partial



from copy import deepcopy

def _UCT(mu_j, c_p, n_p, n_j):
    if n_j == 0:
        return float("Inf")

    return mu_j + 2 * c_p * (2 * log(n_p) / n_j) ** 0.5


class ActionDistribution:
    """
    Action Distribution
    Working with action sequences and their respective probability

    To initialise, Inputs:
    - X: list of action sequences
        - NOTE: X is is simply a state object returned by state_store.
            You are expected to store action sequence in this object
    - q: probability of each action sequence (normalised in intialisation)

    """

    def __init__(self, X, q):

        # Action sequence as provided
        self.X = X

        # Normalise 
        # if sum(q) == 0:
        #     self.q = [1 / float(len(q))] * len(q)
        # else:
        #     self.q = (np.array(q).astype(float) / sum(q)).tolist()

    def best_action(self):
        """
        Most likely action sequence
        """
        return self.X[np.argmax(self.q)]

    def random_action(self):
        """
        Weighted random out of possible action sequences
        """
        return self.X[np.random.choice(len(self.q), p=self.q)]


class Tree:
    """
    DecMCTS tree
    To Initiate, Inputs:
    - data
        - data required to calculate reward, available options
    - reward
        - This is a function which has inputs (data, state) and
            returns the GLOBAL reward to be maximised
        - MUST RETURN POSITIVE VALUE
    - available_actions
        - This is a function which has inputs (data, state) and
            returns the possible actions which can be taken
    - state_store
        - This is a function which has inputs 
            (data, parent_state, action) and returns an object to
            store in the node. 
        - Root Node has parent state None and action None.
    - sim_selection_func
        - This is a function which chooses an available of option
            during simulation (can be random or more advanced)
    - c_p
        - exploration multiplier (number between 0 and 1)

    Usage:
    - grow
        - grow MCTS tree by 1 node
    - send_comms
        - get state of this tree to communicate to others
    - receive_comms
        - Input the state of other trees for use in calculating
            reward/available actions in coordination with others
    """

    def __init__(self,
                 data,
                 reward_func,
                 avail_actions_func,
                 state_store_func,
                 sim_selection_func,
                 sim_avail_actions_func,
                 sim_state_store_func,
                 comm_n,
                 robot_id,
                 horizon,
                 c_p=0.01):

        self.data = data
        self.graph = nx.DiGraph()
        self.reward = reward_func
        self.available_actions = avail_actions_func
        self.sim_available_actions = sim_avail_actions_func
        self.state_store = state_store_func
        self.sim_state_store = sim_state_store_func
        self.sim_selection_func = sim_selection_func
        self.c_p = c_p
        self.id = robot_id
        self.horizon = horizon
        self.comms = {}  # Plan with no robots initially
        self.comm_n = comm_n  # number of action dists to communicate
        self.root_id = 1

        # Graph add root node of tree
        self.graph.add_node(1,
                            mu=0,
                            N=0,
                            best_reward=0,
                            state=self.state_store(self.data, None, None, self.id),
                            stage=0
                            )

        # Set Action sequence as nothing for now
        self.my_act_dist = ActionDistribution([self.graph.nodes[1]["state"]], [1])

        self._expansion(1)

    def _parent(self, node_id):
        """
        wrapper for code readability
        """
        return list(self.graph.predecessors(node_id))[0]

    def _select(self, children):
        """
        Select Child node which maximises UCT
        """

        # N for parent
        n_p = self.graph.nodes[self._parent(children[0])]["N"]

        # UCT values for children
        uct = [_UCT(node["mu"], self.c_p, n_p, node["N"])
               for node in map(self.graph.nodes.__getitem__, children)]

        # Return Child with highest UCT
        return children[np.argmax(uct)]

    def _childNodes(self, node_id):
        """
        wrapper for code readability
        """

        return list(self.graph.successors(node_id))

    def _update_distribution(self):
        """
        Get the top n Action sequences and their "probabilities"
            and store them for communication
        """

        # For now, just using q = mu**2
        temp = nx.get_node_attributes(self.graph, "mu")
        temp.pop(1, None)

        top_n_nodes = sorted(temp, key=temp.get, reverse=True)[:self.comm_n]
        X = [self.graph.nodes[n]["best_rollout"] for n in top_n_nodes if "best_rollout" in self.graph.nodes[n]]
        q = [self.graph.nodes[n]["mu"] ** 2 for n in top_n_nodes if "best_rollout" in self.graph.nodes[n]]

        if len(q) == 0:
            a=1

        self.my_act_dist = ActionDistribution(X, q)
        return True

    def _get_system_state(self, node_id):
        """
        Randomly select 1 path taken by every other robot & path taken by
            this robot to get to this node

        Returns tuple where first element is path of current robot,
            and second element is a dictionary of the other paths
        """

        system_state = {k: self.comms[k].random_action() for k in self.comms}
        system_state[self.id] = self.graph.nodes[node_id]["state"]

        return system_state

    def _null_state(self, state):
        temp = copy(state)
        temp[self.id] = self.graph.nodes[1]["state"]  # Null state is if robot still at root node
        return temp

    def _expansion(self, start_node):
        """
        Does the Expansion step for tree growing.
        Separated into it's own function because also done in
            Init step.
        """

        options = self.available_actions(
            self.data,
            self.graph.nodes[start_node]["state"],
            self.id
        )

        stage = self.graph.nodes[start_node]["stage"]

        if len(options) == 0 or stage == self.horizon:
            return False

        # create empty nodes underneath the node being expanded
        for o in options:
            new_state = self.state_store(self.data, self.graph.nodes[start_node]["state"], o, self.id)
            if new_state is not None:
                self.graph.add_node(len(self.graph) + 1,
                                    mu=0,
                                    best_reward=0,
                                    N=0,
                                    state=new_state,
                                    stage=stage + 1
                                    )
                self.graph.add_edge(start_node, len(self.graph))
        return True

    def _simulate(self, start_node, state, send_end=None, sema=None):
        if sema is not None:
            sema.acquire()

        temp_state = self.graph.nodes[start_node]["state"]
        state[self.id] = temp_state

        d = self.graph.nodes[start_node]["stage"]
        while d < self.horizon:  # also breaks at no available options
            d += 1

            # Get the available actions
            options = self.sim_available_actions(
                self.data,
                state[self.id],
                self.id
            )

            # If no actions possible, simulation complete
            if len(options) == 0:
                break

            # "randomly" choose 1 - function provided by user
            # state[self.id] = temp_state
            sim_action = self.sim_selection_func(self.data, options, temp_state)

            # add that to the actions of the current robot
            temp_state = self.sim_state_store(self.data, temp_state, sim_action, self.id)

            state[self.id] = temp_state

        # calculate the reward at the end of simulation
        rew = self.reward(self.data, state, self.id)
        if send_end is not None:
            send_end.send({"reward": rew, "temp_state": temp_state})
        else:
            return {"reward": rew, "temp_state": temp_state}
        if sema is not None:
            sema.release()

    def grow(self, nsims=10, gamma=0.9, parallelize=False):
        """
        Grow Tree by one node
        gamma between 0.5 and 1
        """

        ### SELECTION
        start_node = self.root_id

        # Sample actions of other robots
        # NOTE: Sampling done at the begining for dependency graph reasons
        state = self._get_system_state(start_node)

        # Propagate down the tree
        # check how _select handles mu, N = 0
        while len(self._childNodes(start_node)) > 0:
            start_node = self._select(self._childNodes(start_node))

        ### EXPANSION
        # check if _expansion changes start_node to the node after jumping
        self._expansion(start_node)
        # print("Grow!")

        ### SIMULATION
        avg_reward = 0
        best_reward = float("-Inf")
        best_rollout = None
        # Parallization

        if parallelize:
            processes = []
            pipe_list = []
            mp_context = multiprocessing.get_context("fork")
            sema = multiprocessing.Semaphore(12)
            for i in range(nsims):
                recv_end, send_end = mp_context.Pipe(False)
                p = mp_context.Process(target=self._simulate,
                                            args=(start_node, state, send_end, sema))
                processes.append(p)
                pipe_list.append(recv_end)
                p.start()

        for i in range(nsims):
            if parallelize:
                recvd = pipe_list[i].recv()
                processes[i].join()
            else:
                recvd = self._simulate(start_node, state)
            rew = recvd["reward"]
            temp_state = recvd["temp_state"]
            avg_reward += rew

            # new_return = 0
            # for j in reversed(range(rew.len())):
            #     new_return = rew[j] + gamma * new_return
            # avg_reward += new_return
            # if best reward so far, store the rollout in the new node
            if rew > best_reward:
                best_reward = rew
                best_rollout = copy(temp_state)

        avg_reward = avg_reward / nsims

        self.graph.nodes[start_node]["mu"] = avg_reward
        self.graph.nodes[start_node]["best_reward"] = best_reward
        self.graph.nodes[start_node]["N"] = 1
        self.graph.nodes[start_node]["best_rollout"] = deepcopy(best_rollout)

        ### BACKPROPOGATION
        while start_node != 1:  # while not root node

            start_node = self._parent(start_node)

            self.graph.nodes[start_node]["mu"] = \
                (gamma * self.graph.nodes[start_node]["mu"] * \
                 self.graph.nodes[start_node]["N"] + avg_reward) \
                / (self.graph.nodes[start_node]["N"] + 1)

            self.graph.nodes[start_node]["N"] = \
                gamma * self.graph.nodes[start_node]["N"] + 1

            if best_reward > self.graph.nodes[start_node]["best_reward"]:
                self.graph.nodes[start_node]["best_reward"] = best_reward
                self.graph.nodes[start_node]["best_rollout"] = deepcopy(best_rollout)

        self._update_distribution()

        return avg_reward



    def send_comms(self):
        return self.my_act_dist

    def receive_comms(self, comms_in, robot_id):
        """
        Save data which has been communicated to this tree
        Only receives from one robot at a time, call once
        for each robot

        Inputs:
        - comms_in
            - An Action distribution object
        - robot_id
            - Robot number/id - used as key for comms
        """
        self.comms[robot_id] = comms_in
        return True

    def prune_tree(self):
        best_action = self.my_act_dist.X[0].action_seq[0]
        for i in list(self.graph.successors(self.root_id)):
            if (self.graph.nodes[i]["state"].action_seq[0] == best_action).all():
                pass
            else:
                T = nx.dfs_tree(self.graph, source=i)
                self.graph.remove_nodes_from(T)
                self.graph.remove_edges_from(T.edges)
                # self.graph.remove_edge(self.root_id, i)
        root_vis_cells = self.graph.nodes[self.root_id]["state"].visib_cells
        self.graph.remove_node(self.root_id)
        self.graph = nx.convert_node_labels_to_integers(self.graph, first_label=1)
        for i in list(self.graph.__iter__()):
            self.graph.nodes[i]["stage"] -= 1
            self.graph.nodes[i]["state"].action_seq.pop(0)
            self.graph.nodes[i]["state"].pose_seq.pop(0)
            self.graph.nodes[i]["state"].obsvd_cells.difference_update(root_vis_cells)
            self.graph.nodes[i]["best_reward"] = 0
            if "best_rollout" in self.graph.nodes[i]:
                # if len(self.graph.nodes[i]["best_rollout"].action_seq) < self.horizon:
                #     a=1
                if self.graph.nodes[i]["best_rollout"].action_seq:
                    self.graph.nodes[i]["best_rollout"].action_seq.pop(0)
                    self.graph.nodes[i]["best_rollout"].pose_seq.pop(0)
                    self.graph.nodes[i]["best_rollout"].obsvd_cells.difference_update(root_vis_cells)
                else:
                    self.graph.nodes[i].pop("best_rollout")
