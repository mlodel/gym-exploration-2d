# from pydecmcts import Tree
from copy import deepcopy
import numpy as np
import cProfile
import pstats

from pydecmcts.DecMCTS import Tree

# importing networkx
import networkx as nx

# importing matplotlib.pyplot
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


data = {}


class State:
    def __init__(self, act_seq, cum_sum):
        self.action_seq = act_seq
        self.cumulative_sum = cum_sum


def state_storer(data, parent_state, action, robot_id):
    
    if parent_state == None:
        return State([],0) # This state is also used Null action when calculating local reward

    state = deepcopy(parent_state)
    state.action_seq.append(action)
    state.cumulative_sum = state.cumulative_sum + action
    return state


def avail_actions(data, state, robot_id):
    return [1,2,3,4,5]


def sim_selection_func(data, options, temp_state):
    return np.random.choice(options)


def reward(dat, states):
    each_robot_sum= [states[robot].cumulative_sum for robot in states]
    return sum(each_robot_sum)


comm_n = 5

tree1 = Tree(data, reward, avail_actions, state_storer, sim_selection_func, avail_actions, comm_n, 1, 10)

# tree2 = Tree(data, reward, avail_actions, state_storer, sim_selection_func, avail_actions, comm_n, 2, 10)

for i in range(100):
    tree1.grow()
    # tree2.grow()
    # tree1.receive_comms(tree2.send_comms(), 2)
    # tree2.receive_comms(tree2.send_comms(), 1)
print("End")

# start_node = 1
# while len(tree1._childNodes(start_node))>0:
#     start_node = tree1._select(tree1._childNodes(start_node))

G = tree1.graph
nx.nx_agraph.write_dot(G,'test.dot')

# same layout using matplotlib with no labels
plt.title('draw_networkx')
pos=graphviz_layout(G, prog='dot')
nx.draw(G, pos, with_labels=False, arrows=False)
# plt.savefig('nx_test.png')

plt.show()