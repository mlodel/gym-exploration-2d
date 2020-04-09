import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy
import rospy
from lmpcc.srv import *
from geometry_msgs.msg import PoseStamped, Twist

class MPCPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="MPC")
        self.is_still_learning = True
        self.ppo_or_learning_policy = True
        self.debug = False
        self.service_topic = "/mpc/plan_srv"
        self.plan_client = rospy.ServiceProxy(self.service_topic,LMPCCPlan)

        self.goal = PoseStamped()
        self.state = PoseStamped()
        self.request = LMPCCPlanRequest()
        self.resp = Twist()

    def network_output_to_action(self, agent, network_output):
        if self.debug:
            print("Waiting for " + self.service_topic)
            rospy.wait_for_service("/mpc/plan_srv")
            print("Found " + self.service_topic)

        # network_output: [x-position 0-1, y-position btwn 0-1]
        self.goal.pose.position.x = network_output[0]
        self.goal.pose.position.y = network_output[1]

        self.state.pose.position.x = agent.pos_global_frame[0]
        self.state.pose.position.y = agent.pos_global_frame[1]

        self.request.goal = self.goal
        self.request.robot_state = self.state

        self.resp = self.plan_client(self.request)

        agent.set_state(px=self.resp.next_robot_state.position.x,py=self.resp.next_robot_state.position.y)

        return np.array([self.resp.control_cmd.linear.x, self.resp.control_cmd.angular.z])

    def find_next_action(self, obs, agents, i):
        raise NotImplementedError
