import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy
from gym_collision_avoidance.envs.policies.static_obstacles.StaticObstacleManager import StaticObstacleManager
from gym_collision_avoidance.envs.policies.mpc import StaticMPCsolver_py
from gym_collision_avoidance.envs.config import Config
class MPCRLStaticObsPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="MPCRLStaticObsPolicy")
        self.is_still_learning = True
        self.ppo_or_learning_policy = True
        self.debug = False
        # Goal Parameterization
        self.goal_ = np.zeros([2])
        self.current_state_ = np.zeros([5])
        self.next_robot_state = np.zeros([5])
        self.control_cmd_linear = 0.0
        self.control_cmd_angular = 0.0
        self.dt = 0.2
        self.subgoal = np.zeros([2])

        # FORCES parameters
        self.M = 4
        self.FORCES_N = 15
        self.FORCES_NU = 3
        self.FORCES_NX = 5
        self.FORCES_TOTAL_V = 8
        self.FORCES_NPAR = 82
        self.FORCES_x0 = np.zeros(int(self.FORCES_TOTAL_V*self.FORCES_N), dtype="double")
        self.FORCES_xinit = np.zeros(self.FORCES_NX, dtype="double")
        self.FORCES_all_parameters = np.zeros(int(self.FORCES_N*self.FORCES_NPAR), dtype="double")

        # Dynamic weights
        self.x_error_weight_ = 2.0
        self.y_error_weight_ = 2.0
        self.velocity_weight_ = 0.01
        self.acceleration_weigth_ = 0.001
        self.angular_velocity_weight_ = 0.01
        self.angular_acc_weight_ = 0.001
        self.theta_error_weight_ = 0.0
        self.reference_velocity_ = 1.0
        self.reference_ang_velocity_ = 0.0
        self.slack_weight_ = 10000
        self.repulsive_weight_ = 0.0
        self.n_obstacles_ = 6

        # Cost function approximation coefficients
        self.coefs = np.zeros([6])
        self.d = 0
        self.cost_function_weight = 0

        self.enable_collision_avoidance = True

        self.policy_name = "MPC"

        self.predicted_traj = np.zeros((self.FORCES_N, 2))
        self.guidance_traj = np.zeros((self.FORCES_N, 2))

        self.map = None
        #self.static_obstacles = StaticObstacleAvoidance()

        self.static_obstacles_manager = StaticObstacleManager()

        self.solve_time = 0

    def __str__(self):
        return "MPCRLStaticObsPolicy"

    def network_output_to_action(self, id, agents,network_output):
        agent = agents[id]
        other_agents = []
        for other_agent in agents:
            if other_agent.id != id:
                other_agents.append(other_agent)

        # network_output: [x-position 0-1, y-position btwn 0-1]
        if np.linalg.norm(agent.pos_global_frame-agent.goal_global_frame) > Config.NEAR_GOAL_THRESHOLD :
            self.goal_[0] = agent.pos_global_frame[0] + network_output[0] # agent.goal_global_frame[0]
            self.goal_[1] = agent.pos_global_frame[1] + network_output[1] # agent.goal_global_frame[1]
        else:
            self.enable_collision_avoidance = True
            self.goal_[0] = agent.goal_global_frame[0]
            self.goal_[1] = agent.goal_global_frame[1]

        agent.next_goal = self.goal_

        self.current_state_[0] = agent.pos_global_frame[0]
        self.current_state_[1] = agent.pos_global_frame[1]
        self.current_state_[2] = agent.heading_global_frame
        self.current_state_[3] = np.clip(np.linalg.norm(agent.vel_global_frame),0,1)
        self.current_state_[4] = np.clip(agent.angular_speed_global_frame,-2,2)

        # Compute Static Collision Constraints
        self.linear_constraints = self.static_obstacles_manager.get_linear_constraints(agent)

        if not self.enable_collision_avoidance:
            self.linear_constraints = np.array([[1, 0, 50, 0],[0, 1, 50, 0],[-1, 0, 50,0 ],[0, -1, 50,0]])

        exit_flag, _ = self.run_solver(agent,other_agents, enable_constraints = self.enable_collision_avoidance, update=True)
        if exit_flag != 1:
            exit_flag, output = self.run_solver(agent,other_agents, enable_constraints = self.enable_collision_avoidance, update=True)
        if exit_flag == 1:
            agent.is_infeasible = False
            return np.array([self.FORCES_x0[0], self.FORCES_x0[1]])
        else:
            agent.is_infeasible = True
            # Compute break command
            return np.array([-self.current_state_[3],-self.current_state_[4]])

    def mpc_output(self, id, agents):
        agent = agents[id]
        other_agents = []
        for other_agent in agents:
            if other_agent.id != id:
                other_agents.append(other_agent)

        self.goal_[0] = agent.goal_global_frame[0]
        self.goal_[1] = agent.goal_global_frame[1]

        # Get Linear Constraints for Static Collision Avoidance
        self.linear_constraints = self.static_obstacles_manager.get_linear_constraints(agent)

        self.current_state_[0] = agent.pos_global_frame[0]
        self.current_state_[1] = agent.pos_global_frame[1]
        self.current_state_[2] = agent.heading_global_frame
        self.current_state_[3] = np.clip(np.linalg.norm(agent.vel_global_frame),0,1)
        self.current_state_[4] = np.clip(agent.angular_speed_global_frame,-2,2)

        exit_flag, output = self.run_solver(agent,other_agents, enable_constraints = True, update=False)
        if exit_flag != 1:
            exit_flag, output = self.run_solver(agent,other_agents, enable_constraints = True, update=False)
        if exit_flag == 1:
            #return ((np.clip(np.array([output["x"+str(self.FORCES_N)][self.FORCES_NU],output["x"+str(self.FORCES_N)][self.FORCES_NU+1]])-self.current_state_[:2],-2,2)+2)/4), exit_flag
            return (np.array([output["x" + str(self.FORCES_N)][self.FORCES_NU],output["x" + str(self.FORCES_N)][self.FORCES_NU + 1]]) - self.current_state_[:2]), exit_flag
        else:
            #return ((np.clip(np.array([self.goal_[0],self.goal_[1]]) - self.current_state_[:2],-2,2)+2)/4), exit_flag
            return (np.array([self.goal_[0], self.goal_[1]]) - self.current_state_[:2]), exit_flag

    def reset_solver(self):
        self.FORCES_x0[:] = 0.0
        self.FORCES_xinit[:] = 0.0
        self.FORCES_all_parameters[:] = 0.0

    def update_current_state(self, new_state_):
        self.current_state_[0] = new_state_[0]
        self.current_state_[1] = new_state_[1]
        self.current_state_[2] = new_state_[2]
        self.current_state_[3] = new_state_[3]

    def run_solver(self,ego_agent,other_agents, enable_constraints, update):
        # Initial conditions
        for i in range(self.FORCES_NX):
            self.FORCES_xinit[i] = self.current_state_[i]
            self.FORCES_x0[self.FORCES_NU + i] = self.current_state_[i] # x position

        other_agents_ordered = np.zeros(((len(other_agents)), 4))

        for ag_id,other_agent in enumerate(other_agents):
            other_agents_ordered[ag_id, 0] = other_agent.id
            other_agents_ordered[ag_id, 1] = np.linalg.norm(other_agent.pos_global_frame - ego_agent.pos_global_frame)
            other_agents_ordered[ag_id, 2] = other_agent.radius
            other_agents_ordered[ag_id, 3] = other_agent.heading_global_frame

        other_agents_ordered = other_agents_ordered[other_agents_ordered[:, 1].argsort()]

        for N_iter in range(0, self.FORCES_N):
            k = N_iter * self.FORCES_NPAR

            self.FORCES_all_parameters[k + 0] = self.goal_[0]
            self.FORCES_all_parameters[k + 1] = self.goal_[1]
            self.FORCES_all_parameters[k + 2] = self.repulsive_weight_
            self.FORCES_all_parameters[k + 3] = self.x_error_weight_
            self.FORCES_all_parameters[k + 4] = self.y_error_weight_
            self.FORCES_all_parameters[k + 5] = self.angular_acc_weight_
            self.FORCES_all_parameters[k + 6] = self.theta_error_weight_
            self.FORCES_all_parameters[k + 7] = self.acceleration_weigth_
            self.FORCES_all_parameters[k + 8] = self.slack_weight_
            self.FORCES_all_parameters[k + 9] = self.velocity_weight_
            self.FORCES_all_parameters[k + 10] = self.angular_velocity_weight_
            self.FORCES_all_parameters[k + 26] = ego_agent.radius +0.01 # disc radius +0.05
            self.FORCES_all_parameters[k + 27] = 0.0 # disc position

            # Static Collision Avoidance Constraints
            for c in range(self.M):
                c_idx = int(min(c,len(self.linear_constraints)-1))
                self.FORCES_all_parameters[k + 70 + c] = self.linear_constraints[c_idx,0]
                self.FORCES_all_parameters[k + 74 + c] = self.linear_constraints[c_idx,1]
                self.FORCES_all_parameters[k + 78 + c] = self.linear_constraints[c_idx,2]

            # todo: order agents by distance , other agent is hard coded
            if enable_constraints and (len(other_agents)>0):
                print("Collision avoidance ON")
                for obs_id in range(np.minimum(len(other_agents),self.n_obstacles_)):
                    other_ag_id = int(other_agents_ordered[obs_id,0])
                    self.FORCES_all_parameters[k + 28 + obs_id * 7] = self.all_predicted_trajectory[other_ag_id,0,N_iter,0]
                    self.FORCES_all_parameters[k + 29 + obs_id * 7] = self.all_predicted_trajectory[other_ag_id,0,N_iter,1]
                    self.FORCES_all_parameters[k + 30 + obs_id * 7] = other_agents_ordered[obs_id, 3] # orientation
                    self.FORCES_all_parameters[k + 31 + obs_id * 7] = other_agents_ordered[obs_id, 2] # TODO: ADD UNCERTAINTY + 3*self.all_predicted_trajectory[other_ag_id,0,N_iter,2] # major axis
                    self.FORCES_all_parameters[k + 32 + obs_id * 7] = other_agents_ordered[obs_id, 2] # TODO: ADD UNCERTAINTY + 3*self.all_predicted_trajectory[other_ag_id,0,N_iter,3] # minor axis
                    self.FORCES_all_parameters[k + 33 + obs_id * 7] = self.all_predicted_trajectory[other_ag_id,0,N_iter,4] # vx
                    self.FORCES_all_parameters[k + 34 + obs_id * 7] = self.all_predicted_trajectory[other_ag_id,0,N_iter,5] #vy

                for j in range(np.minimum(len(other_agents),self.n_obstacles_),self.n_obstacles_):
                    self.FORCES_all_parameters[k + 28 + j * 7] = self.all_predicted_trajectory[other_ag_id,0,N_iter,0]
                    self.FORCES_all_parameters[k + 29 + j * 7] = self.all_predicted_trajectory[other_ag_id,0,N_iter,1]
                    self.FORCES_all_parameters[k + 30 + j * 7] = other_agents_ordered[obs_id, 3] # orientation
                    self.FORCES_all_parameters[k + 31 + j * 7] = other_agents_ordered[obs_id, 2] # TODO: ADD UNCERTAINTY + self.all_predicted_trajectory[other_ag_id,0,N_iter,2] # major axis# major axis
                    self.FORCES_all_parameters[k + 32 + j * 7] = other_agents_ordered[obs_id, 2] # TODO: ADD UNCERTAINTY + self.all_predicted_trajectory[other_ag_id,0,N_iter,3] # minor axis# minor axis
                    self.FORCES_all_parameters[k + 33 + j * 7] = self.all_predicted_trajectory[other_ag_id,0,N_iter,4]
                    self.FORCES_all_parameters[k + 34 + j * 7] = self.all_predicted_trajectory[other_ag_id,0,N_iter,5]

            else:
                print("Collision avoidance OFF")
                for obs_id in range(0,self.n_obstacles_):
                    self.FORCES_all_parameters[k + 28 + obs_id * 7] = self.current_state_[0] + 30.0
                    self.FORCES_all_parameters[k + 29 + obs_id * 7] = self.current_state_[1] + 30.0
                    self.FORCES_all_parameters[k + 30 + obs_id * 7] = 0.0
                    self.FORCES_all_parameters[k + 31 + obs_id * 7] = 0.4 # major axis
                    self.FORCES_all_parameters[k + 32 + obs_id * 7] = 0.4 # minor axis
                    self.FORCES_all_parameters[k + 33 + obs_id * 7] = 0.0 # vx
                    self.FORCES_all_parameters[k + 34 + obs_id * 7] = 0.0 # vy

        PARAMS = {"x0": self.FORCES_x0, "xinit": self.FORCES_xinit, "all_parameters": self.FORCES_all_parameters}
        OUTPUT, EXITFLAG, INFO = StaticMPCsolver_py.StaticMPCsolver_solve(PARAMS)

        self.solve_time = INFO.solvetime

        if self.debug:
            print(INFO.pobj)

        if EXITFLAG == 1 and update:
            for t in range(self.FORCES_N):
                for i in range(0, self.FORCES_TOTAL_V):
                    if t < 9:
                        self.FORCES_x0[i + t*self.FORCES_TOTAL_V] = OUTPUT["x0"+str(t+1)][i]
                    else:
                        self.FORCES_x0[i + t * self.FORCES_TOTAL_V] = OUTPUT["x" + str(t + 1)][i]

            self.control_cmd_linear = self.FORCES_x0[self.FORCES_TOTAL_V+ self.FORCES_NU + 3]
            self.control_cmd_angular = self.FORCES_x0[self.FORCES_TOTAL_V+ self.FORCES_NU + 4]

            for N_iter in range(0, self.FORCES_N):
                self.predicted_traj[N_iter] = self.FORCES_x0[N_iter*self.FORCES_TOTAL_V+ self.FORCES_NU:N_iter*self.FORCES_TOTAL_V+ self.FORCES_NU+2]
        elif EXITFLAG != 1:
            self.FORCES_x0[self.FORCES_TOTAL_V:] *=0
            for N_iter in range(0, self.FORCES_N):
                self.predicted_traj[N_iter] = self.current_state_[:2]
        # If there is not guidance then MPC is warm-started with previously computed trajectory
        self.guidance_traj = self.predicted_traj

        ego_agent.acc_global_frame = self.FORCES_x0[0]
        ego_agent.ang_acc_global_frame = self.FORCES_x0[1]

        return EXITFLAG, OUTPUT

    def find_next_action(self, obs, agents, i):
        self.network_output_to_action(i,agents,obs)