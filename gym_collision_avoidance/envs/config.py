import numpy as np

class Config(object):
    def __init__(self):
        #########################################################################
        # GENERAL PARAMETERS
        self.COLLISION_AVOIDANCE = True
        self.continuous, self.discrete = range(2) # Initialize game types as enum
        self.ACTION_SPACE_TYPE   = self.continuous

        ### DISPLAY
        self.ANIMATE_EPISODES    = False
        self.SHOW_EPISODE_PLOTS = False
        self.SAVE_EPISODE_PLOTS = False
        if not hasattr(self, "PLOT_CIRCLES_ALONG_TRAJ"):
            self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.ANIMATION_PERIOD_STEPS = 5 # plot every n-th DT step (if animate mode on)
        self.PLT_LIMITS = None
        self.PLT_FIG_SIZE = (10, 8)

        self.USE_STATIC_MAP = False
        
        ### TRAIN / PLAY / EVALUATE
        self.TRAIN_MODE           = True # Enable to see the trained agent in action (for testing)
        self.PLAY_MODE           = False # Enable to see the trained agent in action (for testing)
        self.EVALUATE_MODE       = False # Enable to see the trained agent in action (for testing)
        
        ### REWARDS
        self.REWARD_AT_GOAL = 1.0 # reward given when agent reaches goal position
        self.REWARD_COLLISION_WITH_AGENT = -0.25 # reward given when agent collides with another agent
        self.REWARD_COLLISION_WITH_WALL = -0.25 # reward given when agent collides with wall
        self.REWARD_GETTING_CLOSE   = -0.1 # reward when agent gets close to another agent (unused?)
        self.REWARD_ENTERED_NORM_ZONE   = -0.05 # reward when agent enters another agent's social zone
        self.REWARD_TIME_STEP   = 0.0 # default reward given if none of the others apply (encourage speed)
        self.REWARD_WIGGLY_BEHAVIOR = 0.0
        self.WIGGLY_BEHAVIOR_THRESHOLD = np.inf
        self.COLLISION_DIST = 0.0 # meters between agents' boundaries for collision
        self.GETTING_CLOSE_RANGE = 0.2 # meters between agents' boundaries for collision
        # self.SOCIAL_NORMS = "right"
        # self.SOCIAL_NORMS = "left"
        self.SOCIAL_NORMS = "none"

        ### SIMULATION
        self.DT             = 0.2 # seconds between simulation time steps
        self.NEAR_GOAL_THRESHOLD = 0.2
        self.MAX_TIME_RATIO = 2. # agent has this number times the straight-line-time to reach its goal before "timing out"
        
        ### TEST CASE SETTINGS
        self.TEST_CASE_FN = "get_testcase_random"
        self.TEST_CASE_ARGS = {
            'policy_to_ensure': 'learning_ga3c',
            'policies': ['noncoop', 'learning_ga3c', 'static'],
            'policy_distr': [0.05, 0.9, 0.05],
            'speed_bnds': [0.5, 2.0],
            'radius_bnds': [0.2, 0.8],
            'side_length': [
                {'num_agents': [0,5], 'side_length': [4,5]}, 
                {'num_agents': [5,np.inf], 'side_length': [6,8]},
                ],
        }

        if not hasattr(self, "MAX_NUM_AGENTS_IN_ENVIRONMENT"):
            self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 4
        if not hasattr(self, "MAX_NUM_AGENTS_TO_SIM"):
            self.MAX_NUM_AGENTS_TO_SIM = 4
        self.MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        if not hasattr(self, "MAX_NUM_OTHER_AGENTS_OBSERVED"):
            self.MAX_NUM_OTHER_AGENTS_OBSERVED = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1

        ### EXPERIMENTS
        self.NUM_TEST_CASES = 50
        self.PLOT_EVERY_N_EPISODES = 100 # for tensorboard visualization

        ### SENSORS
        self.SENSING_HORIZON  = np.inf
        # self.SENSING_HORIZON  = 3.0
        self.LASERSCAN_LENGTH = 512 # num range readings in one scan
        self.NUM_STEPS_IN_OBS_HISTORY = 1 # number of time steps to store in observation vector
        self.NUM_PAST_ACTIONS_IN_STATE = 0

        ### RVO AGENTS
        self.RVO_TIME_HORIZON = 5.0
        self.RVO_COLLAB_COEFF = 0.5
        self.RVO_ANTI_COLLAB_T = 1.0

        ### OBSERVATION VECTOR
        self.TRAIN_SINGLE_AGENT = False
        self.STATE_INFO_DICT = {
            'dist_to_goal': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("dist_to_goal")',
                'std': np.array([5.], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
                },
            'radius': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([0.5], dtype=np.float32)
                },
            'heading_ego_frame': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.pi, np.pi],
                'attr': 'get_agent_data("heading_ego_frame")',
                'std': np.array([3.14], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
                },
            'pref_speed': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("pref_speed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
                },
            'num_other_agents': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("num_other_agents_observed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
                },
            'other_agent_states': {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)
                },
            'other_agents_states': {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED,7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32), (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32), (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                },
            'laserscan': {
                'dtype': np.float32,
                'size': self.LASERSCAN_LENGTH,
                'bounds': [0., 6.],
                'attr': 'get_sensor_data("laserscan")',
                'std': 5.*np.ones((self.LASERSCAN_LENGTH), dtype=np.float32),
                'mean': 5.*np.ones((self.LASERSCAN_LENGTH), dtype=np.float32)
                },
            'is_learning': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0., 1.],
                'attr': 'get_agent_data_equiv("policy.str", "learning")'
                }
            }
        self.setup_obs()
    
        self.AGENT_SORTING_METHOD = "closest_last"
        # self.AGENT_SORTING_METHOD = "closest_first"
        # self.AGENT_SORTING_METHOD = "time_to_impact"

    def setup_obs(self):
        if not hasattr(self, "STATES_IN_OBS"):
            self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'other_agents_states']
            # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agent_states', 'use_ppo', 'laserscan']
            # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agent_states', 'use_ppo'] # 2-agent net
            # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agents_states', 'use_ppo', 'num_other_agents', 'laserscan'] # LSTM
        if not hasattr(self, "STATES_NOT_USED_IN_POLICY"):
            self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        self.MEAN_OBS = {}; self.STD_OBS = {}
        for state in self.STATES_IN_OBS:
            if 'mean' in self.STATE_INFO_DICT[state]:
                self.MEAN_OBS[state] = self.STATE_INFO_DICT[state]['mean']
            if 'std' in self.STATE_INFO_DICT[state]:
                self.STD_OBS[state] = self.STATE_INFO_DICT[state]['std']

class EvaluateConfig(Config):
    def __init__(self):
        Config.__init__(self)
        self.EVALUATE_MODE = True
        self.TRAIN_MODE = False
        self.DT = 0.1
        self.MAX_TIME_RATIO = 8.
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 19

class Example(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.TRAIN_MODE = False
        self.EVALUATE_MODE = True

class Formations(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = True
        self.NEAR_GOAL_THRESHOLD = 0.2
        self.PLT_LIMITS = [[-5, 6], [-2, 7]]
        self.PLT_FIG_SIZE = (10,10)
        self.PLOT_CIRCLES_ALONG_TRAJ = False

class SmallTestSuite(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = False
        self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.NUM_TEST_CASES = 4

class FullTestSuite(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.NEAR_GOAL_THRESHOLD = 0.8
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = False
        self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.NUM_TEST_CASES = 4
