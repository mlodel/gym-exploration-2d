import numpy as np


class Config:
    #########################################################################
    # GENERAL PARAMETERS
    COLLISION_AVOIDANCE = True
    continuous, discrete = range(2)  # Initialize game types as enum
    ACTION_SPACE_TYPE = discrete

    ANIMATE_EPISODES = True
    SHOW_EPISODE_PLOTS = False
    SAVE_EPISODE_PLOTS = True
    TRAIN_SINGLE_AGENT = True

    #########################################################################
    # COLLISION AVOIDANCE PARAMETER
    DT = 0.1  # seconds between simulation time steps
    REWARD_AT_GOAL = 0.0  # reward given when agent reaches goal position
    REWARD_COLLISION_WITH_AGENT = (
        0.0  # reward given when agent collides with another agent
    )
    REWARD_TIMEOUT = 0.0  # reward given for not reaching the goal
    REWARD_INFEASIBLE = 0.0
    REWARD_COLLISION_WITH_WALL = -0.0  # reward given when agent collides with wall
    REWARD_GETTING_CLOSE = (
        0.0  # reward when agent gets close to another agent (unused?)
    )
    REWARD_TIME_STEP = (
        -0.1
    )  # -0.1  # default reward given if none of the others apply (encourage speed)
    REWARD_DISTANCE_TO_GOAL = (
        0.0  # default reward given if none of the others apply (encourage speed)
    )
    REWARD_WIGGLY_BEHAVIOR = 0.0

    REWARD_DEADLOCKED = -0.0
    REWARD_SUBGOAL_INFEASIBLE = -0.0
    REWARD_FACTOR_DISTANCE = 0.0  # -0.1
    REWARD_COVERAGE = 0.0

    WIGGLY_BEHAVIOR_THRESHOLD = 0.0
    ENABLE_COLLISION_AVOIDANCE = True
    COLLISION_DIST = 0.5  # meters between agents' boundaries for collision
    GETTING_CLOSE_RANGE = 0.2  # meters between agents' boundaries for collision
    JOINT_MPC_RL_TRAINING = False  # select the action that has highets reward (mpc/rl)

    COLLISION_AV_W_STATIC_AGENT = False

    # MPC
    FORCES_N = 15
    FORCES_DT = 0.3
    REPEAT_STEPS = 5

    LASERSCAN_LENGTH = 16  # num range readings in one scan

    NEAR_GOAL_THRESHOLD = 0.25
    MAX_TIME_RATIO = 6.4  # 6.4  # agent has this number times the straight-line-time to reach its goal before "timing out"

    SENSING_HORIZON = np.inf

    MAX_NUM_AGENTS_IN_ENVIRONMENT = 10
    MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT = MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
    MAX_NUM_OTHER_AGENTS_OBSERVED = MAX_NUM_AGENTS_IN_ENVIRONMENT - 1

    # Gridmap parameters
    SUBMAP_WIDTH = 40  # Pixels
    SUBMAP_HEIGHT = 40  # Pixels
    SUBMAP_RESOLUTION = 0.1  # Pixel / meter
    SUBMAP_SCALE = True
    SUBMAP_SCALE_TARGET = (80, 80)

    # STATIC MAP
    MAP_WIDTH = 20  # Meters
    MAP_HEIGHT = 20  # Meters
    MAP_WIDTH_PXL = 20
    MAP_HEIGHT_PXL = 20

    EGO_MAP_SIZE = (84, 84)

    IG_MAP_RESOLUTION = 1.0
    IG_EDF_RESOLUTION_FACTOR = 10
    IG_EXPERT_POLICY = "IG_EXPERT_POLICY"
    IG_SENSE_RADIUS = 3.5
    IG_SENSE_FOV = 360.0
    IG_SENSE_rOcc = 3.0
    IG_SENSE_rEmp = 0.33
    IG_ACCUMULATE_REWARDS = False
    IG_REWARD_MODE = "binary"  # entropy, binary
    IG_REWARD_BINARY_CELL = 0.1
    IG_THRES_VISITED_CELLS = 0.9
    IG_THRES_AVG_CELL_ENTROPY = 0.1  # 0.1
    IG_THRES_ACTIVE = True  # When False fixed episode length by timeout

    IG_REWARD_GOAL_CELL_FACTOR = 0.0
    IG_REWARD_GOAL_PENALTY = -0.0
    IG_REWARD_GOAL_COMPLETION = 0.0

    IG_GOALS_ACTIVE = False
    IG_GOALS_SETTINGS = {"max_steps": 128}

    REWARD_MAX_IG = (
        1.2
        + 13 * IG_REWARD_GOAL_CELL_FACTOR * IG_REWARD_BINARY_CELL
        + IG_REWARD_GOAL_COMPLETION
        if IG_REWARD_MODE == "binary"
        else 4.0
    )  # 6.7 4.0 # 0.2 ## binary 1.2 entropy 4.0 (w/o accumulating)

    REWARD_MIN_IG = IG_REWARD_GOAL_PENALTY

    # IG_CURRICULUM_LEARNING = True
    # IG_CURRICULUM_LEARNING_STEPS_2_OBS = 2000000
    # IG_CURRICULUM_LEARNING_STEPS_3_OBS = 4000000

    PLOT_EVERY_N_STEPS = 200000  # for visualization

    REWARDS_NORMALIZE = True

    TEST_MODE = False
    TEST_N_OBST = 3

    DISCRETE_SUBGOAL_ANGLES = 12
    DISCRETE_SUBGOAL_RADII = [4.0]

    SUBGOALS_EGOCENTRIC = True
    CLIP_ACTION = True
    USE_MPC_EXPERT_IN_TEST = False

    SCENARIOS_FOR_TRAINING = ["IG_single_agent_crossing"]

    # Angular Map
    NUM_OF_SLICES = 16
    MAX_RANGE = 6

    PLOT_CIRCLES_ALONG_TRAJ = False
    ANIMATION_PERIOD_STEPS = 10  # plot every n-th DT step (if animate mode on)
    PLT_LIMITS = ((-MAP_WIDTH / 2, MAP_WIDTH / 2), (-MAP_HEIGHT / 2, MAP_HEIGHT / 2))
    PLT_FIG_SIZE = (12, 8)
    PLT_SHOW_LEGEND = False
    PLT_SUBPLT_TRAJ = False
    PLT_SUBPLT_TARGMAP = True
    PLT_FREE_SPACE = True

    STATES_IN_OBS = [
        "heading_global_frame",
        "angvel_global_frame",
        "pos_global_frame",
        "vel_global_frame",
        "ego_binary_map",
        "ego_explored_map",
        # "local_grid",
    ]

    STATES_NOT_USED_IN_POLICY = ["use_ppo", "num_other_agents"]
    STATE_INFO_DICT = {
        "heading_ego_frame": {
            "dtype": np.float64,
            "size": 1,
            "bounds": [-np.pi, np.pi],
            "agent_attr": "heading_ego_frame",
            "std": np.array([3.14], dtype=np.float64),
            "mean": np.array([0.0], dtype=np.float64),
        },
        "heading_global_frame": {
            "dtype": np.float64,
            "size": 1,
            "bounds": [-np.pi, np.pi],
            "agent_attr": "heading_global_frame",
            "std": np.array([3.14], dtype=np.float64),
            "mean": np.array([0.0], dtype=np.float64),
        },
        "pos_global_frame": {
            "dtype": np.float64,
            "size": 2,
            "bounds": [-np.inf, np.inf],
            "agent_attr": "pos_global_frame",
            "std": np.array([1.0], dtype=np.float64),
            "mean": np.array([0.0], dtype=np.float64),
        },
        "vel_global_frame": {
            "dtype": np.float64,
            "size": 2,
            "bounds": [-np.inf, np.inf],
            "agent_attr": "vel_global_frame",
            "std": np.array([1.0], dtype=np.float64),
            "mean": np.array([0.0], dtype=np.float64),
        },
        "angvel_global_frame": {
            "dtype": np.float64,
            "size": 1,
            "bounds": [-np.inf, np.inf],
            "agent_attr": "angular_speed_global_frame",
            "std": np.array([1.0], dtype=np.float64),
            "mean": np.array([0.0], dtype=np.float64),
        },
        "local_grid": {
            "dtype": np.uint8,
            "size": EGO_MAP_SIZE,
            "bounds": [0, 255],
            "sensor_name": "GlobalMapSensor",
            "sensor_kwargs": dict(obs_type="ego_submap"),
            "std": np.ones((SUBMAP_WIDTH, SUBMAP_HEIGHT), dtype=np.uint8),
            "mean": np.ones((SUBMAP_WIDTH, SUBMAP_HEIGHT), dtype=np.uint8),
        },
        "explored_map": {
            "dtype": np.uint8,
            "size": (
                int(MAP_HEIGHT / SUBMAP_RESOLUTION),
                int(MAP_WIDTH / SUBMAP_RESOLUTION),
            ),
            "bounds": [0, 255],
            "sensor_name": "ExploreMapSensor",
            "sensor_kwargs": dict(obs_type="as_is"),
            "std": np.ones(
                (
                    int(MAP_HEIGHT / SUBMAP_RESOLUTION),
                    int(MAP_WIDTH / SUBMAP_RESOLUTION),
                ),
                dtype=np.uint8,
            ),
            "mean": np.ones(
                (
                    int(MAP_HEIGHT / SUBMAP_RESOLUTION),
                    int(MAP_WIDTH / SUBMAP_RESOLUTION),
                ),
                dtype=np.uint8,
            ),
        },
        "ego_explored_map": {
            "dtype": np.uint8,
            "size": EGO_MAP_SIZE,
            "bounds": [0, 255],
            "sensor_name": "ExploreMapSensor",
            "sensor_kwargs": dict(obs_type="ego_global_map"),
            "std": np.ones(EGO_MAP_SIZE, dtype=np.uint8),
            "mean": np.ones(EGO_MAP_SIZE, dtype=np.uint8),
        },
        "ego_global_map": {
            "dtype": np.uint8,
            "size": EGO_MAP_SIZE,
            "bounds": [0, 255],
            "sensor_name": "EnvMapSensor",
            "sensor_kwargs": dict(obs_type="ego_global_map"),
            "std": np.ones(EGO_MAP_SIZE, dtype=np.uint8),
            "mean": np.ones(EGO_MAP_SIZE, dtype=np.uint8),
        },
        "target_map": {
            "dtype": np.uint8,
            "size": (MAP_WIDTH_PXL, MAP_HEIGHT_PXL),
            "bounds": [-np.inf, np.inf],
            "agent_attr": "ig_model.targetMap.probMap",
            "std": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
            "mean": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
        },
        "entropy_map": {
            "dtype": np.uint8,
            "size": (MAP_WIDTH_PXL, MAP_HEIGHT_PXL),
            "bounds": [-np.inf, np.inf],
            "agent_attr": "ig_model.targetMap.entropyMap",
            "std": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
            "mean": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
        },
        "agent_pos_map": {
            "dtype": np.uint8,
            "size": (MAP_WIDTH_PXL, MAP_HEIGHT_PXL),
            "bounds": [-np.inf, np.inf],
            "agent_attr": "ig_model.agent_pos_map",
            "std": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
            "mean": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
        },
        "binary_map": {
            "dtype": np.uint8,
            "size": (MAP_WIDTH_PXL, MAP_HEIGHT_PXL),
            "bounds": [-np.inf, np.inf],
            "agent_attr": "ig_model.targetMap.binaryMap.astype(float)",
            "std": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
            "mean": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
        },
        "ego_entropy_map": {
            "dtype": np.uint8,
            "size": EGO_MAP_SIZE,
            "bounds": [-np.inf, np.inf],
            "agent_attr": "ig_model.targetMap.ego_map",
            "std": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
            "mean": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
        },
        "ego_binary_map": {
            "dtype": np.uint8,
            "size": EGO_MAP_SIZE,
            "bounds": [0, 255],
            "agent_attr": "ig_model.targetMap.bin_ego_map",
            "std": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
            "mean": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
        },
        "mc_ego_binary_goal": {
            "dtype": np.uint8,
            "size": (2, EGO_MAP_SIZE[0], EGO_MAP_SIZE[1]),
            "bounds": [0, 255],
            "agent_attr": "ig_model.targetMap.mc_ego_binary_goal",
            "std": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
            "mean": np.ones((MAP_WIDTH_PXL, MAP_HEIGHT_PXL), dtype=np.uint8),
        },
        # "dist_to_goal": {
        #     "dtype": np.float64,
        #     "size": 1,
        #     "bounds": [-np.inf, np.inf],
        #     "attr": '"dist_to_goal")',
        #     "std": np.array([5.0], dtype=np.float64),
        #     "mean": np.array([0.0], dtype=np.float64),
        # },
        # "radius": {
        #     "dtype": np.float64,
        #     "size": 1,
        #     "bounds": [0, np.inf],
        #     "attr": 'get_agent_data("radius")',
        #     "std": np.array([1.0], dtype=np.float64),
        #     "mean": np.array([0.5], dtype=np.float64),
        # },
        # "rel_goal": {
        #     "dtype": np.float64,
        #     "size": 2,
        #     "bounds": [-np.inf, np.inf],
        #     "attr": 'get_agent_data("rel_goal")',
        #     "std": np.array([10.0], dtype=np.float64),
        #     "mean": np.array([0.0], dtype=np.float64),
        # },
        # "pref_speed": {
        #     "dtype": np.float64,
        #     "size": 1,
        #     "bounds": [0, np.inf],
        #     "attr": 'get_agent_data("pref_speed")',
        #     "std": np.array([1.0], dtype=np.float64),
        #     "mean": np.array([1.0], dtype=np.float64),
        # },
        # "num_other_agents": {
        #     "dtype": np.float64,
        #     "size": 1,
        #     "bounds": [0, np.inf],
        #     "attr": 'get_agent_data("num_other_agents_observed")',
        #     "std": np.array([1.0], dtype=np.float64),
        #     "mean": np.array([1.0], dtype=np.float64),
        # },
        # "other_agent_states": {
        #     "dtype": np.float64,
        #     "size": 9,
        #     "bounds": [-np.inf, np.inf],
        #     "attr": 'get_agent_data("other_agent_states")',
        #     "std": np.array(
        #         [5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float64
        #     ),
        #     "mean": np.array(
        #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float64
        #     ),
        # },
        # "other_agents_states": {
        #     "dtype": np.float64,
        #     "size": (MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT, 10),
        #     "bounds": [-np.inf, np.inf],
        #     "attr": 'get_sensor_data("other_agents_states")',
        #     "std": np.tile(
        #         np.array(
        #             [5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0, 5.0, 1.0], dtype=np.float64
        #         ),
        #         (MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT, 1),
        #     ),
        #     "mean": np.tile(
        #         np.array(
        #             [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0], dtype=np.float64
        #         ),
        #         (MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT, 1),
        #     ),
        # },
        # "angular_map": {
        #     "dtype": np.float64,
        #     "size": NUM_OF_SLICES,
        #     "bounds": [0.0, 6.0],
        #     "attr": 'get_sensor_data("angular_map")',
        #     "std": np.ones(NUM_OF_SLICES, dtype=np.float64),
        #     "mean": np.ones(NUM_OF_SLICES, dtype=np.float64),
        # },
        # "laserscan": {
        #     "dtype": np.float64,
        #     "size": LASERSCAN_LENGTH,
        #     "bounds": [0.0, 6.0],
        #     "attr": 'get_sensor_data("laserscan")',
        #     "std": 5.0 * np.ones((LASERSCAN_LENGTH), dtype=np.float64),
        #     "mean": 5.0 * np.ones((LASERSCAN_LENGTH), dtype=np.float64),
        # },
        # "use_ppo": {
        #     "dtype": np.float64,
        #     "size": 1,
        #     "bounds": [0.0, 1.0],
        #     "attr": 'get_agent_data_equiv("policy.str", "learning")',
        # },
    }
    MEAN_OBS = {}
    STD_OBS = {}
    for state in STATES_IN_OBS:
        if "mean" in STATE_INFO_DICT[state]:
            MEAN_OBS[state] = STATE_INFO_DICT[state]["mean"]
        if "std" in STATE_INFO_DICT[state]:
            STD_OBS[state] = STATE_INFO_DICT[state]["std"]
