import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor
from gym_collision_avoidance.envs.maps.map_explore import ExploreMap
from gym_collision_avoidance.envs.config import Config


class ExploreMapSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.map = ExploreMap(
            map_size=Config.MAP_SIZE,
            cell_size=Config.SUBMAP_RESOLUTION,
            sensing_fov=Config.IG_SENSE_FOV,
            sensing_range=Config.IG_SENSE_RADIUS,
            obs_size=Config.EGO_MAP_SIZE,
            submap_lookahead=Config.SUBMAP_LOOKAHEAD,
        )

    def sense(self, agents, agent_index, global_map):
        pose = np.append(
            agents[agent_index].pos_global_frame,
            agents[agent_index].heading_global_frame,
        )
        self.map.update(pose=pose, global_map=global_map)

    def get_obs(
        self,
        obs_type="as_is",
    ):
        return self.map.get_obs(obs_type=obs_type)
