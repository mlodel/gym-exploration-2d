from gym_collision_avoidance.envs.maps.map_base import BaseMap
from gym_collision_avoidance.envs.maps.map_env import EnvMap

from typing import Any, Dict, Optional, Type, Union

import cv2
import numpy as np


class ExploreMap(BaseMap):
    def __init__(
        self,
        map_size: tuple,
        cell_size: float,
        sensing_range: float,
        sensing_fov: float = 2 * np.pi,
        obs_size=(80, 80),
        submap_size=(40, 40),
    ):
        super().__init__(map_size, cell_size, obs_size, submap_size)
        self.map_unknown_color = 127
        self.map += self.map_unknown_color
        self.map_scale = 255
        self.sens_range = np.around(sensing_range / cell_size).astype(int)
        self.sens_fov = sensing_fov

    def update(self, pose: np.ndarray, global_map: EnvMap = None):
        if global_map is None:
            raise ValueError("No global map passed for map update")

        super().update(pose)

        cell_pos = self.get_idc_from_pos(pose)

        # Create mask and inverse mask for visible area in circle around robot
        mask = np.zeros_like(global_map.map)
        mask = cv2.circle(
            mask, center=cell_pos[::-1], radius=self.sens_range, color=255, thickness=-1
        )
        # TODO limited FOV
        #  cv2.ellipse(mask, cell_pos[::-1], (self.sens_range, self.sens_range), pose[2], -self.sens_fov/2,
        #  self.sens_fov/2, 255, -1)
        mask_inv = cv2.bitwise_not(mask)

        # Extract contours and filled obstacles in visible area (mask circle)
        update_contours = cv2.bitwise_and(
            global_map.map_contours, global_map.map_contours, mask=mask
        )
        update_filled = cv2.bitwise_and(global_map.map, global_map.map, mask=mask)

        # Scale update maps
        update_contours *= 255 // global_map.map_scale
        update_filled *= 255 // global_map.map_scale

        # Forget prior in sensing area, overwrite with new measurement
        self.map = cv2.bitwise_and(self.map, self.map, mask=mask_inv)

        # Color unknown areas in visible area (mask) grey
        update_unknown = np.zeros_like(mask)
        update_unknown[update_filled == 255] = self.map_unknown_color
        # TODO

        # Combine obstacle contour with unknown areas in visible area
        update = cv2.bitwise_or(update_contours, update_unknown, mask=mask)

        # Add update to map prior (where visible area is 0)
        self.map = cv2.add(self.map, update)

        # # Fill closed obstacles
        # contours, hierarchy = cv2.findContours(
        #     self.map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        # )
        # # draw all contours to image (green if opened else red)
        # for i in range(len(contours)):
        #     opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
        #     if not opened:
        #         cv2.drawContours(self.map, contours, i, color=2, thickness=-1)

    def _map_obs_postprocessor(self, map_array):
        # Invert to make free areas white, occupied black
        return cv2.bitwise_not(map_array)
