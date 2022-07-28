from gym_collision_avoidance.envs.maps.map_base import BaseMap
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
    ):
        super().__init__(map_size, cell_size)

        self.sens_range = sensing_range
        self.sens_fov = sensing_fov

    def _init_maps(self):
        super()._init_maps()

    def update(self, pose: np.ndarray, global_map: BaseMap = None):
        if global_map is None:
            raise ValueError("No global map passed for map update")
        cell_pos = self.get_idc_from_pos(pose)
        mask = np.zeros_like(global_map.map)
        mask = cv2.circle(mask, center=cell_pos, radius=15, color=1, thickness=-1)
        mask_inv = cv2.bitwise_not(mask)

        update = cv2.bitwise_and(global_map.map, global_map.map, mask=mask)
        # Method A: Forget prior in sensing area, overwrite with new measurement
        self.map = cv2.bitwise_and(self.map, self.map, mask=mask_inv)
        self.map = cv2.add(self.map, update)
        # Method B: OR between prior and update
        # img2 = cv2.bitwise_or(img2, update)
