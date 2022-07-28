from gym_collision_avoidance.envs.maps.map_base import BaseMap
from typing import Any, Dict, Optional, Type, Union

import cv2
import numpy as np
from scipy import ndimage


class EnvMap(BaseMap):
    def __init__(
        self,
        map_size: tuple,
        cell_size: float,
        obs_size: tuple,
        submap_size=None,
        obstacles_vert: list = None,
    ):
        super().__init__(map_size, cell_size, obs_size, submap_size)

        self.obstacles = obstacles_vert if obstacles_vert is not None else []
        self.draw_obstacles()
        self.edf_map = (
            ndimage.distance_transform_edt((~(self.map.astype(bool))).astype(int))
            * self.cell_size
        )

    def _init_maps(self):
        super()._init_maps()

    def update(self, pose: np.ndarray, **kwargs):
        super().update(pose)

    def draw_obstacles(self):
        for obst in self.obstacles:
            vert_idc = []
            for vert in obst:
                vert_idc.append(self.get_idc_from_pos(vert)[::-1])

            # self.map = cv2.polylines(
            #     self.map, vert_idc, isClosed=True, color=1, thickness=-1
            # )
            self.map = cv2.fillConvexPoly(
                img=self.map, points=np.array(vert_idc), color=1
            )

        self.map[0, :] = 1
        self.map[-1, :] = 1
        self.map[:, 0] = 1
        self.map[:, -1] = 1

    def check_collision(self, pose: np.ndarray = None, radius: float = 0.0) -> bool:
        if pose is None:
            pose = self.pose

        i, j = self.get_idc_from_pos(pose)

        if self.edf_map[i, j] <= radius:
            return True
        else:
            return False
