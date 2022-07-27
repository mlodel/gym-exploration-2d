from gym_collision_avoidance.envs.maps.map_base import BaseMap
from typing import Any, Dict, Optional, Type, Union

import cv2
import numpy as np


class EnvMap(BaseMap):
    def __init__(self, map_size: tuple, cell_size: float, obstacles_vert: list):
        super().__init__(map_size, cell_size)

        self.obstacles = obstacles_vert
        self.update()

    def _init_maps(self):
        super()._init_maps()

    def _update(self, **kwargs):
        self.draw_obstacles()

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
