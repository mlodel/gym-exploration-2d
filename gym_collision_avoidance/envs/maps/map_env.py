from gym_collision_avoidance.envs.maps.map_base import BaseMap
from typing import Any, Dict, Optional, Type, Union

import cv2
import numpy as np
from scipy import ndimage
import json


class EnvMap(BaseMap):
    def __init__(
        self,
        map_size: tuple,
        cell_size: float,
        obs_size: tuple,
        submap_size=None,
        obstacles_vert: list = None,
        json: str = None,
    ):
        super().__init__(map_size, cell_size, obs_size, submap_size)

        self.map_contours = np.zeros_like(self.map)

        if obstacles_vert is not None and json is None:
            self.draw_from_vert_list(obstacles_vert)
        elif json is not None and obstacles_vert is None:
            self.draw_from_json(json)
        else:
            raise ValueError(
                "Either none or both of obstacles_vert and json_file are given. Define only ONE of both!"
            )
        self.edf_map = (
            ndimage.distance_transform_edt((~(self.map.astype(bool))).astype(int))
            * self.cell_size
        )

    def update(self, pose: np.ndarray, **kwargs):
        super().update(pose)

    def draw_from_vert_list(self, obstacles):
        for obst in obstacles:
            vert_idc = []
            for vert in obst:
                vert_idc.append(self.get_idc_from_pos(vert)[::-1])

            self.map = cv2.fillConvexPoly(
                img=self.map, points=np.array(vert_idc), color=1
            )

        # Draw contour map
        self.draw_contour_map()

        self.map[0, :] = 1
        self.map[-1, :] = 1
        self.map[:, 0] = 1
        self.map[:, -1] = 1

    def draw_from_json(self, json_path):
        with open(json_path.split(".")[0] + ".json") as json_file:
            json_data = json.load(json_file)

        border_pad = 1

        # Draw the contour
        verts = (np.array(json_data["verts"]) / self.cell_size).astype(np.int)
        x_max, x_min, y_max, y_min = (
            np.max(verts[:, 0]),
            np.min(verts[:, 0]),
            np.max(verts[:, 1]),
            np.min(verts[:, 1]),
        )
        shape = (y_max - y_min + border_pad * 2, x_max - x_min + border_pad * 2)
        # map = np.zeros(
        #     shape,
        #     dtype=np.uint8,
        # )

        ratio = self.map.shape[0] / max(shape)
        verts = (verts * ratio).astype(int)

        verts[:, 0] = verts[:, 0] - x_min + border_pad
        verts[:, 1] = verts[:, 1] - y_min + border_pad
        cv2.drawContours(self.map, [verts], 0, 1, 1)

        # shape = cnt_map.shape
        # target_size = shape[1] // 10, shape[0] // 10
        #
        # scaled_map = cv2.resize(
        #     cnt_map, dsize=target_size, interpolation=cv2.INTER_NEAREST
        # )

        # self.map[0 : shape[0], 0 : shape[1]] = map

        # Draw contour map
        self.draw_contour_map()

        self.map[0, :] = 1
        self.map[-1, :] = 1
        self.map[:, 0] = 1
        self.map[:, -1] = 1

    def draw_contour_map(self):
        # Draw contour map
        contours, hierarchy = cv2.findContours(
            self.map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        for i in range(len(contours)):
            self.map_contours = cv2.drawContours(
                self.map_contours, contours, i, color=1, thickness=5
            )

    def check_collision(self, pose: np.ndarray = None, radius: float = 0.0) -> bool:
        if pose is None:
            pose = self.pose

        i, j = self.get_idc_from_pos(pose)

        if self.edf_map[i, j] <= radius:
            return True
        else:
            return False
