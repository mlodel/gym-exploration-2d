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
        submap_lookahead: float = None,
        obstacles_vert: list = None,
        json: str = None,
    ):
        super().__init__(map_size, cell_size, obs_size, submap_lookahead)

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

        self.map_size = (shape[1] * self.cell_size, shape[0] * self.cell_size)
        self.map.resize(shape, refcheck=False)

        # ratio = self.map.shape[0] / max(shape)
        # verts = (verts * ratio).astype(int)

        verts[:, 0] = verts[:, 0] - x_min + border_pad
        verts[:, 1] = verts[:, 1] - y_min + border_pad
        cv2.drawContours(self.map, [verts], 0, 255, -1)
        self.map = cv2.bitwise_not(self.map) // 255

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

        self.map_contours = np.zeros_like(self.map)

        # Draw contour map
        contours, hierarchy = cv2.findContours(
            self.map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        for i in range(len(contours)):
            self.map_contours = cv2.drawContours(
                self.map_contours, contours, i, color=1, thickness=1
            )

        # Dilate the contour map
        kernel = np.ones((3, 3), np.uint8)
        self.map_contours = cv2.dilate(self.map_contours, kernel, iterations=1)

    def check_collision(self, pose: np.ndarray = None, radius: float = 0.0) -> bool:
        if pose is None:
            pose = self.pose

        i, j = self.get_idc_from_pos(pose)

        if self.edf_map[i, j] <= radius:
            return True
        else:
            return False

    def get_local_pointcloud(self, pos: np.ndarray, lookahead: float, pt_type="pos"):
        lookahead_px = int(lookahead / self.cell_size)

        # Create border around map to select submap close to map boundaries
        map_border = cv2.copyMakeBorder(
            self.map,
            lookahead_px,
            lookahead_px,
            lookahead_px,
            lookahead_px,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

        pos_px = self.get_idc_from_pos(pos)

        submap = map_border[
            pos_px[0] : pos_px[0] + 2 * lookahead_px,
            pos_px[1] : pos_px[1] + 2 * lookahead_px,
        ]

        # if pt_type == "pos":
        #     submap[:, [0, -1]] = 1
        #     submap[[0, -1], :] = 1
        # elif pt_type == "idc":
        #     points_free = np.argwhere(submap == 0)
        #     points_free = points_free + (
        #             np.array(pos_px) - np.array([submap.shape[0] / 2, submap.shape[1] / 2])
        #     )

        points_idc = np.argwhere(submap)
        points_idc = points_idc + (
            np.array(pos_px) - np.array([submap.shape[0] / 2, submap.shape[1] / 2])
        )
        points = points_idc[:, [1, 0]]
        # points = (
        #     points * np.array([1, -1]) * self.cell_size
        #     + np.array(self.map_size) * np.array([-1, 1]) / 2
        # )
        # points = np.zeros_like(points, dtype=float)
        points[:, 0] = points[:, 0] * self.cell_size - self.map_size[0] / 2
        points[:, 1] = -points[:, 1] * self.cell_size + self.map_size[1] / 2

        # x = (idc[1]) * self.cell_size - self.map_size[0] / 2  # + self.cell_size / 2
        # y = (-idc[0]) * self.cell_size + self.map_size[1] / 2  # - self.cell_size / 2

        if pt_type == "pos":
            return points, submap
        elif pt_type == "idc":
            return points_idc
        else:
            raise ValueError("pt_type must be 'pos' or 'idc'")
