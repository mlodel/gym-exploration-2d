import numpy as np
import cv2
import os

from typing import Any, Dict, Optional, Type, Union


class Map:
    def __init__(self, map_size: tuple, cell_size: float, obstacles: dict = None):
        self.map_size = map_size
        self.cell_size = cell_size
        self.shape = (
            self.map_size[0] / self.cell_size,
            self.map_size[1] / self.cell_size,
        )

        self.map = np.zeros(self.shape)

        self.obstacles = obstacles

    def get_idc_from_pos(self, pos: Union[np.ndarray, list, tuple]) -> tuple:
        # OpenCV coordinate frame is in the top-left corner, x to the left, y downwards
        x_idx = np.floor((pos[0] + self.map_size[0] / 2) / self.cell_size)
        y_idx = np.floor((-pos[1] + self.map_size[1] / 2) / self.cell_size)

        x_idx = np.clip(x_idx, 0, self.map.shape[1] - 1)
        y_idx = np.clip(y_idx, 0, self.map.shape[0] - 1)
        return y_idx.astype(int), x_idx.astype(int)

    def get_pos_from_idc(self, idc: tuple) -> np.ndarray:
        x = (idc[1]) * self.cell_size - self.map_size[0] / 2  # + self.cell_size / 2
        y = (-idc[0]) * self.cell_size + self.map_size[1] / 2  # - self.cell_size / 2
        return np.array([x, y])

    def draw_obstacles(self):
        for obst in self.obstacles:
            vert_idc = []
            for vert in obst["vertices"]:
                vert_idc.append(self.get_idc_from_pos())
