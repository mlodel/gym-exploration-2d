import numpy as np
import cv2
import os

from typing import Any, Dict, Optional, Type, Union

from abc import ABC, abstractmethod

from gym_collision_avoidance.envs.maps.map_obs_util import (
    ego_global_map,
    ego_submap_from_map,
)


class BaseMap(ABC):
    def __init__(
        self, map_size: tuple, cell_size: float, obs_size: tuple, submap_size=None
    ):
        self.map_size = map_size
        self.cell_size = cell_size
        self.cell_size_decimals = int(-np.floor(np.log10(self.cell_size))) + 1
        self.shape = (
            int(self.map_size[0] / self.cell_size),
            int(self.map_size[1] / self.cell_size),
        )
        self.map = np.zeros(self.shape, dtype=np.uint8)
        self.pose = np.zeros(3)
        self.obs_size = obs_size
        self.submap_size = submap_size

        # map_scale is maximum value of cells in map (important for scaling to grayscale)
        self.map_scale = 1

        # Optional setting for border_value for image rotations when generating obs, if None map_scale is used
        self.obs_border_value = None

    def get_idc_from_pos(self, pos: Union[np.ndarray, list, tuple]) -> tuple:

        # Round to correct for numerical errors before doing np.floor
        pos = pos.round(self.cell_size_decimals)

        # OpenCV coordinate frame is in the top-left corner, x to the left, y downwards
        x_idx = np.floor((pos[0] + self.map_size[0] / 2) / self.cell_size)
        y_idx = np.floor((-pos[1] + self.map_size[1] / 2) / self.cell_size)

        # x_idx = np.clip(x_idx, 0, self.map.shape[1] - 1)
        x_idx = (
            self.map.shape[1] - 1
            if x_idx > self.map.shape[1] - 1
            else (0 if x_idx < 0 else x_idx)
        )
        # y_idx = np.clip(y_idx, 0, self.map.shape[0] - 1)
        y_idx = (
            self.map.shape[0] - 1
            if y_idx > self.map.shape[1] - 1
            else (0 if y_idx < 0 else y_idx)
        )

        return int(y_idx), int(x_idx)

    def get_pos_from_idc(self, idc: tuple) -> np.ndarray:
        x = (idc[1]) * self.cell_size - self.map_size[0] / 2  # + self.cell_size / 2
        y = (-idc[0]) * self.cell_size + self.map_size[1] / 2  # - self.cell_size / 2
        return np.array([x, y])

    def get_map_value(
        self, pos: Union[np.ndarray, list, tuple], map_name: str = None
    ) -> Union[int, bool, float]:
        i, j = self.get_idc_from_pos(pos)
        if map_name is not None:
            if hasattr(self, map_name):
                map_array = self.__getattribute__(map_name)
                if not isinstance(map_array, np.ndarray):
                    raise ValueError("Requested Map is not an Numpy Array")
            else:
                raise ValueError("Requested Map Name does not exist")
        else:
            map_array = self.map

        return map_array[i, j]

    @abstractmethod
    def update(self, pose: np.ndarray, **kwargs):
        self.pose = pose

    def _map_obs_postprocessor(self, map_array):
        return map_array

    def get_obs(self, map_name: str = None, obs_type: str = "as_is"):
        if map_name is not None:
            if (
                hasattr(self, map_name)
                and isinstance(self.__getattribute__(map_name), np.ndarray)
                and len(self.__getattribute__(map_name).shape) == 2
            ):
                map_array = self.__getattribute__(map_name)
            else:
                raise ValueError(
                    "Requested Map Name does not exist is not a 2D numpy array!"
                )
        else:
            map_array = self.map

        map_cell = self.get_idc_from_pos(self.pose[:2])
        angle = self.pose[2] * 180 / np.pi

        if obs_type == "as_is":
            obs = map_array
        elif obs_type == "ego_global_map":
            submap_width = int(
                np.sqrt(map_array.shape[0] ** 2 + map_array.shape[1] ** 2)
            )
            obs = ego_global_map(
                map=map_array,
                map_cell=map_cell,
                angle=angle,
                sub_img_width=submap_width,
                output_size=self.obs_size,
                border_value=(
                    self.obs_border_value
                    if self.obs_border_value is not None
                    else self.map_scale
                ),
            )
        elif obs_type == "ego_submap" and self.submap_size is not None:
            obs = ego_submap_from_map(
                map=map_array,
                pos_pxl=list(map_cell[::-1]),
                angle_deg=angle,
                submap_size=list(self.submap_size),
                scale_size=list(self.obs_size),
                border_value=(
                    self.obs_border_value
                    if self.obs_border_value is not None
                    else self.map_scale
                ),
            )
        else:
            raise ValueError("Unknown or not configured observation type")

        obs = self._map_obs_postprocessor(obs)

        # Scale image to grayscale
        obs = obs * (255 // self.map_scale)

        return obs.astype(np.uint8)
