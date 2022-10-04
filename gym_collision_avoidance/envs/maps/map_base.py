import numpy as np
import cv2
import os

from typing import Any, Dict, Optional, Type, Union

from abc import ABC, abstractmethod

from gym_collision_avoidance.envs.maps.map_obs_util import (
    ego_rot_global_map,
    ego_fixed_global_map,
    ego_submap_from_map,
)


class BaseMap(ABC):
    def __init__(
        self,
        map_size: tuple,
        cell_size: float,
        obs_size: tuple,
        submap_lookahead: float = 3.0,
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
        self.submap_size = int(2 * submap_lookahead / self.cell_size)

        # map_scale is maximum value of cells in map (important for scaling to grayscale)
        self.map_scale = 1

        # Optional setting for border_value for image rotations when generating obs, if None map_scale is used
        self.obs_border_value = None

    def get_idc_from_pos(
        self, pos: Union[np.ndarray, list, tuple]
    ) -> Union[tuple, np.ndarray]:

        pos = np.array(pos)
        if pos.ndim == 1:
            onedim = True
            pos = pos.reshape((1, -1))
        else:
            onedim = False

        # Round to correct for numerical errors before doing np.floor
        pos = pos.round(self.cell_size_decimals)

        # OpenCV coordinate frame is in the top-left corner, x to the left, y downwards
        x_idx = np.floor((pos[:, [0]] + self.map_size[0] / 2) / self.cell_size)
        y_idx = np.floor((-pos[:, [1]] + self.map_size[1] / 2) / self.cell_size)

        if onedim:
            x_idx = (
                self.map.shape[1] - 1
                if x_idx > self.map.shape[1] - 1
                else (0 if x_idx < 0 else x_idx)
            )
            y_idx = (
                self.map.shape[0] - 1
                if y_idx > self.map.shape[0] - 1
                else (0 if y_idx < 0 else y_idx)
            )
            return int(y_idx), int(x_idx)
        else:
            x_idx = np.clip(x_idx, 0, self.map.shape[1] - 1)
            y_idx = np.clip(y_idx, 0, self.map.shape[0] - 1)
            return np.hstack((y_idx, x_idx)).astype(int)

    def get_pos_from_idc(self, idc: Union[np.ndarray, list, tuple]) -> np.ndarray:

        idc = np.array(idc)
        if idc.ndim == 1:
            onedim = True
            idc = idc.reshape((1, -1))
        else:
            onedim = False

        x = (idc[:, [1]]) * self.cell_size - self.map_size[
            0
        ] / 2  # + self.cell_size / 2
        y = (-idc[:, [0]]) * self.cell_size + self.map_size[
            1
        ] / 2  # - self.cell_size / 2
        if onedim:
            return np.hstack((x, y)).squeeze()
        else:
            return np.hstack((x, y))

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
        elif obs_type == "ego_rot_global_map":
            obs = ego_rot_global_map(
                map=map_array,
                map_cell=map_cell,
                angle=angle,
                output_size=self.obs_size,
                border_value=(
                    self.obs_border_value
                    if self.obs_border_value is not None
                    else self.map_scale
                ),
            )
        elif obs_type == "ego_fixed_global_map":
            obs = ego_fixed_global_map(
                map=map_array,
                map_cell=map_cell,
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
                pos_pxl=map_cell,
                angle_deg=angle,
                submap_size=self.submap_size,
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
