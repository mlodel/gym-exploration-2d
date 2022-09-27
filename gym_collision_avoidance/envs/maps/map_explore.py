from gym_collision_avoidance.envs.maps.map_base import BaseMap
from gym_collision_avoidance.envs.maps.map_env import EnvMap

from typing import Any, Dict, Optional, Type, Union

import cv2
import numpy as np

import skimage


class ExploreMap(BaseMap):
    def __init__(
        self,
        map_size: tuple,
        cell_size: float,
        sensing_range: float,
        sensing_fov: float = 2 * np.pi,
        obs_size=(80, 80),
        submap_lookahead: float = 3.0,
    ):
        super().__init__(
            map_size, cell_size, obs_size, submap_lookahead=submap_lookahead
        )
        self.map_unknown_color = 127
        self.map += self.map_unknown_color
        self.map_scale = 255
        self.sens_range = np.around(sensing_range / cell_size).astype(int)
        self.sens_fov = sensing_fov

    def update(self, pose: np.ndarray, global_map: EnvMap = None):
        if global_map is None:
            raise ValueError("No global map passed for map update")

        super().update(pose)

        self._update_masking(pose, global_map)

    def _update_masking(self, pose, global_map):

        cell_pos = global_map.get_idc_from_pos(pose)

        # Calculate offset between explore map and global map
        offset_x = max(0, (self.map.shape[1] - global_map.map.shape[1]) // 2)
        offset_y = max(0, (self.map.shape[0] - global_map.map.shape[0]) // 2)

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
        self.map[
            offset_y : global_map.map.shape[0] + offset_y,
            offset_x : global_map.map.shape[1] + offset_x,
        ] = cv2.bitwise_and(
            self.map[
                offset_y : global_map.map.shape[0] + offset_y,
                offset_x : global_map.map.shape[1] + offset_x,
            ],
            self.map[
                offset_y : global_map.map.shape[0] + offset_y,
                offset_x : global_map.map.shape[1] + offset_x,
            ],
            mask=mask_inv,
        )

        cv2.imshow("update", update_contours)
        cv2.waitKey(1)

        # Color unknown areas in visible area (mask) grey
        update_unknown = np.zeros_like(mask)
        update_unknown[update_filled == 255] = self.map_unknown_color
        # TODO

        # Combine obstacle contour with unknown areas in visible area
        update = cv2.bitwise_or(update_contours, update_unknown, mask=mask)

        # Add update to map prior (where visible area is 0)
        self.map[
            offset_y : global_map.map.shape[0] + offset_y,
            offset_x : global_map.map.shape[1] + offset_x,
        ] = cv2.add(
            self.map[
                offset_y : global_map.map.shape[0] + offset_y,
                offset_x : global_map.map.shape[1] + offset_x,
            ],
            update,
        )

        # ----------------
        # Not working:
        # Fill closed obstacles
        # contours, hierarchy = cv2.findContours(
        #     self.map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        # )
        # # draw all contours to image (green if opened else red)
        # for i in range(len(contours)):
        #     opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
        #     if not opened:
        #         cv2.drawContours(self.map, contours, i, color=2, thickness=-1)

    def _update_from_pointcloud(self, pose, global_map):

        # Get the local point cloud from the global map
        points_occ = global_map.get_local_pointcloud(
            pos=pose[:2], lookahead=2 * self.sens_range * self.cell_size, scope="circle"
        )
        points_free = global_map.get_local_pointcloud(
            pos=pose[:2],
            lookahead=2 * self.sens_range * self.cell_size,
            scope="circle",
            free=True,
        )

        # convert points to map idc
        points_occ_idc = self.get_idc_from_pos(points_occ)
        points_free_idc = self.get_idc_from_pos(points_free)

        # Update the map with the point cloud
        self.map[points_occ_idc[:, 0], points_occ_idc[:, 1]] = self.map_scale
        self.map[points_free_idc[:, 0], points_free_idc[:, 1]] = 0

        cv2.imshow("map", self.map)
        cv2.waitKey(1)

    def _update_laser_scan(self, pose, global_map):

        # Get the idc of the current pose
        pose_idx = self.get_idc_from_pos(pose[:2])

        # Generate indices of a circle with radius self.sens_range
        # and center at the current pose
        circle_grid = cv2.circle(
            np.zeros_like(self.map),
            tuple(pose_idx[::-1]),
            self.sens_range,
            1,
            1,
        )
        circle_idc = np.argwhere(circle_grid == 1)

        for idx in circle_idc:
            # Get line idc between the current pose and the current circle idx
            line_idc = np.array(
                skimage.draw.line(pose_idx[0], pose_idx[1], idx[0], idx[1])
            ).T

            # Clip line_idc to global map shape
            # line_idc = np.clip(line_idc, 0, np.array(self.map.shape) - 1)

            # Convert line_idc to global map idc
            line_pos = self.get_pos_from_idc(line_idc)
            line_idc_global = global_map.get_idc_from_pos(line_pos)

            # Get global map values along the line
            line_values = global_map.map[line_idc_global[:, 0], line_idc_global[:, 1]]

            # Check if the line is blocked by an obstacle
            if np.any(line_values == 1):
                # If yes, find the first obstacle idx
                first_obst_idx = np.argwhere(line_values == 1)[0][0]
                # Set the map value at line idx of the first obstacle idx to map_scale
                self.map[
                    line_idc[first_obst_idx, 0], line_idc[first_obst_idx, 1]
                ] = self.map_scale
                # Set all map values before first obstacle idx to 0
                self.map[line_idc[:first_obst_idx, 0], line_idc[:first_obst_idx, 1]] = 0
            else:
                # if not, set all map values along the line to 0
                self.map[line_idc[:, 0], line_idc[:, 1]] = 0

    def _map_obs_postprocessor(self, map_array):
        # Invert to make free areas white, occupied black
        return cv2.bitwise_not(map_array)
