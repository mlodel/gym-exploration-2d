import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.config import Config
import math
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from gym_collision_avoidance.envs.sensors.ego_submap_from_map import ego_submap_from_map


class OccupancyGridSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.x_width = Config.SUBMAP_WIDTH
        self.y_width = Config.SUBMAP_HEIGHT
        self.grid_cell_size = Config.SUBMAP_RESOLUTION
        self.plot = False
        self.name = "local_grid"
        self.floatmap_rot = None
        self.map = None
        self.agent_idx = None

    def sense(self, agents, agent_index, top_down_map):

        # Get position of ego agent
        ego_agent = agents[agent_index]
        ego_agent_pos = ego_agent.pos_global_frame
        ego_agent_heading = ego_agent.heading_global_frame

        # Get map indices of ego agent
        agent_idx, _ = top_down_map.world_coordinates_to_map_indices(ego_agent_pos)
        agent_idx = np.flip(agent_idx)

        submap = ego_submap_from_map(
            top_down_map.map.astype(np.uint8),
            pos_pxl=agent_idx,
            angle_deg=180 * ego_agent_heading / np.pi,
            submap_size=[Config.SUBMAP_WIDTH, Config.SUBMAP_HEIGHT],
            scale_size=list(Config.SUBMAP_SCALE_TARGET),
        )

        # submap = np.expand_dims((submap * 255).astype(np.uint8), axis=0)
        submap = (submap * 255).astype(np.uint8)
        return submap

    def sense_old(self, agents, agent_index, top_down_map):

        """
        # Grab (i,j) coordinates of the upper right and lower left corner of the desired OG map, within the entire map
        host_agent = agents[agent_index]
        [map_i_high, map_j_low], _ = top_down_map.world_coordinates_to_map_indices(host_agent.pos_global_frame-np.array([3., 3]))
        [map_i_low, map_j_high], _ = top_down_map.world_coordinates_to_map_indices(host_agent.pos_global_frame+np.array([3., 3.]))

        # Assume areas outside static_map should be filled with zeros
        og_map = np.zeros((int(6/top_down_map.grid_cell_size), int(6/top_down_map.grid_cell_size)), dtype=bool)

        if map_i_low >= top_down_map.map.shape[0] or map_i_high < 0 or map_j_low >= top_down_map.map.shape[1] or map_j_high < 0:
            # skip rest ==> og_map and top_down_map have no overlap
            print("*** no overlap btwn og_map and top_down_map ***")
            print("*** map dims:", map_i_low, map_i_high, map_j_low, map_j_high)
            return og_map

        # super crappy logic to handle when the OG map partially overlaps with the map
        if map_i_low < 0:
            og_i_low = -map_i_low
            og_i_high = og_map.shape[0]
        elif map_i_high >= top_down_map.map.shape[0]:
            og_i_low = 0
            og_i_high = og_map.shape[0] - (map_i_high - top_down_map.map.shape[0])
        else:
            og_i_low = 0
            og_i_high = og_map.shape[0]

        if map_j_low < 0:
            og_j_low = -map_j_low
            og_j_high = og_map.shape[1]
        elif map_j_high >= top_down_map.map.shape[1]:
            og_j_low = 0
            og_j_high = og_map.shape[1] - (map_j_high - top_down_map.map.shape[1])
        else:
            og_j_low = 0
            og_j_high = og_map.shape[1]

        # Don't grab a map index outside the map's boundaries
        map_i_low = np.clip(map_i_low, 0, top_down_map.map.shape[0])
        map_i_high = np.clip(map_i_high, 0, top_down_map.map.shape[0])
        map_j_low = np.clip(map_j_low, 0, top_down_map.map.shape[1])
        map_j_high = np.clip(map_j_high, 0, top_down_map.map.shape[1])

        # Fill the correct OG map indices with the section of the map that has been selected
        og_map[og_i_low:og_i_high, og_j_low:og_j_high] = top_down_map.map[map_i_low:map_i_high, map_j_low:map_j_high]
        resized_og_map = self.resize(og_map)
        return resized_og_map

        """
        # Get position of ego agent
        ego_agent = agents[agent_index]
        ego_agent_pos = ego_agent.pos_global_frame
        ego_agent_heading = ego_agent.heading_global_frame

        # Get map indices of ego agent
        self.agent_idx, _ = top_down_map.world_coordinates_to_map_indices(ego_agent_pos)

        span_x = int(np.ceil(self.x_width))  # 60
        span_y = int(np.ceil(self.y_width))  # 60

        self.map = top_down_map.map

        # Expand map
        self.expand_map()

        # Get submap indices around ego agent
        start_idx_x, start_idx_y, end_idx_x, end_idx_y = self.getSubmapByIndices(
            span_x, span_y
        )

        # Get the batch_grid with filled in values
        # float_map = self.map.astype(float)
        # Rotate grid such that it is aligned with the heading
        self.rotate_grid_around_center(angle=-ego_agent_heading * 180 / np.pi)
        batch_grid = self.floatmap_rot[start_idx_y:end_idx_y, start_idx_x:end_idx_x]

        # batch_grid = self.fill_invisible(batch_grid)

        if self.plot:
            self.plot_top_down_map(
                top_down_map.map,
                ego_agent_pos_idx,
                start_idx_x,
                start_idx_y,
                ego_agent_heading,
                title="Original",
            )
            self.plot_top_down_map(
                float_map,
                ego_agent_pos_idx,
                start_idx_x,
                start_idx_y,
                ego_agent_heading,
                title="Rotated",
            )
            self.plot_batch_grid(batch_grid)

        if Config.SUBMAP_SCALE:
            batch_grid = cv2.resize(
                batch_grid, Config.SUBMAP_SCALE_TARGET, interpolation=cv2.INTER_CUBIC
            )

        batch_grid = batch_grid.astype(bool)

        return batch_grid

    # def fill_invisible(self, grid):
    #     grid = 1 - grid
    #     pos = (self.x_width//2, self.y_width//2)
    #     im = np.array(grid * 255, dtype=np.uint8)
    #
    #     contours, _ = cv2.findContours(im, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #
    #     for cnt in contours:
    #         if cv2.pointPolygonTest(cnt, pos, measureDist=False) == -1:
    #             cv2.drawContours(im, [cnt], 0, 0, -1)
    #
    #     return 1 - np.array(im / 255)

    # Plot
    def plot_top_down_map(
        self, top_down_map, ego_agent_idx, start_idx_x, start_idx_y, heading, title
    ):
        fig = plt.figure(title)
        ax = fig.subplots(1)
        ax.imshow(top_down_map, aspect="equal")
        ax.scatter(ego_agent_idx[1], ego_agent_idx[0], s=100, c="red", marker="o")
        rect = patches.Rectangle(
            (start_idx_y, start_idx_x),
            self.x_width,
            self.y_width,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        aanliggend = 20 * math.cos(heading)
        overstaand = -20 * math.sin(heading)
        if "Rotated" == str(title):
            ax.arrow(
                ego_agent_idx[1],
                ego_agent_idx[0],
                20,
                0,
                width=3,
                head_width=10,
                head_length=10,
                fc="yellow",
            )  # agent poiting direction
        else:
            ax.arrow(
                ego_agent_idx[1],
                ego_agent_idx[0],
                aanliggend,
                overstaand,
                width=3,
                fc="yellow",
                head_width=10,
                head_length=10,
            )  # agent poiting direction

        plt.show()

    def plot_batch_grid(self, batch_grid):
        fig = plt.figure("batch_grid")
        ax = fig.subplots(1)
        ax.imshow(batch_grid, aspect="equal")
        ax.scatter(self.x_width / 2, self.y_width / 2, s=100, c="red", marker="o")
        ax.arrow(
            self.x_width / 2,
            self.y_width / 2,
            10,
            0,
            width=1,
            head_width=3,
            head_length=3,
            fc="yellow",
        )  # agent poiting direction

        plt.show()

    def resize(self, og_map):
        resized_og_map = og_map.copy()
        return resized_og_map

    def rotate_grid_around_center(self, angle):
        """
        inputs:
          grid: numpy array (gridmap) that needs to be rotated
          angle: rotation angle in degrees
        """
        agent_pos = self.agent_idx

        # Rotate grid into direction of initial heading
        rows, cols = self.map.shape
        M = cv2.getRotationMatrix2D(
            center=(float(agent_pos[1]), float(agent_pos[0])), angle=angle, scale=1
        )
        self.floatmap_rot = cv2.warpAffine(
            self.map.astype(float), M, (cols, rows), borderValue=1.0
        )

    def expand_map(self):

        # submap_size = [Config.SUBMAP_WIDTH / Config.SUBMAP_RESOLUTION, Config.SUBMAP_HEIGHT / Config.SUBMAP_RESOLUTION]
        submap_size = (np.array([Config.SUBMAP_WIDTH, Config.SUBMAP_HEIGHT])).astype(
            int
        )
        # map_size = [Config.MAP_WIDTH / Config.SUBMAP_RESOLUTION, Config.MAP_HEIGHT / Config.SUBMAP_RESOLUTION]
        map_size = (
            np.array([Config.MAP_WIDTH, Config.MAP_HEIGHT]) / Config.SUBMAP_RESOLUTION
        ).astype(int)

        if self.agent_idx[1] + submap_size[0] / 2 > map_size[1]:
            trues_v = np.ones([map_size[0], submap_size[1] // 2], bool)
            self.map = np.c_[self.map, trues_v]
            map_size[1] += submap_size[1] / 2
        if self.agent_idx[0] + submap_size[1] / 2 > map_size[0]:
            trues_h = np.ones([submap_size[0] // 2, map_size[1]], bool)
            self.map = np.r_[self.map, trues_h]
            map_size[0] += submap_size[0] / 2
        if self.agent_idx[1] - submap_size[0] / 2 < 0:
            trues_v = np.ones([map_size[0], submap_size[1] // 2], bool)
            self.map = np.c_[trues_v, self.map]
            map_size[1] += submap_size[0] / 2
            self.agent_idx[1] += submap_size[0] / 2
        if self.agent_idx[0] - submap_size[1] / 2 < 0:
            trues_h = np.ones([submap_size[0] // 2, map_size[1]], bool)
            self.map = np.r_[trues_h, self.map]
            map_size[0] += submap_size[1] / 2
            self.agent_idx[0] += submap_size[1] / 2

    def getSubmapByIndices(self, span_x, span_y):
        """
        Extract a submap of span (span_x, span_y) around
        center index (center_idx_x, center_idx_y)
        """

        center_idx_x = self.agent_idx[1]
        center_idx_y = self.agent_idx[0]

        # Start corner indices of the submap
        start_idx_x = max(0, int(center_idx_x - np.floor(span_x / 2)))
        start_idx_y = max(0, int(center_idx_y - np.floor(span_y / 2)))

        # Compute end indices (assure size of submap is correct, if out pf bounds)
        max_idx_x = self.map.shape[1] - 1
        max_idx_y = self.map.shape[0] - 1

        # End indices of the submap (this corrects for the bounds of the grid
        end_idx_x = start_idx_x + span_x
        if end_idx_x > max_idx_x:
            end_idx_x = max_idx_x
            start_idx_x = end_idx_x - span_x
        end_idx_y = start_idx_y + span_y
        if end_idx_y > max_idx_y:
            end_idx_y = max_idx_y
            start_idx_y = end_idx_y - span_y

        return start_idx_x, start_idx_y, end_idx_x, end_idx_y


if __name__ == "__main__":
    from gym_collision_avoidance.envs.Map import Map
    from gym_collision_avoidance.envs.agent import Agent
    from gym_collision_avoidance.envs.policies.NonCooperativePolicy import (
        NonCooperativePolicy,
    )
    from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics

    obstacle_1 = [(10, 10), (-6, 10), (-6, -6), (10, -6)]
    obstacle_5 = [(-14.5, 15), (-15, 15), (-15, -15), (-14.5, -15)]
    obstacle_6 = [(15, 15), (14.5, 15), (14.5, -15), (15, -15)]
    obstacle_7 = [(-15, 14.5), (-15, 15), (15, 15), (15, 14.5)]
    obstacle_8 = [(15, -14.5), (-15, -14.5), (-15, -15), (15, -15)]
    obstacle = []
    obstacle.extend([obstacle_1, obstacle_5, obstacle_6, obstacle_7, obstacle_8])

    top_down_map = Map(
        x_width=30, y_width=30, grid_cell_size=0.1, map_filename=obstacle
    )
    agents = [
        Agent(
            11,
            11,
            10,
            10,
            0.5,
            1.0,
            0 * np.pi / 180,
            NonCooperativePolicy,
            UnicycleDynamics,
            [],
            0,
        )
    ]
    # top_down_map.add_agents_to_map(agents)
    og = OccupancyGridSensor()
    og_map = og.sense(agents, 0, top_down_map)

    # print(og_map)

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(og_map, extent=[-30, 30, -30, 30])
    ax.scatter(0, 0, s=100, c="red", marker="o")
    ax.arrow(
        0, 0, 5, 0, width=0.5, head_width=1.5, head_length=1.5, fc="yellow"
    )  # agent poiting direction
    pos = og.agent_idx
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(og.floatmap_rot)
    ax2.scatter(pos[1], pos[0], s=10, c="red", marker="o")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(top_down_map.map)
    # ax3.scatter(pos[1], pos[0], s=10, c='red', marker='o')

    plt.show()
