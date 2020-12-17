import numpy as np
import math
import pylab as pl
from gym_collision_avoidance.envs.sensors.Sensor import Sensor
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.sensors.OccupancyGridSensor import OccupancyGridSensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class AngularMapSensor(Sensor):
    def __init__(self):
        Sensor.__init__(self)
        self.no_of_slices = Config.NUM_OF_SLICES
        self.max_range = Config.MAX_RANGE
        self.x_width = 20 * self.max_range
        self.y_width = 20 * self.max_range
        # For full FOV:
        self.angle_max = np.pi
        self.angle_min = -np.pi
        self.name = 'angular_map'
        self.plot = False
        self.Laserscan = False
        self.Occupancygrid = True

        # For laserscan
        if self.Laserscan:
            self.num_beams = Config.LASERSCAN_LENGTH
            self.range_resolution = 0.1
            self.min_range = 0  # meters
            self.min_angle = -np.pi
            self.max_angle = np.pi

            self.angles = np.linspace(self.min_angle, self.max_angle, self.num_beams)
            self.ranges = np.arange(self.min_range, self.max_range, self.range_resolution)
            self.debug = True

        if self.Occupancygrid:
            self.grid_cell_size = Config.SUBMAP_RESOLUTION
            #self.plot = False
            self.name = 'local_grid'

    def sense(self, agents, agent_index, top_down_map):
        '''
        The angular map encodes the distance of the next obstacle within angular ranges.
        The output of the map is a one dimensional vector 'l'.

        This function looks at every obstacle inside a grid around the ego agent. For every point in the obstacle it
        calculates the distance to the the ego_agent and in which slice it is using polar coordinates.
        Then it saves the point of the obstacle that is closest to the agent inside that slice.
        '''

        # Get position of ego agent
        self.ego_agent = agents[agent_index]

        # Initialize angular map
        Angular_Map = self.max_range * np.ones([self.no_of_slices])  # vector of 72

        # Angles
        self.radial_resolution = (self.angle_max - self.angle_min) / self.no_of_slices

        ## Batch grid
        if self.Occupancygrid:
            sensor = OccupancyGridSensor
            batch_grid = sensor.sense(self, agents, agent_index, top_down_map)
            angular_map = self.angular_map_from_batch_grid(Angular_Map, top_down_map)

        ## Laserscan
        if self.Laserscan:
            sensor = LaserScanSensor
            ranges = sensor.sense(self, agents, agent_index, top_down_map)
            angular_map = self.angular_map_from_laser_scan(Angular_Map, ranges)

        if self.plot:
            self.plot_angular_grid(angular_map)

        return angular_map

    def angular_map_from_batch_grid(self, Angular_Map, top_down_map):
        ## Get data from batch grid

        ego_agent_pos = self.ego_agent.pos_global_frame

        # Get map indices of ego agent
        ego_agent_pos_idx, _ = top_down_map.world_coordinates_to_map_indices(ego_agent_pos)

        span_x = int(np.ceil(self.x_width))
        span_y = int(np.ceil(self.y_width))

        # Get submap indices around ego agent
        start_idx_x, start_idx_y, end_idx_x, end_idx_y = top_down_map.getSubmapByIndices(ego_agent_pos_idx[0],
                                                                                         ego_agent_pos_idx[1], span_x,
                                                                                         span_y)
        # Get the batch_grid with filled in values
        batch_grid = top_down_map.map[start_idx_x:end_idx_x, start_idx_y:end_idx_y]

        # Get all indices where an obstacle is
        idx = np.where(batch_grid == True)

        ego_idx = np.array([0, 0])
        # Get euclidean distance
        for i in range(len(idx[0])):
            ind_x = idx[0][i]
            ind_y = idx[1][i]
            ind = self.indices_to_map_world_coordinates(np.array([ind_x, ind_y]))
            rel_coords = ind - ego_idx  # Relative coordinate
            # Convert to polar coordinates
            l2norm = np.linalg.norm(rel_coords)  # Distance between ego agent and obstacle point
            l2norm = l2norm * 0.1
            # We start counting from positive x-axis
            phi = math.atan2(rel_coords[1], rel_coords[0])
            rad_idx = int(phi / self.radial_resolution)
            if rad_idx < 0:
                rad_idx = rad_idx + self.no_of_slices
            Angular_Map[rad_idx] = min(Angular_Map[rad_idx], l2norm)

        return Angular_Map

    #todo: not working properly yet
    def angular_map_from_laser_scan(self, Angular_Map, ranges):
        # Take the heading of the agent into account
        heading = self.ego_agent.heading_global_frame
        phi = self.angles + heading

        # Get correct ranges for a specific slice
        rad_idx = phi / self.radial_resolution
        rad_idx = rad_idx.astype(int)

        # range value = distance value already!!
        j = 0
        for i in rad_idx:
            if i >= 0 and i < self.no_of_slices:
                Angular_Map[i] = min(Angular_Map[i], ranges[j])
                j += 1

        return Angular_Map

    def indices_to_map_world_coordinates(self, ind):
        pos_x = (ind[1]-(self.x_width/2))
        pos_y = -(ind[0]-(self.y_width/2))
        world_coords = np.array([pos_x, pos_y])
        return world_coords

    # Plot
    def plot_angular_grid(self, Angular_Map):
        fig = pl.figure("Angular grid")
        ax_ped_grid = pl.subplot()
        ax_ped_grid.clear()
        #Angular_Map_flipped = np.flip(Angular_Map)
        self.plot_Angular_map_vector(ax_ped_grid, Angular_Map, max_range=6.0, min_angle=0.0,
                                     max_angle=2 * np.pi)
        ax_ped_grid.plot(30, 30, color='r', marker='o', markersize=4)
        ax_ped_grid.scatter(0, 0, s=100, c='red', marker='o')
        #ax_ped_grid.arrow(0, 0, 1, 0, head_width=0.5,
        #                 head_length=0.5)  # agent poiting direction
        # x- and y-range only need to be [-1, 1] since the pedestrian grid is normalized
        ax_ped_grid.set_xlim([-self.max_range - 1, self.max_range + 1])
        ax_ped_grid.set_ylim([-self.max_range - 1, self.max_range + 1])
        fig.canvas.draw()

        # sleep(0.5)  # Time in seconds.
        #pl.show(block=False)
        #sleep(0.5)  # Time in seconds.

    def plot_Angular_map_vector(self, ax, Angular_Map, max_range=6, min_angle=0.0, max_angle=2 * np.pi):
        number_elements = Angular_Map.shape[0]
        angular_resolution = (self.angle_max - self.angle_min) / self.no_of_slices

        cmap = pl.get_cmap('gnuplot')

        for ii in range(number_elements):
            angle_start = (min_angle + ii * angular_resolution) * 180 / np.pi
            angle_end = (min_angle + (ii + 1) * angular_resolution) * 180 / np.pi

            distance_cone = pl.matplotlib.patches.Wedge((0.0, 0.0),
                                                        Angular_Map[ii],
                                                        angle_start, angle_end,
                                                        facecolor=cmap(Angular_Map[ii] / max_range),
                                                        alpha=0.5)

            ax.add_artist(distance_cone)

