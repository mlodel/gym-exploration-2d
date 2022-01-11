import numpy as np
from gym_collision_avoidance.envs.config import Config

class StaticObstacleAvoidance(object):
    def __init__(self,str="StaticObstacleAvoidance"):

        self.free_space_assumption_ = False
        self.occupied_threshold_ = 30
        self.collision_free_delta_max_ = 3
        self.res = Config.SUBMAP_RESOLUTION
        self.width = int(Config.MAP_WIDTH/self.res)
        self.height = int(Config.MAP_HEIGHT/self.res)
        self.FORCES_N = Config.FORCES_N

        self.collision_free_C1 = np.zeros(self.FORCES_N)
        self.collision_free_C2 = np.zeros(self.FORCES_N)
        self.collision_free_C3 = np.zeros(self.FORCES_N)
        self.collision_free_C4 = np.zeros(self.FORCES_N)

        self.collision_free_a1x = np.zeros(self.FORCES_N)
        self.collision_free_a1y = np.zeros(self.FORCES_N)
        self.collision_free_a2x = np.zeros(self.FORCES_N)
        self.collision_free_a2y = np.zeros(self.FORCES_N)
        self.collision_free_a3x = np.zeros(self.FORCES_N)
        self.collision_free_a3y = np.zeros(self.FORCES_N)
        self.collision_free_a4x = np.zeros(self.FORCES_N)
        self.collision_free_a4y = np.zeros(self.FORCES_N)

        self.collision_free_xmin = np.zeros(self.FORCES_N)
        self.collision_free_xmax = np.zeros(self.FORCES_N)
        self.collision_free_ymin = np.zeros(self.FORCES_N)
        self.collision_free_ymax = np.zeros(self.FORCES_N)

    def ComputeCollisionFreeArea(self,pred_traj_, static_map, radius):

        search_steps = 10
        self.static_map_ = static_map
        close_to_obs = False
        for N in range(self.FORCES_N):

            # Convert to map coordinates
            x_i = int((pred_traj_[N,0])/self.res+self.width/2)
            y_i = int((pred_traj_[N,1])/self.res+self.height/2)
            psi_path = pred_traj_[N,2]

            # define maximum search distance in occupancy grid cells, based on discretization
            r_max_i_min = int(-self.collision_free_delta_max_ / self.res)
            r_max_i_max = int(self.collision_free_delta_max_ / self.res)

            # Initialize found rectangle values with maxium search distance
            x_min = r_max_i_min
            x_max = r_max_i_max
            y_min = r_max_i_min
            y_max = r_max_i_max

            # Initialize search distance iterator int
            search_distance = 1
            # Initialize boolean that indicates whether the region has been found
            search_region = True

            # Iterate until the region is found
            while search_region:

                # Only search in x_min direction if no value has been found yet
                if (x_min == r_max_i_min):
                    search_x = -search_distance
                    for search_y_it in range(max(-search_distance, y_min),min(search_distance, y_max)):
                        # Assign value if occupied cell is found
                        if (self.getRotatedOccupancy(x_i, search_x, y_i, search_y_it, psi_path)):#> self.occupied_threshold_):
                            x_min = search_x

                # Only search in x_max direction if no value has been found yet
                if (x_max == r_max_i_max):
                    search_x = search_distance
                    for search_y_it in range(max(-search_distance, y_min),min(search_distance, y_max)):
                        # Assign value if occupied cell is found
                        if (self.getRotatedOccupancy(x_i, search_x, y_i, search_y_it, psi_path)):# > self.occupied_threshold_):
                            x_max = search_x

                # Only search in y_min direction if no value has been found yet
                if (y_min == r_max_i_min):
                    search_y = -search_distance
                    for search_x_it in range(max(-search_distance, x_min),min(search_distance, x_max)):
                        # Assign value if occupied cell is found
                        if (self.getRotatedOccupancy(x_i, search_x_it, y_i, search_y, psi_path)):# > self.occupied_threshold_):
                            y_min = search_y

                # Only search in y_min direction if no value has been found yet
                if (y_max == r_max_i_max):
                    search_y = search_distance;
                for search_x_it in range(max(-search_distance, x_min),min(search_distance, x_max)):
                    # Assign value if occupied cell is found
                    if (self.getRotatedOccupancy(x_i, search_x_it, y_i, search_y, psi_path)):# > self.occupied_threshold_):
                        y_max = search_y

                # Increase search distance
                search_distance += 1
                # Determine whether the search is finished
                search_region = (search_distance < r_max_i_max) and (x_min == r_max_i_min or x_max == r_max_i_max or y_min == r_max_i_min or y_max == r_max_i_max)

            # Assign the rectangle values
            self.collision_free_xmin[N] = x_min * self.res +0.6
            self.collision_free_xmax[N] = x_max * self.res -0.6
            self.collision_free_ymin[N] = y_min * self.res +0.6
            self.collision_free_ymax[N] = y_max * self.res -0.6

            # Box corner points
            obs1 = np.array((pred_traj_[N,0] + np.cos(psi_path) * self.collision_free_xmax[N], pred_traj_[N,1] + np.sin(psi_path) * self.collision_free_xmax[N]))
            obs2 = np.array((pred_traj_[N,0] + np.cos(psi_path) * self.collision_free_xmin[N], pred_traj_[N,1] + np.sin(psi_path) * self.collision_free_xmin[N]))
            obs3 = np.array((pred_traj_[N,0] - np.sin(psi_path) * self.collision_free_ymax[N], pred_traj_[N,1] + np.cos(psi_path) * self.collision_free_ymax[N]))
            obs4 = np.array((pred_traj_[N,0] - np.sin(psi_path) * self.collision_free_ymin[N], pred_traj_[N,1] + np.cos(psi_path) * self.collision_free_ymin[N]))

            if N == 0:
                corner1 = np.array((pred_traj_[N, 0] + np.cos(psi_path) * self.collision_free_xmax[N]- np.sin(psi_path) * self.collision_free_ymax[N],
                                 pred_traj_[N, 1] + np.sin(psi_path) * self.collision_free_xmax[N]+ np.cos(psi_path) * self.collision_free_ymax[N]))
                corner2 = np.array((pred_traj_[N, 0] + np.cos(psi_path) * self.collision_free_xmin[N]- np.sin(psi_path) * self.collision_free_ymax[N],
                                 pred_traj_[N, 1] + np.sin(psi_path) * self.collision_free_xmin[N]+ np.cos(psi_path) * self.collision_free_ymax[N]))
                corner4 = np.array((pred_traj_[N, 0] + np.cos(psi_path) * self.collision_free_xmax[N]- np.sin(psi_path) * self.collision_free_ymin[N],
                                 pred_traj_[N, 1] + np.sin(psi_path) * self.collision_free_xmax[N]+ np.cos(psi_path) * self.collision_free_ymin[N]))
                corner3 = np.array((pred_traj_[N, 0] + np.cos(psi_path) * self.collision_free_xmin[N]- np.sin(psi_path) * self.collision_free_ymin[N],
                                 pred_traj_[N, 1] + np.sin(psi_path) * self.collision_free_xmin[N]+ np.cos(psi_path) * self.collision_free_ymin[N]))
                self.obstacles = [corner1,corner2,corner3,corner4]

            n1 = obs1 - pred_traj_[N,:2]
            n2 = obs2 - pred_traj_[N,:2]
            n3 = obs3 - pred_traj_[N,:2]
            n4 = obs4 - pred_traj_[N,:2]

            distances = np.array([np.linalg.norm(n1),np.linalg.norm(n2),np.linalg.norm(n3),np.linalg.norm(n4)])
            """
            if np.min(distances) < 0.6:
                close_to_obs = True
                self.collision_free_a1x[N] = self.collision_free_a1x[N-1]
                self.collision_free_a2x[N] = self.collision_free_a2x[N-1]
                self.collision_free_a3x[N] = self.collision_free_a3x[N-1]
                self.collision_free_a4x[N] = self.collision_free_a4x[N-1]

                self.collision_free_a1y[N] = self.collision_free_a1y[N-1]
                self.collision_free_a2y[N] = self.collision_free_a2y[N-1]
                self.collision_free_a3y[N] = self.collision_free_a3y[N-1]
                self.collision_free_a4y[N] = self.collision_free_a4y[N-1]

                # b_ = p12_norm.transpose *p2 - r_disc - obstacle size
                self.collision_free_C1[N] = self.collision_free_C1[N-1]  # -(radius + 0.3)
                self.collision_free_C2[N] = self.collision_free_C2[N-1]
                self.collision_free_C3[N] = self.collision_free_C3[N-1]
                self.collision_free_C4[N] = self.collision_free_C4[N-1]
            elif close_to_obs:
                self.collision_free_a1x[N] = self.collision_free_a1x[N-1]
                self.collision_free_a2x[N] = self.collision_free_a2x[N-1]
                self.collision_free_a3x[N] = self.collision_free_a3x[N-1]
                self.collision_free_a4x[N] = self.collision_free_a4x[N-1]

                self.collision_free_a1y[N] = self.collision_free_a1y[N-1]
                self.collision_free_a2y[N] = self.collision_free_a2y[N-1]
                self.collision_free_a3y[N] = self.collision_free_a3y[N-1]
                self.collision_free_a4y[N] = self.collision_free_a4y[N-1]

                # b_ = p12_norm.transpose *p2 - r_disc - obstacle size
                self.collision_free_C1[N] = self.collision_free_C1[N-1]  # -(radius + 0.3)
                self.collision_free_C2[N] = self.collision_free_C2[N-1]
                self.collision_free_C3[N] = self.collision_free_C3[N-1]
                self.collision_free_C4[N] = self.collision_free_C4[N-1]
            else:
            """
            if True:
                # (p2 -p1)/norm(p2-p1)
                A1 = n1/np.linalg.norm(n1)
                A2 = n2/np.linalg.norm(n2)
                A3 = n3/np.linalg.norm(n3)
                A4 = n4/np.linalg.norm(n4)

                # ax_[0] = p12_norm(0) , ay_[0] = p12_norm(1)
                self.collision_free_a1x[N] = A1[0]
                self.collision_free_a2x[N] = A2[0]
                self.collision_free_a3x[N] = A3[0]
                self.collision_free_a4x[N] = A4[0]

                self.collision_free_a1y[N] = A1[1]
                self.collision_free_a2y[N] = A2[1]
                self.collision_free_a3y[N] = A3[1]
                self.collision_free_a4y[N] = A4[1]

                # b_ = p12_norm.transpose *p2 - r_disc - obstacle size
                self.collision_free_C1[N] = np.matmul(np.transpose(A1),obs1)
                self.collision_free_C2[N] = np.matmul(np.transpose(A2),obs2)
                self.collision_free_C3[N] = np.matmul(np.transpose(A3),obs3)
                self.collision_free_C4[N] = np.matmul(np.transpose(A4),obs4)

    def getRotatedOccupancy(self,x_i,search_x,y_i,search_y,psi):

        x_search_rotated = int(np.cos(psi) * search_x - np.sin(psi) * search_y)
        y_search_rotated = int(np.sin(psi) * search_x + np.cos(psi) * search_y)

        # Check if position is outside the map boundaries
        if ((x_i + x_search_rotated) > self.width or (y_i + y_search_rotated) > self.height or (x_i + x_search_rotated) < 0 or (y_i + y_search_rotated) < 0):
            if self.free_space_assumption_:
                return False
            else:
                return True
        else:
            return self.static_map_[self.height-y_i + -y_search_rotated, x_i + x_search_rotated]