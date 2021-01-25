import numpy as np

class Obstacle(object):
    def __init__(self, ego_agent, obstacles):
        '''
        Class that can handle obstacles in the environment
        ego_agent = agent[0]
        obstacles = list of all obstacles with their 4 cornerpoints
        m = the amount of closest obstacles you want in the list. Default = 6
        '''
        # Get position of ego_agent in coordinates
        self.ego_agent_pos = ego_agent.pos_global_frame
        self.obstacle = obstacles
        self.obstacles_no_ext = self.obstacle.copy()

    def get_list_of_nearest_obstacles(self, m=1):
        '''
        This function returns a list of 'm' closest obstacles to the ego_agent
        '''
        Nearest_info = self.get_info_nearest_obstacles()
        N_obstacles = len(self.obstacle)

        for i in range(N_obstacles):
            self.obstacle[i] = self.obstacles_no_ext[Nearest_info[i][1]]

        self.obstacle = self.obstacle[:m]
        return self.obstacle

    def get_info_nearest_obstacles(self):
        '''
        This function returns information about the obstacle and the the length to the agent.
        '''
        N_obstacles = len(self.obstacle)
        Nearest_info = []

        # Get one extra point in the middle between the corner points to get more accurate results
        for i in range(N_obstacles):
            self.obstacle[i] = self.add_middle_points(self.obstacle[i])

            # Determine which obstacle is closest to agent and make a list of this information
            shortest_length = np.inf
            contour_idx = self.obstacle[i]
            rel_coords = contour_idx - self.ego_agent_pos

            # Convert to a length
            for j in range(len(rel_coords)):
                l2norm = np.linalg.norm(rel_coords[j])
                shortest_length = min(shortest_length, l2norm)

            Nearest_info.append([shortest_length, i]) # Length next to obstacle id

        # Sort obstacles based on how close the obstacle is
        Nearest_info = sorted(Nearest_info)

        return Nearest_info

    def add_middle_points(self, obstacle):
        n_obstacles = len(obstacle)
        for i in range(n_obstacles):
            if i+1 > (n_obstacles - 1):
                middle_point = self.make_new_point(obstacle[i], obstacle[0])
            else:
                middle_point = self.make_new_point(obstacle[i], obstacle[i+1])
            obstacle = obstacle + [middle_point]
        return obstacle

    def make_new_point(self, first, second):
        middle_point = (first[0] - ((first[0] - second[0]) / 2), first[1] - ((first[1] - second[1]) / 2))
        return middle_point






