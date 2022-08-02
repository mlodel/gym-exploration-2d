import copy

import numpy as np
import cv2

# import scipy
from gym_collision_avoidance.envs.information_models.edfMap import edfMap
from gym_collision_avoidance.envs.config import Config


class targetMap:
    def __init__(
        self,
        mapSize,
        cellSize,
        sensFOV,
        sensRange,
        rOcc,
        rEmp,
        edfmap_res_factor,
        tolerance=0.01,
        prior=0.0,
        p_false_neg=0.1,
        p_false_pos=0.05,
        logmap_bound=30.0,
    ):

        self.edfMapObj = edfMap(cellSize / edfmap_res_factor, mapSize)

        self.cellSize = cellSize
        self.mapSize = np.asarray(mapSize)
        self.sensFOV = sensFOV
        self.sensRange = sensRange

        self.lOcc = np.log(rOcc)
        self.lEmp = np.log(rEmp)
        self.rOcc = rOcc
        self.rEmp = rEmp
        self.tolerance = tolerance

        self.p_false_neg = p_false_neg
        self.p_false_pos = p_false_pos

        shape = (
            int(self.mapSize[1] / self.cellSize),
            int(self.mapSize[0] / self.cellSize),
        )
        self.map = np.ones(shape) * prior

        p_prior = np.exp(prior) / (np.exp(prior) + 1)
        self.probMap = np.ones(shape) * p_prior
        # self.logMap = np.log(self.map)

        self.logMap_bound = logmap_bound

        cell_entropy_prior = (
            -p_prior * np.log(p_prior) - (1 - p_prior) * np.log(1 - p_prior)
        ) / np.log(2)
        self.entropyMap = np.ones(shape) * cell_entropy_prior

        self.binaryMap = np.zeros(shape).astype(bool)

        self.received_map = False
        self.free_cells = set()
        self._init_free_cells()
        self.n_free_cells = len(self.free_cells)
        self.entropy_free_space = (
            cell_entropy_prior * self.n_free_cells
        )  # * shape[0] * shape [1]

        self.visitedCells = set()
        self.visited_share = 0.0

        self.thres_share_vis_cells = Config.IG_THRES_VISITED_CELLS
        self.thres_entropy = Config.IG_THRES_AVG_CELL_ENTROPY * self.n_free_cells

        self.finished_binary = False
        self.finished_entropy = False
        self.finished = False

        self.ego_map_inner_size = int(np.ceil(np.sqrt(shape[0] ** 2 + shape[1] ** 2)))

        # ego centric entropy map
        self.ego_map = self.create_ego_map(
            np.zeros(3), self.entropyMap, self.ego_map_inner_size
        )
        self.bin_ego_map = self.create_ego_map(
            np.zeros(3), (~self.binaryMap).astype(float), self.ego_map_inner_size
        )

        # Goal Map init
        # self.goal_map = np.zeros(Config.EGO_MAP_SIZE)
        self.current_goals = []
        self.goal_map = np.zeros(shape)
        self.goal_cells = []
        self.goal_ego_map = self.create_ego_map(
            np.zeros(3),
            self.goal_map.astype(float),
            self.ego_map_inner_size,
        )
        self.goal_completion_counter = 0

        # Multi-Channel Map
        self.mc_ego_binary_goal = np.stack((self.bin_ego_map, self.goal_ego_map))

    def _init_free_cells(self):

        self.free_cells.clear()
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.received_map:
                    pose = self.getPoseFromCell((i, j))
                    if self.edfMapObj.get_edf_value_from_pose(pose) >= 0.001:
                        self.free_cells.add((i, j))
                else:
                    self.free_cells.add((i, j))

    def update_map(self, occ_map=None, edf_map=None):
        map_updated = False

        if occ_map is not None:
            self.edfMapObj.update_from_occmap(occ_map)
            map_updated = True
        elif edf_map is not None:
            self.edfMapObj.update_from_edfmap(edf_map)
            map_updated = True

        if map_updated:
            self.received_map = True
            self._init_free_cells()

    def getCellsFromPose(self, pose):
        if len(pose) > 2:
            pose = pose[0:2]
        # OpenCV coordinate frame is in the top-left corner, x to the left, y downwards
        xIdc = np.floor((pose[0] + self.mapSize[0] / 2) / self.cellSize)
        yIdc = np.floor((-pose[1] + self.mapSize[1] / 2) / self.cellSize)

        # xIdc = np.clip(xIdc, 0, self.map.shape[1] - 1)
        xIdc = (
            self.map.shape[1] - 1
            if xIdc > self.map.shape[1] - 1
            else (0 if xIdc < 0 else xIdc)
        )
        # yIdc = np.clip(yIdc, 0, self.map.shape[0] - 1)
        yIdc = (
            self.map.shape[0] - 1
            if yIdc > self.map.shape[1] - 1
            else (0 if yIdc < 0 else yIdc)
        )

        return (int(yIdc), int(xIdc))

    def getPoseFromCell(self, cell):
        x = (cell[1]) * self.cellSize - self.mapSize[0] / 2 + self.cellSize / 2
        y = (-cell[0]) * self.cellSize + self.mapSize[1] / 2 - self.cellSize / 2
        return np.array([x, y])

    def get_pos_in_map_lims(self, pose):
        if len(pose) > 2:
            pose = pose[0:2]
        return np.max(
            np.array(
                [np.min(np.array([pose, self.mapSize / 2]), axis=0), -self.mapSize / 2]
            ),
            axis=0,
        )

    def getVisibleCells(self, pose):

        # Robot heading angle
        phi = pose[2]
        ## Get rectangular map section to be updated
        # FOV center, left, right limiting point
        if self.sensFOV <= np.pi:
            left = pose[0:2] + self.sensRange * np.array(
                [np.cos(phi + self.sensFOV / 2), np.sin(phi + self.sensFOV / 2)]
            )
            right = pose[0:2] + self.sensRange * np.array(
                [np.cos(phi - self.sensFOV / 2), np.sin(phi - self.sensFOV / 2)]
            )
            center = pose[0:2] + self.sensRange * np.array([np.cos(phi), np.sin(phi)])
            posepos = pose[0:2]
        else:
            left = pose[0:2] + self.sensRange * np.array([1, 1])
            right = pose[0:2] + self.sensRange * np.array([1, -1])
            center = pose[0:2] + self.sensRange * np.array([-1, 1])
            posepos = pose[0:2] + self.sensRange * np.array([-1, -1])
        # Check if in Map Limits
        center = self.get_pos_in_map_lims(center)
        left = self.get_pos_in_map_lims(left)
        right = self.get_pos_in_map_lims(right)

        # Find Cell indices of pose, center, left, right
        limCellsX, limCellsY = np.zeros(4).astype(int), np.zeros(4).astype(int)
        limCellsY[0], limCellsX[0] = self.getCellsFromPose(posepos)
        limCellsY[1], limCellsX[1] = self.getCellsFromPose(center)
        limCellsY[2], limCellsX[2] = self.getCellsFromPose(left)
        limCellsY[3], limCellsX[3] = self.getCellsFromPose(right)

        # Find indices of rectangular map section
        x_idc_start, x_idc_end = (np.min(limCellsX), np.max(limCellsX))
        y_idc_start, y_idc_end = (np.min(limCellsY), np.max(limCellsY))

        c, s = np.cos(phi), np.sin(phi)
        R = np.array(((c, s), (-s, c)))

        # Iterate over map section, check for FOV,range and visibility
        visible_cells = set()
        for i in range(y_idc_start, y_idc_end + 1):
            for j in range(x_idc_start, x_idc_end + 1):
                cellPos = self.getPoseFromCell((i, j))
                r = np.dot(R, np.asarray(cellPos - pose[0:2]))
                dphi = np.arctan2(r[1], r[0])
                r_norm = np.sqrt(r[0] ** 2 + r[1] ** 2)
                if r_norm < self.sensRange and abs(dphi) <= self.sensFOV / 2:
                    if (i, j) in self.free_cells:
                        visible = self.edfMapObj.checkVisibility(pose, cellPos)
                        if visible:
                            visible_cells.add((i, j))

        return visible_cells

    def update(self, poses, observations, frame="global"):
        obsvdCells = set()
        reward = 0
        # Update for all agents observations
        for pose, obs in zip(poses, observations):
            c, s = np.cos(pose[2]), np.sin(pose[2])
            # R_plus = np.array(((c, -s), (s, c)))
            R_minus = np.array(((c, s), (-s, c)))

            n_detected = len(obs)
            detections = []
            for target in obs:
                if frame == "global":
                    ego_pose = np.dot(R_minus, (target - pose[0:2]))
                elif frame == "ego":
                    ego_pose = target
                else:
                    raise Exception("Unsupported Frame for Target Map Update")
                detections.append(ego_pose)
            visibleCells = self.getVisibleCells(pose)
            for i, j in visibleCells:
                if n_detected > 0:
                    cellPos = self.getPoseFromCell((i, j))
                    r = np.dot(R_minus, np.asarray(cellPos - pose[0:2]))
                    # dphi = np.arctan2(r[1], r[0])

                    in_current_cell = False
                    for r_target in detections:
                        r_diff = r_target - r
                        r_diff_norm = np.sqrt(r_diff[0] ** 2 + r_diff[1] ** 2)
                        if r_diff_norm < (
                            np.sqrt(0.5) * self.cellSize + self.tolerance
                        ):
                            in_current_cell = True
                            break

                    if in_current_cell:
                        lSens = self.lOcc
                    else:
                        lSens = self.lEmp
                else:
                    lSens = self.lEmp

                # REWARD computation
                if Config.IG_REWARD_MODE == "binary":
                    reward += (
                        Config.IG_REWARD_BINARY_CELL
                        if not self.binaryMap[i, j]
                        else 0.0
                    )
                else:
                    reward += self.get_reward_from_cells([(i, j)], force_mi=True)

                self.binaryMap[i, j] = True

                self.map[i, j] += lSens
                self.map[i, j] = np.clip(
                    self.map[i, j], -self.logMap_bound, self.logMap_bound
                )

                # Update probabilities

                # p_cell = self.map[j,i] / (self.map[j,i] + 1)
                p_cell = 1 / ((1 / np.exp(self.map[i, j])) + 1)
                self.probMap[i, j] = p_cell

                # Update Entropies and obtain reward
                cell_entropy = (
                    -p_cell * np.log(p_cell) - (1 - p_cell) * np.log(1 - p_cell)
                ) / np.log(2)

                # reward += self.entropyMap[j,i] - cell_entropy
                self.entropy_free_space = (
                    self.entropy_free_space - self.entropyMap[i, j] + cell_entropy
                )
                self.entropyMap[i, j] = cell_entropy

                # Update Goal map and get goal rewards
                if Config.IG_GOALS_ACTIVE:
                    for goal_k in range(len(self.goal_cells)):
                        if (i, j) in self.goal_cells[goal_k]:
                            reward += (
                                Config.IG_REWARD_BINARY_CELL
                                * Config.IG_REWARD_GOAL_CELL_FACTOR
                            )
                            self.goal_map[i, j] -= 1.0
                            self.goal_cells[goal_k].remove((i, j))

            obsvdCells.update(visibleCells)
            self.visitedCells.update(obsvdCells)
            self.visited_share = len(self.visitedCells) / len(self.free_cells)

        self.ego_map = self.create_ego_map(
            poses[0], self.entropyMap, self.ego_map_inner_size
        )
        self.bin_ego_map = self.create_ego_map(
            poses[0], (~self.binaryMap).astype(float), self.ego_map_inner_size
        )
        if Config.IG_GOALS_ACTIVE:
            self.goal_ego_map = self.create_ego_map(
                poses[0],
                self.goal_map.astype(float),
                self.ego_map_inner_size,
            )

        # Check Termination
        if Config.IG_THRES_ACTIVE:

            self.finished_entropy = self.entropy_free_space <= self.thres_entropy
            self.finished_binary = self.visited_share >= self.thres_share_vis_cells

            if Config.IG_REWARD_MODE == "binary":
                self.finished = self.finished_binary
            else:
                self.finished = self.finished_entropy

        if Config.IG_GOALS_ACTIVE:
            # Check for completed goals
            reward += self.check_goal_completion()

            if Config.IG_GOALS_TERMINATION:
                if len(self.current_goals) == 0:
                    self.goal_completion_counter += 1
                else:
                    self.goal_completion_counter = 0

                self.finished = self.finished and (
                    self.goal_completion_counter > Config.IG_GOALS_TERMINATION_WAIT
                )

            # Update Multi-Channel Map
            self.mc_ego_binary_goal = np.stack((self.bin_ego_map, self.goal_ego_map))

        return obsvdCells, reward

    def get_reward_from_cells(self, cells, force_mi=False):
        cell_mi = []
        for i, j in cells:
            if Config.IG_REWARD_MODE == "binary" and not force_mi:
                reward = (
                    Config.IG_REWARD_BINARY_CELL if not self.binaryMap[i, j] else 0.0
                )
            else:
                if np.abs(self.map[i, j]) >= self.logMap_bound:
                    reward = 0.0
                else:
                    r = np.exp(self.map[i, j])
                    p = r / (r + 1)
                    f_p = np.log((r + 1) / (r + (1 / self.rOcc))) - np.log(
                        self.rOcc
                    ) / (r * self.rOcc + 1)
                    f_n = np.log((r + 1) / (r + (1 / self.rEmp))) - np.log(
                        self.rEmp
                    ) / (r * self.rEmp + 1)

                    P_p = p * (1 - self.p_false_neg) + (1 - p) * self.p_false_pos
                    P_n = p * self.p_false_neg + (1 - p) * (1 - self.p_false_pos)

                    reward = P_p * f_p + P_n * f_n

            # Get goal rewards
            for goal_k in range(len(self.goal_cells)):
                if (i, j) in self.goal_cells[goal_k]:
                    reward += (
                        Config.IG_REWARD_BINARY_CELL * Config.IG_REWARD_GOAL_CELL_FACTOR
                    )

            cell_mi.append(reward)
        return sum(cell_mi)

    def get_reward_from_pose(self, pose, force_mi=False):
        visibleCells = self.getVisibleCells(pose)
        return self.get_reward_from_cells(visibleCells, force_mi)

    def update_goal_map(self, new_goal, radius=2):

        self.current_goals.append(new_goal)

        (i, j) = self.getCellsFromPose(new_goal)

        old_goal_map = copy.copy(self.goal_map)

        cv2.circle(self.goal_map, center=(j, i), radius=radius, color=1, thickness=-1)

        rows, cols = np.where(self.goal_map != old_goal_map)

        cells = list(zip(rows.tolist(), cols.tolist()))
        for cell in cells:
            if cell not in self.free_cells:
                cells.remove(cell)

        self.goal_cells.append(cells)

    def check_goal_completion(self):

        reward = 0

        # Check if goal completed
        completed_goal_idc = []
        for goal_idx in range(len(self.goal_cells)):
            if len(self.goal_cells[goal_idx]) == 0:
                completed_goal_idc.append(goal_idx)
            else:
                reward += Config.IG_REWARD_GOAL_PENALTY
        # Save completed goals and delete from buffers
        completed_goals = []
        completed_goal_idc.sort()
        for goal_idx in reversed(completed_goal_idc):
            reward += Config.IG_REWARD_GOAL_COMPLETION
            completed_goals.append(self.current_goals[goal_idx])
            del self.current_goals[goal_idx]
            del self.goal_cells[goal_idx]

        return reward

    def create_ego_map(self, pose, map, newImageWidth, border_value=0.0):

        # # Clip position to be inside map bounds
        # position = np.clip(
        #     pose[:2], -self.mapSize * np.ones(2), self.mapSize * np.ones(2)
        # )

        map_cell = self.getCellsFromPose(pose[:2])
        # map_cell = (map_cell[0], 20-map_cell[1])
        angle = pose[2] * 180 / np.pi

        # Taking image height and width
        imgHeight, imgWidth = map.shape[0], map.shape[1]

        # Computing the centre x,y coordinates
        # of an image
        centreY, centreX = imgHeight // 2, imgWidth // 2

        # Computing 2D rotation Matrix to rotate an image
        rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), 45, 1.0)

        # After computing the new height & width of an image
        # we also need to update the values of rotation matrix
        rotationMatrix[0][2] += (newImageWidth / 2) - centreX
        rotationMatrix[1][2] += (newImageWidth / 2) - centreY

        # Now, we will perform actual image rotation
        # bordervalue = 1.0 if bin else 0.0
        rotatingimage = cv2.warpAffine(
            map,
            rotationMatrix,
            (newImageWidth, newImageWidth),
            borderValue=border_value,
        )

        ext_map = np.zeros((3 * newImageWidth, 3 * newImageWidth), dtype=np.float32)
        ext_map[
            newImageWidth : 2 * newImageWidth, newImageWidth : 2 * newImageWidth
        ] = rotatingimage

        transform_point = cv2.transform(
            np.asarray(np.flip(map_cell)).reshape((1, 1, 2)), rotationMatrix
        )[0][0]

        point = transform_point + np.array([newImageWidth, newImageWidth], dtype=int)

        rot_mat = cv2.getRotationMatrix2D(
            (int(point[0]), int(point[1])), 45 - angle, 1.0
        )
        rot2 = cv2.warpAffine(
            ext_map, rot_mat, ext_map.shape[1::-1], borderValue=border_value
        )

        final = rot2[
            point[1] - newImageWidth : point[1] + newImageWidth,
            point[0] - newImageWidth : point[0] + newImageWidth,
        ]
        # final = cv2.flip(final, 1)
        final_resize = cv2.resize(
            final, Config.EGO_MAP_SIZE, interpolation=cv2.INTER_LINEAR
        )

        # return np.expand_dims((final_resize * 255).astype(np.uint8), axis=0)
        return (final_resize * 255).astype(np.uint8)
