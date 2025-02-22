import numpy as np
import scipy
from gym_collision_avoidance.envs.information_models.edfMap import edfMap


class targetMap():
    def __init__(self, edfMapObj, mapSize, cellSize, sensFOV, sensRange, rOcc, rEmp, tolerance=0.01, prior=1.0,
                 p_false_neg=0.1, p_false_pos=0.05):
        self.edfMapObj = edfMapObj
        
        self.cellSize = cellSize
        self.mapSize = np.asarray(mapSize)
        self.sensFOV = sensFOV
        self.sensRange = sensRange

        self.rOcc = rOcc
        self.rEmp = rEmp
        self.tolerance = tolerance

        self.p_false_neg = p_false_neg
        self.p_false_pos = p_false_pos

        shape = (int(self.mapSize[1]/self.cellSize), int(self.mapSize[0]/self.cellSize))
        self.map = np.ones(shape) * prior

    def getCellsFromPose(self, pose):
        if len(pose) > 2:
            pose = pose[0:2]
        xIdc = np.floor( (pose[0] + self.mapSize[0]/2) / self.cellSize )
        yIdc = np.floor( (pose[1] + self.mapSize[1]/2) / self.cellSize )
        return (xIdc.astype(int), yIdc.astype(int))
    
    def getPoseFromCell(self, cell):
        x = (cell[0])*self.cellSize - self.mapSize[0]/2 + self.cellSize/2
        y = (cell[1])*self.cellSize - self.mapSize[1]/2 + self.cellSize/2
        return np.array([x,y])

    def get_pos_in_map_lims(self,pose):
        if len(pose) > 2:
            pose = pose[0:2]
        return np.max(np.array([np.min(np.array([pose, self.mapSize / 2]), axis=0), -self.mapSize / 2]), axis=0)

    def getVisibleCells(self, pose):
        
        # Robot heading angle
        phi = pose[2]
        ## Get rectangular map section to be updated
        # FOV center, left, right limiting point
        center  = pose[0:2] + self.sensRange * np.array([ np.cos(phi), np.sin(phi) ])
        left    = pose[0:2] + self.sensRange * np.array([ np.cos(phi + self.sensFOV), np.sin(phi + self.sensFOV) ])
        right   = pose[0:2] + self.sensRange * np.array([ np.cos(phi - self.sensFOV), np.sin(phi - self.sensFOV) ])
        # Check if in Map Limits
        center = self.get_pos_in_map_lims(center)
        left = self.get_pos_in_map_lims(left)
        right = self.get_pos_in_map_lims(right)

        # Find Cell indices of pose, center, left, right
        limCellsX, limCellsY = np.zeros(4).astype(int), np.zeros(4).astype(int)
        limCellsX[0], limCellsY[0] = self.getCellsFromPose(pose) 
        limCellsX[1], limCellsY[1] = self.getCellsFromPose(center)
        limCellsX[2], limCellsY[2] = self.getCellsFromPose(left)
        limCellsX[3], limCellsY[3] = self.getCellsFromPose(right)

        # Find indices of rectangular map section
        xIdcStart, xIdcEnd = ( np.min(limCellsX), np.max(limCellsX) )
        yIdcStart, yIdcEnd = ( np.min(limCellsY), np.max(limCellsY) )

        c, s = np.cos(phi), np.sin(phi)
        R = np.array(((c, s), (-s, c)))

        # Iterate over map section, check for FOV,range and visibility
        visibleCells = set()
        for i in range(xIdcStart, xIdcEnd):
            for j in range(yIdcStart, yIdcEnd):
                cellPos = self.getPoseFromCell((i,j))
                r = np.dot(R, np.asarray(cellPos - pose[0:2]))
                dphi = np.arctan2(r[1], r[0])
                r_norm = np.sqrt(r[0]**2 + r[1]**2)
                if r_norm < self.sensRange and abs(dphi) < self.sensFOV/2:
                    visible = self.edfMapObj.checkVisibility(pose, cellPos)
                    if visible:
                        visibleCells.add((i,j))
        
        return visibleCells

    def update(self, poses, observations, frame='global'):
        obsvdCells = set()
        # Update for all agents observations
        for pose, obs in zip(poses, observations):
            c, s = np.cos(pose[2]), np.sin(pose[2])
            # R_plus = np.array(((c, -s), (s, c)))
            R_minus= np.array(((c, s), (-s, c)))

            n_detected = len(obs)
            detections = []
            for target in obs:
                if frame == 'global':
                    ego_pose = np.dot(R_minus,(target - pose[0:2]))
                elif frame == 'ego':
                    ego_pose = target
                else:
                    raise Exception("Unsupported Frame for Target Map Update")
                detections.append(ego_pose)

            visibleCells = self.getVisibleCells(pose)
            for i,j in visibleCells:
                if n_detected > 0:
                    cellPos = self.getPoseFromCell((i,j))
                    r = np.dot(R_minus, np.asarray(cellPos - pose[0:2]))
                    # dphi = np.arctan2(r[1], r[0])

                    in_current_cell = False
                    for r_target in detections:
                        r_diff = r_target - r
                        r_diff_norm = np.sqrt(r_diff[0]**2 + r_diff[1]**2)
                        if r_diff_norm < (np.sqrt(0.5)*self.cellSize + self.tolerance):
                            in_current_cell = True
                            break

                    if in_current_cell:
                        rSens = self.rOcc
                    else:
                        rSens = self.rEmp
                else:
                    rSens = self.rEmp
                self.map[j,i] *= rSens
            obsvdCells.update(visibleCells)
        return obsvdCells

    def get_reward_from_cells(self, cells):
        cell_mi = []
        for i, j in cells:
            r = self.map[j, i]
            p = r / (r + 1)
            f_p = np.log((r + 1) / (r + (1 / self.rOcc))) - np.log(self.rOcc) / (r * self.rOcc + 1)
            f_n = np.log((r + 1) / (r + (1 / self.rEmp))) - np.log(self.rEmp) / (r * self.rEmp + 1)

            P_p = p * (1 - self.p_false_neg) + (1 - p) * self.p_false_pos
            P_n = p * self.p_false_neg + (1 - p) * (1 - self.p_false_pos)

            mi = P_p * f_p + P_n * f_n
            cell_mi.append(mi)
        return sum(cell_mi)

    def get_reward_from_pose(self,pose):
        visibleCells = self.getVisibleCells(pose)
        return self.get_reward_from_cells(visibleCells)