import numpy as np
from scipy import ndimage

class edfMap():
    def __init__(self, obstMap, cellSize, mapSize):
        self.cellSize = cellSize
        self.mapSize = mapSize
        self.map = None
        self.update(obstMap)
    
    def update(self, obstMap):
        self.map = ndimage.distance_transform_edt((~obstMap.map).astype(int)) * self.cellSize
    
    def get_edf_value_from_pose(self,pose):
        if len(pose) > 2:
            pose = pose[0:2]
        xIdc = np.floor((pose[0] + self.mapSize[0] / 2) / self.cellSize)
        yIdc = np.floor((pose[1] + self.mapSize[1] / 2) / self.cellSize)
        return self.map[int(yIdc),int(xIdc)]

    def checkVisibility(self, pose, goal):
        pose = np.asarray(pose) if 'list' in str(type(pose)) else pose
        goal = np.asarray(goal) if 'list' in str(type(goal)) else goal

        if len(pose) > 2:
            pose = pose[0:2]
        if len(goal) > 2:
            goal = goal[0:2]
        
        distIncr = 0.05
        thres = 0.001

        visible = True
        diff = goal - pose
        u = distIncr / np.sqrt(diff[0]**2 + diff[1]**2)

        while u < 1:
            nextPoint = (1-u)*pose + u*goal
            xIdc = int(np.floor( (nextPoint[0] + self.mapSize[0]/2) / self.cellSize ))
            yIdc = int(np.floor( (nextPoint[1] + self.mapSize[1]/2) / self.cellSize ))
            minDist = self.map[yIdc, xIdc]

            if minDist < thres:
                visible = False
                break
            u += minDist / np.sqrt(diff[0]**2 + diff[1]**2)
        return visible