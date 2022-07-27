import numpy as np
import cv2

from abc import ABC, abstractmethod


class ObstacleBase(ABC):
    def __init__(self, pos, orient, length, width) -> None:
        self.length = length
        self.width = width

        self.pos = None
        self.orient = None
        self.update(pos, orient)

    def _update(self):
        pass

    def update(self, pos, orient):
        self.pos = pos
        self.orient = orient
        self._update()

    @abstractmethod
    def draw_obstacle(self, img, res, map_size):
        raise NotImplementedError


class RectangleObst(ObstacleBase):
    def __init__(self, pos, orient, length, width) -> None:
        super().__init__(pos, orient, length, width)

        # self.points = self._get_points()

    def _update(self):
        x = self.pos
        a = self.orient
        l = self.length
        w = self.width

        s = np.sin(a)
        c = np.cos(a)

        A = x + l / 2 * np.array([c, s]) + w / 2 * np.array([-s, c])
        D = x + l / 2 * np.array([c, s]) - w / 2 * np.array([-s, c])
        C = x - l / 2 * np.array([c, s]) - w / 2 * np.array([-s, c])
        B = x - l / 2 * np.array([c, s]) + w / 2 * np.array([-s, c])
        self.points = [A, B, C, D]

    def draw_obstacle(self, img, res, map_size):
        top_left_x = np.around(self.points[1][0] * res + map_size[0] / 2).astype(int)
        top_left_y = np.around(-self.points[1][1] * res + map_size[1] / 2).astype(int)

        bottom_right_x = np.around(self.points[3][0] * res + map_size[0] / 2).astype(
            int
        )
        bottom_right_y = np.around(-self.points[3][1] * res + map_size[1] / 2).astype(
            int
        )
        return cv2.rectangle(
            img,
            pt1=(top_left_x, top_left_y),
            pt2=(bottom_right_x, bottom_right_y),
            color=1,
            thickness=-1,
        )


class CircleObst(ObstacleBase):
    def __init__(self, pos, orient, radius) -> None:
        super().__init__(pos, orient, radius, radius)

        self.radius = radius

    def draw_obstacle(self, img, res, map_size):
        coordinate_x = np.around(self.pos[0] * res + map_size[0] / 2).astype(int)
        coordinate_y = np.around(-self.pos[1] * res + map_size[1] / 2).astype(int)

        return cv2.circle(
            img,
            tuple((coordinate_x, coordinate_y)),
            int(np.around(self.radius * res)),
            1,
            -1,
        )


def get_obstacle_class(obst_config):
    if obst_config["shape"] == "circle":
        return CircleObst
    elif obst_config["shape"] == "rect":
        return RectangleObst
    else:
        raise NotImplementedError
