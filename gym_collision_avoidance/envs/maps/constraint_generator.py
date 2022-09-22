import numpy as np
from dataclasses import dataclass


@dataclass
class Ellipse:
    rx: float
    ry: float
    d: np.ndarray
    R: np.ndarray
    alpha: float


class ConstraintGen:
    def __init__(self, resolution, robot_radius, n_constraints):

        self.resolution = resolution
        self.robot_radius = robot_radius
        self.n_constraints = n_constraints

        # init buffers
        self.points = np.array(())
        self.el = Ellipse(rx=0.0, ry=0.0, d=np.array([]), R=np.array([]), alpha=0.0)

    def constraints_from_pointcloud(self, points, pos, goal=None, radius=None):

        if radius is not None:
            self.robot_radius = radius

        self.points = points

        if goal is not None:
            results = self._make_goal_feasible(pos, goal)
            new_goal, new_goal_feas = results
            a, b, vis, distances = self.get_constraints_from_ellipse(pos, new_goal)

        else:
            a, b, vis, distances = self.get_constraints_from_circle(pos)
            new_goal = goal

        # Sort constraints by distance to robot
        sort_idc = np.argsort(distances)
        a_sorted = np.array(a)[sort_idc]
        b_sorted = np.array(b)[sort_idc]

        # Limit number of constraints
        a_sorted = a_sorted[: self.n_constraints]
        b_sorted = b_sorted[: self.n_constraints]

        # Add dummy constraints to fill up to n_constraints
        if a_sorted.shape[0] < self.n_constraints:
            a_sorted = np.vstack(
                (a_sorted, np.ones((self.n_constraints - a_sorted.shape[0], 2)))
            )
            b_sorted = np.hstack(
                (b_sorted, np.ones((self.n_constraints - b_sorted.shape[0])) * 100)
            )

        return a_sorted, b_sorted, new_goal, vis

    def get_constraints_from_ellipse(self, pos, goal):

        # Construct ellipsis around line to goal
        # ... by shrinking circle around line between pos and goal

        # Init ellipse as circle
        self.el.d = pos + 0.5 * (goal - pos)
        self.el.rx = np.linalg.norm(goal - pos) / 2
        self.el.ry = self.el.rx
        self.el.alpha = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])
        self.el.R = np.array(
            [
                [np.cos(self.el.alpha), -np.sin(self.el.alpha)],
                [np.sin(self.el.alpha), np.cos(self.el.alpha)],
            ]
        )

        # Shrink ellipse along orthoginal axis until no obstacle points are inside
        closest_point = self._shrink_ellipse()

        # Store ellipse size for visualization
        el_ax_small = (self.el.rx, self.el.ry)

        # Grow ellipse around robot position, when reaching an obstacle create tangential halfspace constraint
        # For every halfspace constraint remove all obstacle points behind that constraint
        # Loop until all points removed

        # Constraint buffers ax<b
        a = []
        b = []
        distances = []
        closest_points = []

        k = -1
        while self.points.shape[0] > 0:
            k += 1

            # For k=0 constraint can be generated directly at last shrinking point (= closest point)
            # if initial circle (before shrinking) was empty, no closest point is available
            if k > 0 or closest_point is None:
                el_E, el_E_inv = self._el_E_and_inv()
                points_dist = np.linalg.norm(
                    el_E_inv @ (self.points - self.el.d).T, axis=0
                )
                closest_point_idx = np.argmin(points_dist)
                closest_point = self.points[closest_point_idx]
                closest_point_dist = points_dist[closest_point_idx]

                if closest_point[0] < self.el.d[0]:
                    closest_point[0] += self.resolution
                if closest_point[1] > self.el.d[1]:
                    closest_point[1] -= self.resolution

                # Scale ellipse to intersect new closest point
                # transform point to ellipse frame, apply standard ellipse equation
                # to get axis scale factor el_scale
                el_point = self.el.R.T @ (self.points[closest_point_idx] - self.el.d).T
                el_scale = np.sqrt(
                    el_point[0] ** 2 / self.el.rx**2
                    + el_point[1] ** 2 / self.el.ry**2
                )
                self.el.rx *= el_scale
                self.el.ry *= el_scale
            else:
                if closest_point[0] < self.el.d[0]:
                    closest_point[0] += self.resolution
                if closest_point[1] > self.el.d[1]:
                    closest_point[1] -= self.resolution

                closest_point_dist = np.linalg.norm(closest_point - self.el.d)

            # Get tangent equation to create halfspace constraint
            new_a, new_b = self._get_tangent(closest_point, ellipse=True)

            # Shift by radius
            new_b -= new_a.T @ new_a * self.robot_radius

            a.append(new_a)
            b.append(new_b)
            distances.append(closest_point_dist)
            closest_points.append(closest_point)

            # Remove all points in other halfspace, as they can be ignored for remaining constraints
            self._remove_points(new_a, new_b)

        # Store visualization data
        el_ax = (self.el.rx, self.el.ry)
        vis = dict(
            small=dict(axes=el_ax_small, angle=self.el.alpha, center=self.el.d),
            large=dict(axes=el_ax, angle=self.el.alpha, center=self.el.d),
            closest_points=closest_points,
        )

        return a, b, vis, distances

    def get_constraints_from_circle(self, pos):

        # Grow circle around robot position, when reaching an obstacle create tangential halfspace constraint
        # For every halfspace constraint remove all obstacle points behind that constraint
        # Loop until all points removed

        # Init circle around robot position
        self.el.R = np.eye(2)
        self.el.d = pos

        # Constraint buffers ax<b
        a = []
        b = []
        distances = []
        closest_points = []

        k = -1
        while self.points.shape[0] > 0:
            k += 1

            # Compute point-to-center distance list for current set of points
            points_dist = np.linalg.norm(self.points - self.el.d, axis=1)

            # Get closest point
            closest_point_idx = np.argmin(points_dist)
            closest_point = self.points[closest_point_idx]
            closest_point_dist = points_dist[closest_point_idx]

            if closest_point[0] < self.el.d[0]:
                closest_point[0] += self.resolution
            if closest_point[1] > self.el.d[1]:
                closest_point[1] -= self.resolution

            # Set circle radius to closest point (using ellipse container)
            # self.el.rx = points_dist[closest_point_idx]
            self.el.rx = np.linalg.norm(closest_point - self.el.d)

            # Store for visualization
            if k == 0:
                el_ax_small = (self.el.rx, self.el.rx)

            # Get tangent equation to create halfspace constraint
            new_a, new_b = self._get_tangent(closest_point, ellipse=False)

            # Shift by radius
            new_b -= new_a.T @ new_a * self.robot_radius

            a.append(new_a)
            b.append(new_b)
            distances.append(closest_point_dist)
            closest_points.append(closest_point)

            # Remove all points in other halfspace, as they can be ignored for remaining constraints
            self._remove_points(new_a, new_b)

        # Store visualization data
        el_ax = (self.el.rx, self.el.rx)
        vis = dict(
            small=dict(axes=el_ax_small, angle=self.el.alpha, center=self.el.d),
            large=dict(axes=el_ax, angle=self.el.alpha, center=self.el.d),
            closest_points=closest_points,
        )

        return a, b, vis, distances

    def _get_tangent(self, closest_point, ellipse=True):
        if ellipse:
            el_E, el_E_inv = self._el_E_and_inv()
            # el_E_inv = np.linalg.inv(el_E)

            new_a = 2 * el_E_inv @ el_E_inv.T @ (closest_point - self.el.d)

            # Normalize normal vector
            new_a /= np.sqrt(np.sum(new_a**2))

            new_b = new_a.T @ closest_point
        else:
            # Line coefficients are simple normal vector of tangent = vector from center to point
            new_a = closest_point - self.el.d
            # Normalize normal vector
            new_a /= np.sqrt(np.sum(new_a**2))
            # Constant term of line equation
            new_b = new_a.T @ closest_point

        return new_a, new_b

    def _remove_points(self, new_a, new_b):
        remove_points_idc = np.nonzero(
            new_a[0] * self.points[:, 0] + new_a[1] * self.points[:, 1] >= new_b
        )

        self.points = np.delete(self.points, remove_points_idc[0], axis=0)

    def _shrink_ellipse(self):

        # Compute distance list (point to ellipse/circle center)
        points_dist = np.linalg.norm(self.points - self.el.d, axis=1)

        # Extract points inside initial circle
        points_in_circle_idc = np.argwhere(points_dist <= self.el.rx).squeeze()
        points_dist_in_circle = points_dist[points_in_circle_idc]
        points_in_circle = self.points[points_in_circle_idc]

        if points_in_circle.ndim == 1:
            points_in_circle = points_in_circle.reshape(1, -1)
        # Correct points in circle for resolution
        correct_x_idc = np.argwhere(points_in_circle[:, 0] < self.el.d[0])
        points_in_circle[correct_x_idc, 0] = (
            points_in_circle[correct_x_idc, 0] + self.resolution
        )
        correct_y_idc = np.argwhere(points_in_circle[:, 1] > self.el.d[1])
        points_in_circle[correct_y_idc, 1] = (
            points_in_circle[correct_y_idc, 1] - self.resolution
        )

        # Tranform points in circle into (rotated) frame in ellipse center
        points_in_circle_elframe = (self.el.R.T @ (points_in_circle - self.el.d).T).T

        # Sort points for distance to center
        sorted_idc = np.argsort(points_dist_in_circle)
        # points_dist_in_circle = points_dist_in_circle[sorted_idc]
        points_in_circle_elframe = points_in_circle_elframe[sorted_idc]

        # Helper function to check if any ellipse frame point is inside ellipse
        def are_points_in_el(points_elframe):
            return np.any(
                points_elframe[:, 0] ** 2 / self.el.rx**2
                + points_elframe[:, 1] ** 2 / self.el.ry**2
                < 1 - (self.resolution / 2) ** 2
            )

        points_in_el = are_points_in_el(points_in_circle_elframe)
        # Store initial state
        circle_not_empty = points_in_el

        # Shrink ellipse axis orthogonal to goal line to closest point,
        # repeat until no points remaining in ellipse
        j = -1
        while points_in_el:
            j += 1
            # Get next closest point
            closest_point_elframe = points_in_circle_elframe[j]

            # Compute new orthogonal semi-axis for given point
            new_el_b = np.sqrt(
                closest_point_elframe[1] ** 2
                / (1 - (closest_point_elframe[0] / self.el.rx) ** 2)
            )

            # Store semi-axis and update success state only if new value is smaller (we want to shrink)
            if new_el_b < self.el.ry:
                self.el.ry = new_el_b
                points_in_el = are_points_in_el(points_in_circle_elframe)

        # Transform last closest point back to global frame (point that touches the final ellipse)
        if circle_not_empty:
            closest_point = self.el.R @ closest_point_elframe + self.el.d

            return closest_point
        else:
            return None

    def _el_E_and_inv(self):

        el_S = np.array([[self.el.rx, 0.0], [0.0, self.el.ry]])

        el_E = self.el.R @ el_S @ self.el.R.T

        el_E_inv = np.array([[el_E[1, 1], -el_E[0, 1]], [-el_E[1, 0], el_E[0, 0]]])

        return el_E, el_E_inv

    def _make_goal_feasible(self, pos, goal):
        alpha = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])

        # Define region around line to goal
        # Circles with robot radius around pos and goal
        circle_cond1 = (
            np.sum((self.points - pos) ** 2, axis=1) <= self.robot_radius**2
        )
        circle_cond2 = (
            np.sum((self.points - goal) ** 2, axis=1) <= self.robot_radius**2
        )

        # Rectangle between pos and goal with width=2*radius
        # defined by 4 linear equation --> polyhedron

        n1 = pos - goal
        n1 /= np.sqrt(np.sum(n1**2))
        n2 = -n1
        n3 = np.array([-n1[1], n1[0]])
        n4 = -n3

        p1 = pos
        p2 = goal
        p3 = pos + self.robot_radius * n3
        p4 = pos + self.robot_radius * n4

        b1 = n1.T @ p1
        b2 = n2.T @ p2
        b3 = n3.T @ p3
        b4 = n4.T @ p4

        A = np.array([n1, n2, n3, n4])
        b = np.array([b1, b2, b3, b4])

        rect_cond = np.all(self.points @ A.T <= b.T, axis=1)

        points_in_region_idc = np.argwhere(circle_cond1 | circle_cond2 | rect_cond)

        if points_in_region_idc.shape[0] > 0:

            points_in_region = self.points[points_in_region_idc]

            closest_point_idc = np.argmin(np.sum((points_in_region - pos) ** 2, axis=2))
            pc = points_in_region[closest_point_idc].squeeze()

            # Unit vector along line position to goal
            v = n2
            # Vector position to closest obstacle point
            u = pc - pos
            # Project closest obstacle point on line -> vector from pos to projected point
            u1 = np.dot(u, v) * v
            # Get vector from projected point to closest obstacle point
            u2 = u - u1
            # Get length of u2
            u2_norm = np.linalg.norm(u2)
            # get distance from projected point to intersection point with pythagoras
            m = np.sqrt(self.robot_radius**2 - u2_norm**2)

            # Get intersection point closer to position
            point = pos + u1 - m * v

            return pos + u1, point

        else:
            return goal, goal
