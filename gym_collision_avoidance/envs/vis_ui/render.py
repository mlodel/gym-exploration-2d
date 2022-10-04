import cv2
import numpy as np
import pypoman
from gym_collision_avoidance.envs.maps.map_env import EnvMap
import time
import os
from gym_collision_avoidance.envs.utils.nested_attr import rgetattr

plt_colors = []
plt_colors.append([0.8500, 0.3250, 0.0980])  # orange
plt_colors.append([0.0, 0.4470, 0.7410])  # blue
plt_colors.append([0.4660, 0.6740, 0.1880])  # green
plt_colors.append([0.4940, 0.1840, 0.5560])  # purple
plt_colors.append([0.9290, 0.6940, 0.1250])  # yellow
plt_colors.append([0.3010, 0.7450, 0.9330])  # cyan
plt_colors.append([0.6350, 0.0780, 0.1840])  # chocolate
plt_colors.append([0.8, 0.0, 0.80])  # magenta
plt_colors.append([0.62, 0.62, 0.62])  # grey


class GymRenderer:
    def __init__(
        self,
        path: str,
        map_size: tuple,
        obstacles: list,
        map_file: str,
        map_min_shape: int = 500,
        map_border_width: int = 5,
        small_map_size: int = 168,
        res_factor=4,
    ):
        # Settings
        self.path = path
        self.margins = 50
        self.map_border_width = map_border_width
        self.small_map_size = small_map_size
        self.res_factor = res_factor

        self.colors = np.array(plt_colors)
        self.colors = np.around(self.colors * 255).astype(np.uint8)
        self.colors = self.colors[:, ::-1]

        self.obs_keys = ["ego_explored_map", "binary_map"]

        # Init Map
        cellsize = min(map_size) / (res_factor * map_min_shape)
        self.res = int(np.around(1 / cellsize))
        self.render_map = EnvMap(
            map_size=map_size,
            cell_size=cellsize,
            obs_size=(84, 84),
            submap_lookahead=3.0,
            json=map_file,
            obstacles_vert=obstacles,
        )

        # Image buffer
        self.frame_buffer = []

        # Video saving flag
        self.is_storing_video = False

    def init_path(self):
        pass

    def draw_frame(self, agents):

        self.render_map.update(
            pose=np.append(agents[0].pos_global_frame, agents[0].heading_global_frame)
        )

        img = cv2.bitwise_not((self.render_map.map * 255).astype(np.uint8))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = self._draw_agent(agents[0], img)

        img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0, sigmaY=0)

        downsample = (img.shape[1] // self.res_factor, img.shape[0] // self.res_factor)
        img = cv2.resize(img, dsize=downsample, interpolation=cv2.INTER_AREA)

        img = cv2.copyMakeBorder(
            img, *((self.map_border_width,) * 4), cv2.BORDER_CONSTANT, (0, 0, 0)
        )

        map_shape = img.shape

        max_width = self.small_map_size
        small_maps = []
        for obs in self.obs_keys:
            small_maps.append(self._get_small_map(agents[0], obs))
            max_width = max(max_width, small_maps[-1].shape[1])

        canvas_shape = (
            map_shape[0] + 2 * self.margins,
            map_shape[1] + 3 * self.margins + max_width,
            3,
        )
        canvas = np.ones(canvas_shape, dtype=np.uint8) * 255

        canvas[
            self.margins : self.margins + map_shape[0],
            self.margins : self.margins + map_shape[1],
            :,
        ] = img

        prev_height = 0
        for i, small_map in enumerate(small_maps):
            canvas[
                (2 * i + 1) * self.margins
                + prev_height : (2 * i + 1) * self.margins
                + prev_height
                + small_map.shape[0],
                2 * self.margins
                + map_shape[1] : 2 * self.margins
                + map_shape[1]
                + small_map.shape[1],
                :,
            ] = small_map
            prev_height = small_map.shape[0]

        self.frame_buffer.append(canvas)

    def get_last_frame(self):
        return self.frame_buffer[-1]

    def save_frame(self):
        pass

    def save_video(self, fps: int = 10):
        cap = cv2.VideoCapture(0)
        writer = cv2.VideoWriter(
            os.path.join(self.path, "filename.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            self.get_last_frame().shape[:2],
        )

        if writer.isOpened():
            for frame in self.frame_buffer:
                writer.write(frame)

        cap.release()
        writer.release()

    def display(self, fps: int = 1000):
        cv2.imshow("gym_display", self.get_last_frame())
        cv2.waitKey(1000 // fps)
        test = 1

    def image_coord_2_map_pos(self, coord: tuple):

        # Get map coordinates from image coordinates
        map_x = (coord[0] - self.margins - self.map_border_width) * self.res_factor
        map_y = (coord[1] - self.margins - self.map_border_width) * self.res_factor

        if (
            map_x < 0
            or map_y < 0
            or map_x > self.render_map.shape[1] - 1
            or map_y > self.render_map.shape[0] - 1
        ):
            print("Goal Interface: Given goal is outside the map!")
            return None
        else:
            pos = self.render_map.get_pos_from_idc((map_y, map_x))
            return pos

    def _draw_agent(self, agent, img):
        agent_pos = agent.pos_global_frame
        agent_orientation = agent.heading_ego_frame

        if hasattr(agent.policy, "policy_goal"):
            subgoal = agent.policy.goal_
            subgoal2 = agent.policy.policy_goal
        else:
            subgoal = None
            subgoal2 = None

        if hasattr(agent.policy, "predicted_traj"):
            trajectory = agent.policy.predicted_traj
        else:
            trajectory = None

        path_history = agent.global_state_history[: agent.step_num - 1, 1:3]

        current_goals = rgetattr(agent, "ig_model.targetMap.current_goals")
        goal_radius = rgetattr(agent, "ig_model.goal_radius")
        goal_radius = goal_radius if goal_radius is not None else 2.0

        # Draw Agent
        agent_cell = self.render_map.get_idc_from_pos(agent_pos)[::-1]
        ## Draw Filled Circle
        img = cv2.circle(
            img,
            center=agent_cell,
            radius=np.around(agent.radius * self.res).astype(int) - self.res_factor,
            thickness=-1,
            color=(self.colors[1, :] + (255 - self.colors[1, :]) // 2).tolist(),
            lineType=cv2.LINE_AA,
        )
        ## Draw Outer Circle
        img = cv2.circle(
            img,
            center=agent_cell,
            radius=np.around(agent.radius * self.res).astype(int) - self.res_factor,
            thickness=self.res_factor,
            color=self.colors[1, :].tolist(),
            lineType=cv2.LINE_AA,
        )
        # Draw Orientation
        d = np.around(
            (agent.radius * self.res)
            * np.array([np.cos(agent_orientation), np.sin(-agent_orientation)])
        ).astype(int)
        p2 = (agent_cell[0] + d[0], agent_cell[1] + d[1])
        img = cv2.line(
            img,
            pt1=agent_cell,
            pt2=p2,
            color=self.colors[1, :].tolist(),
            thickness=self.res_factor,
            lineType=cv2.LINE_AA,
        )

        # Draw Subgoal
        if subgoal is not None:
            subgoal_cell = self.render_map.get_idc_from_pos(subgoal)[::-1]
            ## Draw Filled Circle
            img = cv2.circle(
                img,
                center=subgoal_cell,
                radius=np.around(agent.radius * self.res).astype(int),
                thickness=-1,
                color=(self.colors[1, :] + (255 - self.colors[1, :]) // 1.5).tolist(),
                lineType=cv2.LINE_AA,
            )

        # Draw Feasible Subgoal
        if subgoal2 is not None:
            subgoal_cell = self.render_map.get_idc_from_pos(subgoal2)[::-1]
            ## Draw Filled Circle
            img = cv2.circle(
                img,
                center=subgoal_cell,
                radius=np.around(agent.radius * self.res * 0.3).astype(int),
                thickness=-1,
                color=(self.colors[1, :] + (255 - self.colors[1, :]) // 3).tolist(),
                lineType=cv2.LINE_AA,
            )

        # Draw MPC plan
        if trajectory is not None:
            trajectory_cells = np.array(
                [self.render_map.get_idc_from_pos(pos)[::-1] for pos in trajectory]
            )
            img = cv2.polylines(
                img,
                pts=[trajectory_cells],
                color=(self.colors[1, :] + (255 - self.colors[1, :]) // 3).tolist(),
                isClosed=False,
                lineType=cv2.LINE_AA,
                thickness=self.res_factor,
            )

        # Draw path history
        if path_history is not None:
            path_cells = np.array(
                [self.render_map.get_idc_from_pos(pos)[::-1] for pos in path_history]
            )
            img = cv2.polylines(
                img,
                pts=[path_cells],
                color=(self.colors[1, :] + (255 - self.colors[1, :]) // 4).tolist(),
                isClosed=False,
                lineType=cv2.LINE_AA,
                thickness=self.res_factor,
            )

        # Draw intermediate goals
        if current_goals is not None:
            for goal in current_goals:
                # Draw Agent
                goal_cell = self.render_map.get_idc_from_pos(goal)[::-1]

                ## Draw Filled Circle with transparency
                overlay = np.zeros_like(img, np.uint8)
                alpha = 0.5
                overlay = cv2.circle(
                    overlay,
                    center=goal_cell,
                    radius=np.around(goal_radius * self.res).astype(int),
                    thickness=-1,
                    color=(self.colors[2, :]).tolist(),
                    lineType=cv2.LINE_AA,
                )
                ret, mask = cv2.threshold(overlay[:, :, 0], 1, 255, cv2.THRESH_BINARY)
                mask = mask.astype(bool)
                img[mask] = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0)[mask]

                ## Draw Outer Circle
                img = cv2.circle(
                    img,
                    center=goal_cell,
                    radius=np.around(goal_radius * self.res).astype(int),
                    thickness=self.res_factor,
                    color=self.colors[2, :].tolist(),
                    lineType=cv2.LINE_AA,
                )

        # Draw constraints
        img = self._draw_constraints(img, agent)

        return img

    def _get_small_map(self, agent, obs):
        small_map = agent.get_observation(obs).squeeze()
        max_val = np.max(small_map)
        if max_val > 0 and max_val <= 1:
            small_map = small_map * 255
        small_map = small_map.astype(np.uint8)

        # Make sure the map is square
        if small_map.shape[0] != small_map.shape[1]:
            if small_map.shape[0] > small_map.shape[1]:
                pad = small_map.shape[0] - small_map.shape[1]
                small_map = np.pad(
                    small_map, ((0, 0), (pad // 2, pad - pad // 2)), "constant"
                )
            else:
                pad = small_map.shape[1] - small_map.shape[0]
                small_map = np.pad(
                    small_map, ((pad // 2, pad - pad // 2), (0, 0)), "constant"
                )

        small_map = cv2.cvtColor(small_map, cv2.COLOR_GRAY2BGR)
        small_map = cv2.resize(
            small_map,
            dsize=(self.small_map_size, self.small_map_size),
            fx=0,
            fy=0,
            interpolation=cv2.INTER_LINEAR,
        )
        small_map = cv2.circle(
            small_map,
            center=(self.small_map_size // 2, self.small_map_size // 2),
            radius=3,
            color=self.colors[1, :].tolist(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        small_map = cv2.copyMakeBorder(
            small_map, 3, 3, 3, 3, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        return small_map

    def _draw_constraints(self, img, agent):

        constraints = agent.policy.linear_constraints

        if "a" in constraints:
            A, b = constraints["a"], constraints["b"]
            A = np.array(A)
            b = np.array(b)

            vert = pypoman.compute_polytope_vertices(A, b)
            pts = np.array([(self.render_map.get_idc_from_pos(pt)) for pt in vert])
            convex = cv2.convexHull(pts).squeeze()
            pts_sorted = convex[np.newaxis, :, [1, 0]]
            img = cv2.drawContours(
                img,
                pts_sorted,
                -1,
                color=self.colors[3, :].tolist(),
                thickness=self.res_factor,
            )

            # Display closest points
            constr_visualization = agent.policy.constr_visualization
            # closest_points.append(np.zeros(2))
            if (
                constr_visualization is not None
                and "closest_points" in constr_visualization
            ):
                closest_points = constr_visualization["closest_points"]
                for pt in closest_points:
                    pt_px = self.render_map.get_idc_from_pos(pt)[::-1]
                    img = cv2.circle(
                        img,
                        center=pt_px,
                        radius=self.res_factor * 3,
                        color=self.colors[5, :].tolist(),
                        thickness=-1,
                    )

        return img
