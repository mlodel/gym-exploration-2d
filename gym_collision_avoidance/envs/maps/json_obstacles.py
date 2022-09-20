import numpy as np
import cv2
import json
import os
import copy
import time
import random as rng

import cProfile
import pstats
import timeit
import pypoman

from constraint_generator import ConstraintGen
from map_env import EnvMap

ALLOWED_ROOM_TYPES = {
    "indoor": 1,
    "kitchen": 2,
    "dining_room": 4,
    "living_room": 8,
    "bathroom": 16,
    "bedroom": 32,
    "office": 64,
    "hallway": 128,
}


def _get_room_tp_id(room):
    room = room.lower()
    if room == "toilet":
        room = "bathroom"
    elif room == "guest_room":
        room = "bedroom"
    if room not in ALLOWED_ROOM_TYPES:
        return ALLOWED_ROOM_TYPES["indoor"]
    return ALLOWED_ROOM_TYPES[room]


if __name__ == "__main__":
    json_prefix = (
        "/home/max/Documents/projects/exploration_2d/HouseExpo/HouseExpo/json/"
    )
    # file_name = "0a5c77794ab1c44936682ccf4562f3c3.json"
    # file_name = "0a7d100165ef451e2ab508a227e4c403.json"
    file_name = "0a1b29dba355df2ab02630133187bfab.json"

    map = EnvMap(
        map_size=(20, 20),
        submap_size=(40, 40),
        cell_size=0.05,
        obs_size=(80, 80),
        json=json_prefix + file_name,
    )

    # cv2.imshow("map", map.map * 255)
    # cv2.waitKey(0)
    radius = 0.2
    obj = ConstraintGen(resolution=map.cell_size, robot_radius=radius)

    pos = np.array([0.0, 0.0])
    goal = np.array([0.0, 1.5])
    lookahead = 2

    def to_time():
        points, submap = map.get_local_pointcloud(pos, lookahead)
        results = obj.constraints_from_pointcloud(points, pos, goal)
        return submap, results

    print(min(timeit.Timer(to_time).repeat(repeat=3, number=1000)) / 1000)

    profiler = cProfile.Profile()
    profiler.enable()

    submap, results = to_time()
    constr_res, goal_res = results
    a, b, vis_el, closest_points = constr_res

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats(os.path.dirname(os.path.realpath(__file__)) + "/stats.prof")

    print(a)
    print(b)

    color_map = cv2.cvtColor(map.map * 255, cv2.COLOR_GRAY2BGR)
    factor = 4
    color_map = cv2.resize(
        color_map,
        (color_map.shape[1] * factor, color_map.shape[0] * factor),
        interpolation=cv2.INTER_NEAREST,
    )
    pos_map = map.get_idc_from_pos(pos)
    goal_map = map.get_idc_from_pos(goal)

    color_map = cv2.arrowedLine(
        color_map,
        pt1=(pos_map[1] * factor, pos_map[0] * factor),
        pt2=(goal_map[1] * factor, goal_map[0] * factor),
        color=(255, 0, 0),
        thickness=1,
    )

    color_map = cv2.circle(
        color_map,
        center=(pos_map[1] * factor, pos_map[0] * factor),
        radius=int(radius * factor / map.cell_size),
        color=(255, 0, 0),
        thickness=1,
    )

    center = map.get_idc_from_pos(vis_el["small"]["center"])
    color_map = cv2.ellipse(
        color_map,
        center=(center[1] * factor, center[0] * factor),
        axes=(
            int(vis_el["small"]["axes"][0] * factor / map.cell_size),
            int(vis_el["small"]["axes"][1] * factor / map.cell_size),
        ),
        angle=-vis_el["small"]["angle"] * 180 / np.pi,
        color=(0, 0, 255),
        thickness=1,
        startAngle=0,
        endAngle=360,
    )
    color_map = cv2.ellipse(
        color_map,
        center=(center[1] * factor, center[0] * factor),
        axes=(
            int(vis_el["large"]["axes"][0] * factor / map.cell_size),
            int(vis_el["large"]["axes"][1] * factor / map.cell_size),
        ),
        angle=-vis_el["large"]["angle"] * 180 / np.pi,
        color=(0, 0, 180),
        thickness=1,
        startAngle=0,
        endAngle=360,
    )

    """
    for i in range(len(a)):
        new_a = a[i]
        new_b = b[i]

        pt1 = (
            (
                (int(new_b / new_a[0]), 0)
                if new_b / new_a[0] >= 0
                else (
                    int((new_b - new_a[1] * color_map.shape[0]) / new_a[0]),
                    color_map.shape[0],
                )
            )
            if np.abs(new_a[0]) > 1e-6
            else (color_map.shape[1], int(new_b / new_a[1]))
        )
        pt2 = (
            (
                (0, int(new_b / new_a[1]))
                if new_b / new_a[1] >= 0
                else (
                    color_map.shape[1],
                    int((new_b - new_a[0] * color_map.shape[1]) / new_a[1]),
                )
            )
            if np.abs(new_a[1]) > 1e-6
            else (int(new_b / new_a[0]), color_map.shape[0])
        )

        pt1 = map.get_idc_from_pos(pt1)
        pt2 = map.get_idc_from_pos(pt2)

        color_map = cv2.line(
            color_map,
            pt1=pt1,
            pt2=pt2,
            color=(0, 255, 0),
            thickness=1,
        )
    """

    A = np.stack(a)
    B = np.stack(b)
    vert = pypoman.compute_polytope_vertices(A, B)
    pts = np.array([(map.get_idc_from_pos(pt)) for pt in vert]) * factor

    convex = cv2.convexHull(pts).squeeze()

    # mean = np.mean(pts, axis=0)
    # r = np.linalg.norm(pts - mean)
    # angles = np.where(
    #     (pts[:, 1] - mean[1]) > 0,
    #     np.arccos((pts[:, 0] - mean[0]) / r),
    #     2 * np.pi - np.arccos((pts[:, 0] - mean[0]) / r),
    # )
    # mask = np.argsort(angles)
    # pts_sorted = pts[np.newaxis, mask, :]
    pts_sorted = convex[np.newaxis, :, [1, 0]]
    color_map = cv2.drawContours(
        color_map, pts_sorted, -1, color=(0, 255, 0), thickness=1
    )

    for pt in closest_points:
        pt_px = map.get_idc_from_pos(pt)
        color_map = cv2.circle(
            color_map,
            center=(pt_px[1] * factor, pt_px[0] * factor),
            radius=3,
            color=(255, 100, 100),
            thickness=-1,
        )

    pc, point = goal_res
    pc_px = map.get_idc_from_pos(pc)
    point_px = map.get_idc_from_pos(point)

    color_map = cv2.circle(
        color_map,
        center=(pc_px[1] * factor, pc_px[0] * factor),
        radius=3,
        color=(255, 0, 255),
        thickness=-1,
    )
    color_map = cv2.circle(
        color_map,
        center=(point_px[1] * factor, point_px[0] * factor),
        radius=3,
        color=(255, 255, 0),
        thickness=-1,
    )

    cv2.imshow("submap", color_map)
    cv2.waitKey(0)
    # color_map2 = cv2.flip(color_map, 0)
    # color_map2 = cv2.resize(color_map2, dsize=(600, 600))
    # cv2.imshow("submap2", color_map2)
    # cv2.waitKey(0)
