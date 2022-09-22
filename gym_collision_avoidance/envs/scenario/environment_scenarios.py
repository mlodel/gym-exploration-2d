import numpy as np
from gym_collision_avoidance.envs.maps.map_env import EnvMap


def json_map_file(Config, **kwargs):
    json_prefix = (
        "/home/max/Documents/projects/exploration_2d/HouseExpo/HouseExpo/json/"
    )

    # file_name = "0a1b29dba355df2ab02630133187bfab.json"
    file_name = "0a1a5807d65749c1194ce1840354be39.json"
    # file_name = "0a5c77794ab1c44936682ccf4562f3c3.json"

    json_path = json_prefix + file_name

    return None, json_path


def exploration_random_obstacles(
    Config, n_obstacles=0, radius=0.5, rng=None, seed=None
):
    if seed is not None and rng is None:
        rng = np.random.default_rng(seed)
    elif seed is None and rng is None:
        rng = np.random.default_rng(1)

    obstacle = []

    obstacle_5 = [(-9.8, 10), (-10, 10), (-10, -10), (-9.8, -10)]
    obstacle_6 = [(10, 10), (9.8, 10), (9.8, -10), (10, -10)]
    obstacle_7 = [(-10, 9.8), (-10, 10), (10, 10), (10, 9.8)]
    obstacle_8 = [(10, -9.8), (-10, -9.8), (-10, -10), (10, -10)]

    pos_lims_map = Config.MAP_HEIGHT / 2
    obstacle_margin = 4 * radius

    obstacle_np = []
    obstacle_at_wall = False

    while len(obstacle) < n_obstacles:
        obst_width = 1.0 * rng.integers(6, 15)
        obst_height = 1.0 * rng.integers(1, 8)
        obst_heading = 0.5 * np.pi * rng.integers(0, 2)
        # if obstacle_at_wall:
        #     obst_center = (2*pos_lims_margin - obst_width) \
        #                   * np.random.rand(2) - pos_lims_margin + obst_width / 2
        # else:
        obst_center = (
            (2 * pos_lims_map - obst_width / 2) * rng.random(2)
            - pos_lims_map
            + obst_width / 4
        )
        obst_heading = 0.5 * np.pi * rng.integers(0, 2)

        obstacle_dummy = np.array(
            [
                [obst_width / 2, obst_height / 2],
                [-obst_width / 2, obst_height / 2],
                [-obst_width / 2, -obst_height / 2],
                [obst_width / 2, -obst_height / 2],
            ]
        )
        obstacle_shift = obstacle_dummy + (np.ones(obstacle_dummy.shape) * obst_center)
        R = np.array(
            [
                [np.cos(obst_heading), -np.sin(obst_heading)],
                [np.sin(obst_heading), np.cos(obst_heading)],
            ]
        )
        obstacle_rot = np.dot(R, obstacle_shift.transpose()).transpose()
        obstacle_rand = [(p[0], p[1]) for p in list(obstacle_rot)]
        obstacle_rand = (
            [obstacle_rand[(i + 3) % 4] for i in range(4)]
            if obst_heading != 0.0
            else obstacle_rand
        )

        if any(
            [
                0.2
                < pos_lims_map - np.max(np.abs(obstacle_rot[:, i]))
                < obstacle_margin + 0.2
                for i in range(2)
            ]
        ):
            continue
        elif any(
            [0.2 >= pos_lims_map - np.max(np.abs(obstacle_rot[:, i])) for i in range(2)]
        ):
            if obstacle_at_wall:
                continue
            else:
                obstacle_at_wall = True
                obstacle_okay = True
        else:
            obstacle_okay = True

        obstacle_okay = True
        for obst in obstacle_np:
            obstacle_okay = False

            min1, max1 = np.min(obstacle_rot, axis=0), np.max(obstacle_rot, axis=0)
            min2, max2 = np.min(obst, axis=0), np.max(obst, axis=0)
            if (
                (
                    (0 < min1[0] - max2[0] < obstacle_margin)
                    or (0 < min2[0] - max1[0] < obstacle_margin)
                    and (
                        (max2[1] - min1[1] > -obstacle_margin / 2)
                        and (max1[1] - min2[1] > -obstacle_margin / 2)
                    )
                )
                or (
                    (max2[0] - min1[0] > -obstacle_margin / 2)
                    and (max1[0] - min2[0] > -obstacle_margin / 2)
                )
                and (0 < min1[1] - max2[1] < obstacle_margin)
                or (0 < min2[1] - max1[1] < obstacle_margin)
            ):
                break
            else:
                intersecting_area = max(
                    0, min(max1[0], max2[0]) - max(min1[0], min2[0])
                ) * max(0, -max(min1[1], min2[1]) + min(max1[1], max2[1]))
                obstacle_area1 = obst_width * obst_height
                obstacle_area2 = (max2[0] - min2[0]) * (max2[1] - min2[1])
                if (
                    intersecting_area / obstacle_area1 > 0.3
                    or intersecting_area / obstacle_area2 > 0.3
                ):
                    break
                else:
                    obstacle_okay = True

        if obstacle_okay:
            obstacle_np.append(obstacle_rot)
            obstacle.extend([obstacle_rand])

    obstacle.extend([obstacle_5, obstacle_6, obstacle_7, obstacle_8])

    return obstacle, None
