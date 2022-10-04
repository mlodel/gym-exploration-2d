import numpy as np
import cv2
import json
import os
import copy
import time
import random as rng

import cProfile
import pstats

ALLOWED_ROOM_TYPES = {
    "indoor": 10,
    "kitchen": 30,
    "dining_room": 60,
    "living_room": 90,
    "bathroom": 120,
    "bedroom": 150,
    "office": 180,
    "hallway": 210,
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
    file_name = "0a5c77794ab1c44936682ccf4562f3c3.json"
    # file_name = "0a7d100165ef451e2ab508a227e4c403.json"
    # file_name = "0a1b29dba355df2ab02630133187bfab.json"
    # file_name = "c3458b174d847985d5179c5b46d28da1.json"

    meter2pixel = 50
    border_pad = 0

    with open(json_prefix + file_name.split(".")[0] + ".json") as json_file:
        json_data = json.load(json_file)

    # Draw the contour
    verts = (np.array(json_data["verts"]) * meter2pixel).astype(int)
    x_max, x_min, y_max, y_min = (
        np.max(verts[:, 0]),
        np.min(verts[:, 0]),
        np.max(verts[:, 1]),
        np.min(verts[:, 1]),
    )
    cnt_map = np.zeros(
        (y_max - y_min + border_pad * 2, x_max - x_min + border_pad * 2), dtype=np.uint8
    )
    # shape = (y_max - y_min + border_pad * 2, x_max - x_min + border_pad * 2)
    # cnt_map = np.zeros((200, 200), dtype=np.uint8)

    verts[:, 0] = verts[:, 0] - x_min + border_pad
    verts[:, 1] = verts[:, 1] - y_min + border_pad

    # ratio = cnt_map.shape[0] / max(shape)
    # verts = (verts * ratio).astype(int)

    cv2.drawContours(cnt_map, [verts], 0, 255, -1)
    cnt_map = cv2.bitwise_not(cnt_map)

    cv2.namedWindow("map", cv2.WINDOW_NORMAL)

    cv2.imshow("map", cnt_map)
    cv2.waitKey(0)

    # pos = (150, 300)
    # pos = (140, 180)
    # pos = (270, 270)
    # lookahead = 3 * meter2pixel
    #
    # cnt_map_border = cv2.copyMakeBorder(
    #     cnt_map,
    #     lookahead,
    #     lookahead,
    #     lookahead,
    #     lookahead,
    #     borderType=cv2.BORDER_CONSTANT,
    #     value=255,
    # )
    #
    # profiler = cProfile.Profile()
    # profiler.enable()
    #
    # submap = cnt_map_border[
    #     pos[1] : pos[1] + 2 * lookahead, pos[0] : pos[0] + 2 * lookahead
    # ]
    #
    # submap = cv2.copyMakeBorder(submap, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    #
    # kernel_q = np.ones((3, 3), np.uint8)
    # submap_q = cv2.erode(submap, kernel_q, iterations=3)
    #
    # mask = cv2.bitwise_not(submap_q)
    # submap = cv2.bitwise_and(submap, submap, mask=mask)
    #
    # kernel_h = np.ones((3, 1), np.uint8)
    # submap_h = cv2.erode(submap, kernel_h, iterations=10)
    #
    # kernel_v = np.ones((1, 3), np.uint8)
    # submap_v = cv2.erode(submap, kernel_v, iterations=10)
    #
    # kernel_d = np.eye(5, dtype=np.uint8)
    # submap_d = cv2.erode(submap, kernel_d, iterations=3)
    #
    # kernel_d2 = np.eye(5, dtype=np.uint8)[::-1]
    # submap_d2 = cv2.erode(submap, kernel_d2, iterations=3)
    #
    # submap_or = cv2.bitwise_or(submap_v, submap_h, mask=mask)
    # submap_or = cv2.bitwise_or(submap_or, submap_d)
    # submap_or = cv2.bitwise_or(submap_or, submap_d2)
    #
    # dist = submap_or
    #
    # contours, _ = cv2.findContours(dist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Create the marker image for the watershed algorithm
    # markers = np.zeros(dist.shape, dtype=np.int32)
    # # Draw the foreground markers
    # for i in range(len(contours)):
    #     cv2.drawContours(markers, contours, i, (i + 1), -1)
    #
    # # Draw the background marker
    # cv2.circle(markers, pos, 3, 255, -1)
    # markers_8u = (markers * 10).astype("uint8")
    #
    # color_map = cv2.cvtColor(submap, cv2.COLOR_GRAY2BGR)
    #
    # cv2.watershed(color_map, markers)
    #
    # mark = markers.astype("uint8")
    # mark = cv2.bitwise_not(mark)
    #
    # cv2.imshow("map", submap)
    # cv2.imshow("map_d", submap_d)
    # cv2.imshow("map_d2", submap_d2)
    # cv2.imshow("map_or", submap_or)
    # cv2.imshow("Markers", markers_8u)
    # cv2.imshow("Markers_v2", mark)
    # cv2.waitKey(0)
    #
    # kernel_q = np.ones((2, 2), np.uint8)
    # mark = cv2.erode(mark, kernel_q, iterations=1)
    #
    # # cv2.imshow("Markers_v3", mark)
    # # cv2.waitKey(0)
    #
    # contours, hierarchy = cv2.findContours(mark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Generate random colors
    # colors = []
    # rng.seed(12345)
    # for contour in contours:
    #     colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
    #
    # # Create the result image
    # dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    # dst_hulls = np.zeros_like(dst)
    # dst_rect = np.zeros_like(dst)
    # # # Fill labeled objects with random colors
    # # for i in range(markers.shape[0]):
    # #     for j in range(markers.shape[1]):
    # #         index = markers[i, j]
    # #         if index > 0 and index <= len(contours):
    # #             dst[i, j, :] = colors[index - 1]
    # # # Visualize the final image
    # # cv2.imshow("Final Result", dst)
    # # cv2.waitKey(0)
    #
    # for i, cnt in enumerate(contours):
    #     dst = cv2.drawContours(dst, [cnt], 0, color=colors[i], thickness=-1)
    #     # print(str(i) + ": " + str(cv2.isContourConvex(cnt)))
    #     #
    #     hull = cv2.convexHull(cnt, returnPoints=True)
    #     dst_hulls = cv2.drawContours(
    #         dst_hulls, [hull], -1, color=colors[i], thickness=-1
    #     )
    #
    #     rect = cv2.minAreaRect(cnt)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(dst_rect, [box], 0, color=colors[i], thickness=-1)
    #
    #     # defects = cv2.convexityDefects(cnt, hull)
    #     # if defects is not None:
    #     #     for i in range(defects.shape[0]):
    #     #         s, e, f, d = defects[i, 0]
    #     #         start = tuple(cnt[s][0])
    #     #         end = tuple(cnt[e][0])
    #     #         far = tuple(cnt[f][0])
    #     #         cv2.line(color_map2, start, end, [0, 255, 0], 2)
    #     #         cv2.circle(color_map2, far, 5, [255, 0, 0], -1)
    #
    #     # cv2.imshow("colormap", color_map2)
    #     # cv2.waitKey(0)
    #
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats(os.path.dirname(os.path.realpath(__file__)) + "/stats.prof")
    # # Visualize the final image
    # cv2.imshow("shapes", dst)
    # cv2.imshow("hulls", dst_hulls)
    # cv2.imshow("rect", dst_rect)
    # cv2.waitKey(0)

    # shape = cnt_map.shape
    # target_size = shape[1] // 10, shape[0] // 10
    #
    # scaled_map = cv2.resize(cnt_map, dsize=target_size, interpolation=cv2.INTER_NEAREST)

    # Merge the tps into an allowed subset
    tp_map = np.ones_like(cnt_map, dtype=np.uint8)
    for tp in json_data["room_category"]:
        tp_id = _get_room_tp_id(tp)
        for bbox_tp in json_data["room_category"][tp]:
            bbox_tp = (np.array(bbox_tp) * meter2pixel).astype(np.int)
            bbox = [
                np.max([bbox_tp[0] - x_min + border_pad, 0]),
                np.max([bbox_tp[1] - y_min + border_pad, 0]),
                np.min([bbox_tp[2] - x_min + border_pad, cnt_map.shape[1]]),
                np.min([bbox_tp[3] - y_min + border_pad, cnt_map.shape[0]]),
            ]
            tp_map[bbox[1] : bbox[3], bbox[0] : bbox[2]] |= tp_id
    # tp_map[cnt_map == 1] = 0
    cnt_map = cv2.bitwise_not(cnt_map)
    tp_map = cv2.bitwise_and(tp_map, cnt_map)

    # display(tp_map, tp_flag=True)
    imC = cv2.applyColorMap(tp_map, cv2.COLORMAP_JET)
    cv2.imshow("img", imC)
    cv2.waitKey(0)
