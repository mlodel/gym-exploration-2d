import numpy as np
import cv2
import json
import os

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
    file_name = "0a1b29dba355df2ab02630133187bfab.json"

    meter2pixel = 10
    border_pad = 1

    with open(json_prefix + file_name.split(".")[0] + ".json") as json_file:
        json_data = json.load(json_file)

    # Draw the contour
    verts = (np.array(json_data["verts"]) * meter2pixel).astype(np.int)
    x_max, x_min, y_max, y_min = (
        np.max(verts[:, 0]),
        np.min(verts[:, 0]),
        np.max(verts[:, 1]),
        np.min(verts[:, 1]),
    )
    cnt_map = np.zeros((y_max - y_min + border_pad * 2, x_max - x_min + border_pad * 2))

    verts[:, 0] = verts[:, 0] - x_min + border_pad
    verts[:, 1] = verts[:, 1] - y_min + border_pad
    cv2.drawContours(cnt_map, [verts], 0, 255, 1)

    # shape = cnt_map.shape
    # target_size = shape[1] // 10, shape[0] // 10
    #
    # scaled_map = cv2.resize(cnt_map, dsize=target_size, interpolation=cv2.INTER_NEAREST)

    # # Merge the tps into an allowed subset
    # tp_map = np.ones_like(cnt_map, dtype=np.uint8)
    # for tp in json_data["room_category"]:
    #     tp_id = _get_room_tp_id(tp)
    #     for bbox_tp in json_data["room_category"][tp]:
    #         bbox_tp = (np.array(bbox_tp) * meter2pixel).astype(np.int)
    #         bbox = [
    #             np.max([bbox_tp[0] - x_min + border_pad, 0]),
    #             np.max([bbox_tp[1] - y_min + border_pad, 0]),
    #             np.min([bbox_tp[2] - x_min + border_pad, cnt_map.shape[1]]),
    #             np.min([bbox_tp[3] - y_min + border_pad, cnt_map.shape[0]]),
    #         ]
    #         tp_map[bbox[1] : bbox[3], bbox[0] : bbox[2]] |= tp_id
    # tp_map[cnt_map == 0] = 0
    # # display(tp_map, tp_flag=True)
    # imC = cv2.applyColorMap(tp_map, cv2.COLORMAP_JET)

    cv2.imshow("img", cnt_map)
    cv2.waitKey(0)
