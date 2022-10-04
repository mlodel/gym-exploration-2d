import os
import json

import numpy as np

import cv2
from tinydb import TinyDB, Query
from tqdm import tqdm

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

    db = TinyDB("map_db.json")
    # db.truncate()
    # db.all()
    json_prefix = (
        "/home/max/Documents/projects/exploration_2d/HouseExpo/HouseExpo/json/"
    )

    # file_name = "0a5c77794ab1c44936682ccf4562f3c3"
    # file_name = "0a7d100165ef451e2ab508a227e4c403"
    # file_name = "0a1b29dba355df2ab02630133187bfab"
    # file_name = "0a1ee68696e0c1e7b16f44de681a6d3d"

    # create buffers
    map_info = dict(id="", area=0, size=0, rooms=0)

    # iterate over directory
    for file_name in tqdm(os.listdir(json_prefix)):
        file_name = file_name.split(".")[0]

        meter2pixel = 20
        border_pad = 0

        with open(json_prefix + file_name + ".json") as json_file:
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
            (y_max - y_min + border_pad * 2, x_max - x_min + border_pad * 2),
            dtype=np.uint8,
        )

        verts[:, 0] = verts[:, 0] - x_min + border_pad
        verts[:, 1] = verts[:, 1] - y_min + border_pad

        cv2.drawContours(cnt_map, [verts], 0, 255, -1)

        map_info["id"] = file_name
        map_info["area"] = cnt_map.shape[0] * cnt_map.shape[1] / meter2pixel**2
        map_info["size"] = np.max(cnt_map.shape) / meter2pixel
        map_info["rooms"] = json_data["room_num"]

        db.insert(map_info)
