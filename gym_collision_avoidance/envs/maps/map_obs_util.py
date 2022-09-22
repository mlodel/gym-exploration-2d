import numpy as np
import cv2


def ego_rot_global_map(
    map: np.ndarray,
    map_cell: tuple,
    angle: float,
    output_size: tuple,
    border_value: float = 0.0,
):
    # Taking image height and width
    imgHeight, imgWidth = map.shape[0], map.shape[1]

    # Computing the centre x,y coordinates
    # of an image
    centreY, centreX = imgHeight // 2, imgWidth // 2

    # Computing 2D rotation Matrix to rotate an image
    alpha = np.rad2deg(np.arctan2(imgWidth, imgHeight))
    rotationMatrix = cv2.getRotationMatrix2D((centreX, centreY), alpha, 1.0)

    # After computing the new height & width of an image
    # we also need to update the values of rotation matrix
    sub_img_width = int(np.ceil(np.sqrt(map.shape[0] ** 2 + map.shape[1] ** 2)))
    rotationMatrix[0][2] += (sub_img_width / 2) - centreX
    rotationMatrix[1][2] += (sub_img_width / 2) - centreY

    # Now, we will perform actual image rotation
    # bordervalue = 1.0 if bin else 0.0
    rotatingimage = cv2.warpAffine(
        map,
        rotationMatrix,
        (sub_img_width, sub_img_width),
        borderValue=border_value,
        flags=cv2.INTER_LINEAR,
    )

    # Extend the map to perform second rotation around robot position
    # create map with size 3x sub_img_width and place rotated map in the middle
    ext_map = (
        np.ones((3 * sub_img_width, 3 * sub_img_width), dtype=np.uint8) * border_value
    )
    ext_map[
        sub_img_width : 2 * sub_img_width, sub_img_width : 2 * sub_img_width
    ] = rotatingimage

    # Transform robot position to rotated map
    transform_point = cv2.transform(
        np.asarray(np.flip(map_cell)).reshape((1, 1, 2)), rotationMatrix
    )[0][0]
    # Shift robot position in extended map
    point = transform_point + np.array([sub_img_width, sub_img_width], dtype=int)

    # Rotate extended map around robot position with robot orientation
    rot_mat = cv2.getRotationMatrix2D(
        (int(point[0]), int(point[1])), alpha - angle, 1.0
    )
    rot2 = cv2.warpAffine(
        ext_map,
        rot_mat,
        ext_map.shape[1::-1],
        borderValue=border_value,
        flags=cv2.INTER_LINEAR,
    )

    # Crop rotated map to be centered around robot position
    # Size is 2x sub_img_width (must fit 2x diagonal of map)
    final = rot2[
        point[1] - sub_img_width : point[1] + sub_img_width,
        point[0] - sub_img_width : point[0] + sub_img_width,
    ]
    # final = cv2.flip(final, 1)

    cv2.imshow("map", final * 255)
    cv2.waitKey(1)

    # Dilate final map to preserve features
    final = cv2.dilate(final, np.ones((3, 3), np.uint8), iterations=3)

    cv2.imshow("map2", final * 255)
    cv2.waitKey(1)

    if final.size == 0:
        test = 1

    # Resize to output size
    final_resize = cv2.resize(final, output_size, interpolation=cv2.INTER_LINEAR)

    return final_resize


def ego_fixed_global_map(
    map: np.ndarray,
    map_cell: tuple,
    output_size: tuple,
    border_value: float = 0.0,
):
    # Create squared extended map with size 3x larger map size and place square map in the middle
    long_size = max(map.shape)
    ext_map = np.ones((3 * long_size, 3 * long_size), dtype=np.uint8) * border_value

    # Make map square
    if map.shape[0] != map.shape[1]:
        if map.shape[0] > map.shape[1]:
            diff = map.shape[0] - map.shape[1]
            pad1 = diff // 2 if diff % 2 == 0 else diff // 2 + 1
            pad2 = diff // 2
            map_square = np.pad(
                map, ((0, 0), (pad1, pad2)), "constant", constant_values=border_value
            )
            pad_x = pad1
            pad_y = 0
        else:
            diff = map.shape[1] - map.shape[0]
            pad1 = diff // 2 if diff % 2 == 0 else diff // 2 + 1
            pad2 = diff // 2
            map_square = np.pad(
                map, ((pad1, pad2), (0, 0)), "constant", constant_values=border_value
            )
            pad_x = 0
            pad_y = pad1
    else:
        map_square = map
        pad_x = 0
        pad_y = 0

    ext_map[long_size : 2 * long_size, long_size : 2 * long_size] = map_square

    # Shift robot position in extended map
    point = np.array(map_cell, dtype=int) + np.array(
        [long_size + pad_y, long_size + pad_x], dtype=int
    )

    # Crop map to be centered around robot position
    # Size is 2x long_size
    final = ext_map[
        point[0] - long_size : point[0] + long_size,
        point[1] - long_size : point[1] + long_size,
    ]

    # Dilate final map to preserve features
    iterations = np.ceil(final.shape[0] / 200).astype(int)
    final = cv2.dilate(final, np.ones((3, 3), np.uint8), iterations=iterations)

    # Resize to output size
    final_resize = cv2.resize(final, output_size, interpolation=cv2.INTER_LINEAR)

    return final_resize


def ego_submap_from_map(
    map: np.ndarray,
    pos_pxl: tuple,
    angle_deg: float,
    submap_size: int,
    scale_size: list = None,
    border_value: int = 1,
) -> np.ndarray:
    """Creates an egocentric submap from a global map (map),
    the submap is centered in the agent's position (pos_pxl) and rotated according to its orientation (angle_deg)

    Args:
        map (cv2.Mat): global map
        pos_pxl (list): agent position in PIXELS in the global mal
        angle_deg (float): agent orientation IN DEGREES
        submap_size (int): width/height of the submap in pixels
        scale_size(list): OPTIONAL Final output size of submap in pixels [width,height]

    Returns:
        cv2.Mat: _description_
    """

    pos = pos_pxl
    angle = angle_deg

    # Add border to map with half size of submap
    img = cv2.copyMakeBorder(
        map,
        submap_size // 2,
        submap_size // 2,
        submap_size // 2,
        submap_size // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=border_value,
    )

    # Shift robot position in extended map
    pos = np.array([pos[0] + submap_size // 2, pos[1] + submap_size // 2], dtype=int)

    # Rotate extended map around robot position with robot orientation
    M = cv2.getRotationMatrix2D(
        center=(int(pos[1]), int(pos[0])), angle=-angle, scale=1
    )
    rot_img = cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]), borderValue=1, flags=cv2.INTER_LINEAR
    )

    # Crop rotated map to be centered around robot position
    # Compute indices first
    submap_idc_x_l = int(pos[1] - submap_size / 2)
    submap_idc_x_h = int(pos[1] + submap_size / 2)
    submap_idc_y_l = int(pos[0] - submap_size / 2)
    submap_idc_y_h = int(pos[0] + submap_size / 2)
    submap_img = rot_img[submap_idc_y_l:submap_idc_y_h, submap_idc_x_l:submap_idc_x_h]

    cv2.imshow("map", submap_img * 255)
    cv2.waitKey(1)

    # Dilate final map to preserve features
    # iterations = np.ceil(submap_img.shape[0] / 200).astype(int)
    submap_img = cv2.dilate(submap_img, np.ones((3, 3), np.uint8), iterations=1)

    # Resize to output size if specified
    if scale_size is not None and isinstance(scale_size, list):
        if len(scale_size) == 2:
            scale_img = cv2.resize(
                submap_img, tuple(scale_size), interpolation=cv2.INTER_NEAREST
            )
        else:
            raise TypeError(
                "Scale size for submap must be list of two integers [width, height]"
            )

        return scale_img
    else:
        return submap_img
