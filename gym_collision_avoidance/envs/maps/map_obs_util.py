import numpy as np
import cv2


def ego_global_map(
    map: np.ndarray,
    map_cell: tuple,
    angle: float,
    sub_img_width: int,
    output_size: tuple,
    border_value: float = 0.0,
):
    # Taking image height and width
    imgHeight, imgWidth = map.shape[0], map.shape[1]

    # Computing the centre x,y coordinates
    # of an image
    centreY, centreX = imgHeight // 2, imgWidth // 2

    # Computing 2D rotation Matrix to rotate an image
    rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), 45, 1.0)

    # After computing the new height & width of an image
    # we also need to update the values of rotation matrix
    rotationMatrix[0][2] += (sub_img_width / 2) - centreX
    rotationMatrix[1][2] += (sub_img_width / 2) - centreY

    # Now, we will perform actual image rotation
    # bordervalue = 1.0 if bin else 0.0
    rotatingimage = cv2.warpAffine(
        map,
        rotationMatrix,
        (sub_img_width, sub_img_width),
        borderValue=border_value,
    )

    ext_map = np.zeros((3 * sub_img_width, 3 * sub_img_width), dtype=np.float32)
    ext_map[
        sub_img_width : 2 * sub_img_width, sub_img_width : 2 * sub_img_width
    ] = rotatingimage

    transform_point = cv2.transform(
        np.asarray(np.flip(map_cell)).reshape((1, 1, 2)), rotationMatrix
    )[0][0]

    point = transform_point + np.array([sub_img_width, sub_img_width], dtype=int)

    rot_mat = cv2.getRotationMatrix2D((int(point[0]), int(point[1])), 45 - angle, 1.0)
    rot2 = cv2.warpAffine(
        ext_map, rot_mat, ext_map.shape[1::-1], borderValue=border_value
    )

    final = rot2[
        point[1] - sub_img_width : point[1] + sub_img_width,
        point[0] - sub_img_width : point[0] + sub_img_width,
    ]
    # final = cv2.flip(final, 1)
    final_resize = cv2.resize(final, output_size, interpolation=cv2.INTER_LINEAR)

    # return np.expand_dims((final_resize * 255).astype(np.uint8), axis=0)
    return (final_resize * 255).astype(np.uint8)


def ego_submap_from_map(
    map: np.ndarray,
    pos_pxl: list,
    angle_deg: float,
    submap_size: list,
    scale_size: list = None,
) -> np.ndarray:
    """Creates an egocentric submap from a global map (map),
    the submap is centered in the agent's position (pos_pxl) and rotated according to its orientation (angle_deg)

    Args:
        map (cv2.Mat): global map
        pos_pxl (list): agent position in PIXELS in the global mal
        angle_deg (float): agent orientation IN DEGREES
        submap_size (list): size of the submap in pixels [width, height]
        scale_size(list): OPTIONAL Final output size of submap in pixels [width,height]

    Returns:
        cv2.Mat: _description_
    """

    pos = pos_pxl
    angle = angle_deg

    img = cv2.copyMakeBorder(
        map,
        submap_size[0] // 2,
        submap_size[0] // 2,
        submap_size[1] // 2,
        submap_size[1] // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=1,
    )

    pos = np.array(
        [pos[0] + submap_size[0] // 2, pos[1] + submap_size[1] // 2], dtype=int
    )

    M = cv2.getRotationMatrix2D(
        center=(int(pos[0]), int(pos[1])), angle=-angle, scale=1
    )

    rot_img = cv2.warpAffine(img, M, img.shape, borderValue=1)

    submap_idc_x_l = int(pos[0] - submap_size[0] / 2)
    submap_idc_x_h = int(pos[0] + submap_size[0] / 2)
    submap_idc_y_l = int(pos[1] - submap_size[1] / 2)
    submap_idc_y_h = int(pos[1] + submap_size[1] / 2)

    submap_img = rot_img[submap_idc_y_l:submap_idc_y_h, submap_idc_x_l:submap_idc_x_h]

    if scale_size is not None and isinstance(scale_size, list):
        if len(scale_size) == 2:
            scale_img = cv2.resize(
                submap_img, tuple(scale_size), interpolation=cv2.INTER_CUBIC
            )
        else:
            raise TypeError(
                "Scale size for submap must be list of two integers [width, height]"
            )

    return scale_img
