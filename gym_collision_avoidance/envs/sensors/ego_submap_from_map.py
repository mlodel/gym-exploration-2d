import cv2
import numpy as np


def ego_submap_from_map(
    map: cv2.Mat,
    pos_pxl: list(),
    angle_deg: float,
    submap_size: list(),
    scale_size: list() = None,
) -> cv2.Mat:
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
