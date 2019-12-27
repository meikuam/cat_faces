import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from skimage.morphology import binary_dilation, binary_erosion, disk


def binary_mask(mask: np.ndarray):
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    thresh = threshold_otsu(mask)
    binary = mask > thresh
    return binary.astype(np.uint8)


def bbox_mask(img: np.ndarray) -> list:
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return [cmin, rmin, cmax, rmax]


def bbox_poly(poly: (list, np.ndarray)) -> list:
    if isinstance(poly, np.ndarray):
        xmin = np.min(poly[:, 0])
        xmax = np.max(poly[:, 0])
        ymin = np.min(poly[:, 1])
        ymax = np.max(poly[:, 1])
    elif isinstance(poly, list):
        xmin = np.min(poly[0][:, 0])
        xmax = np.max(poly[0][:, 0])
        ymin = np.min(poly[0][:, 1])
        ymax = np.max(poly[0][:, 1])
        for p in poly[1:]:
            xmin = min(xmin, np.min(p[:, 0]))
            xmax = max(xmax, np.max(p[:, 0]))
            ymin = min(ymin, np.min(p[:, 1]))
            ymax = max(ymax, np.max(p[:, 1]))

    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def poly_mask(mask: np.ndarray) -> list:
    thresh = threshold_otsu(mask)
    contours = find_contours(mask, level=thresh)
    return contours


def dilate_mask(mask: np.ndarray, radius=20) -> np.ndarray:
    selem = disk(radius=radius)
    dilated = binary_dilation(mask, selem).astype(np.uint8)
    dilated[dilated > 0] = 255

    return dilated


def erode_mask(mask: np.ndarray, radius=4) -> np.ndarray:
    selem = disk(radius=radius)
    eroded = binary_erosion(mask, selem).astype(np.uint8)
    eroded[eroded > 0] = 255

    return eroded


def crop_image(image: np.ndarray, bbox: (list, tuple)) -> np.ndarray:
    crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return crop


def crop_image_by_mask(
        image: np.ndarray,
        mask: np.ndarray,
        erode_thickness=0,
        edge_thickness=0,
        edge_color=(255, 255, 255, 255)
) -> np.ndarray:
    binary = binary_mask(mask)

    if erode_thickness > 0:
        binary = erode_mask(binary, erode_thickness)

    if edge_thickness > 0:
        h, w = binary.shape[:2]
        bbox = bbox_mask(binary)
        bbox = [
            max(bbox[0] - (edge_thickness + 5), 0),
            max(bbox[1] - (edge_thickness + 5), 0),
            min(bbox[2] + (edge_thickness + 5), w),
            min(bbox[3] + (edge_thickness + 5), h)
        ]

        image = crop_image(image, bbox)
        binary = crop_image(binary, bbox)

        dilated = dilate_mask(binary, radius=edge_thickness)
        image[dilated == 0] = 0
        dilated[binary > 0] = 0
        alpha_image = np.concatenate([image, binary[:, :, np.newaxis]], axis=2)
        #         alpha_image[dilated > 0] = 255

        for c in range(len(edge_color)):
            alpha_image[:, :, c] = np.where(
                dilated > 0,
                edge_color[c],
                alpha_image[:, :, c]
            )
    else:
        bbox = bbox_mask(binary)
        image = crop_image(image, bbox)
        binary = crop_image(binary, bbox)
        alpha_image = np.concatenate([image, binary[:, :, np.newaxis]], axis=2)

    return alpha_image
