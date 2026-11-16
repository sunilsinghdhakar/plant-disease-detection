import cv2
import numpy as np


def calculate_severity(image):
    """
    Experimental leaf disease severity estimate.
    Uses HSV color segmentation.
    Not a validated agricultural diagnosis.
    """

    if image is None or image.size == 0:
        return None, "Invalid image"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Broad mask: include green leaf and possible yellow/brown lesions.
    lower_leaf = np.array([0, 35, 25])
    upper_leaf = np.array([100, 255, 255])

    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(
        leaf_mask, cv2.MORPH_OPEN, kernel
    )
    leaf_mask = cv2.morphologyEx(
        leaf_mask, cv2.MORPH_CLOSE, kernel
    )

    total_leaf_pixels = cv2.countNonZero(leaf_mask)

    if total_leaf_pixels < 100:
        return None, "Leaf area not detected clearly"

    # Approximate yellow/brown lesion pixels within the leaf region.
    lower_lesion = np.array([5, 40, 20])
    upper_lesion = np.array([35, 255, 230])

    lesion_mask = cv2.inRange(hsv, lower_lesion, upper_lesion)
    lesion_mask = cv2.bitwise_and(lesion_mask, leaf_mask)

    infected_pixels = cv2.countNonZero(lesion_mask)

    severity = (infected_pixels / total_leaf_pixels) * 100
    severity = round(min(severity, 100), 2)

    if severity < 5:
        level = "Mild"
    elif severity < 15:
        level = "Moderate"
    else:
        level = "Severe"

    return severity, level


def get_severity_masks(image):
    """
    Return visual masks for leaf area and possible lesions.
    White pixels indicate detected regions.
    """

    if image is None or image.size == 0:
        return None, None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_leaf = np.array([0, 35, 25])
    upper_leaf = np.array([100, 255, 255])

    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    kernel = np.ones((5, 5), np.uint8)

    leaf_mask = cv2.morphologyEx(
        leaf_mask, cv2.MORPH_OPEN, kernel
    )
    leaf_mask = cv2.morphologyEx(
        leaf_mask, cv2.MORPH_CLOSE, kernel
    )

    lower_lesion = np.array([5, 40, 20])
    upper_lesion = np.array([35, 255, 230])

    lesion_mask = cv2.inRange(
        hsv, lower_lesion, upper_lesion
    )

    lesion_mask = cv2.bitwise_and(
        lesion_mask, leaf_mask
    )

    return leaf_mask, lesion_mask
