
from ultralytics import YOLO
import numpy as np
import cv2

# Load YOLO segmentation model
model = YOLO("models/leaf_lesion_yolo26_seg.pt")


def analyze_severity(image_bgr):

    results = model.predict(
        source=image_bgr,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    h, w = result.orig_shape

    leaf_mask = np.zeros((h, w), dtype=bool)
    spot_mask = np.zeros((h, w), dtype=bool)

    if result.masks is None or result.boxes is None:
        return None, None, None, "No leaf or disease spots detected"

    masks = result.masks.data.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    # Separate leaf and disease spot masks
    for mask, class_id in zip(masks, class_ids):

        resized = cv2.resize(
            mask.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        ) > 0.5

        if class_id == 0:
            leaf_mask |= resized

        elif class_id == 1:
            spot_mask |= resized

    # Calculate leaf and affected area
    leaf_area = np.count_nonzero(leaf_mask)
    disease_area = np.count_nonzero(spot_mask & leaf_mask)

    if leaf_area == 0:
        return leaf_mask, spot_mask, None, "Leaf not detected"

    severity = (disease_area / leaf_area) * 100

    if severity <= 10:
        category = "Low"
    elif severity <= 30:
        category = "Moderate"
    else:
        category = "Severe"

    return leaf_mask, spot_mask, round(severity, 2), category


def create_severity_overlay(image_bgr, leaf_mask, spot_mask):

    if leaf_mask is None or spot_mask is None:
        return image_bgr.copy()

    # Create a copy of original image
    overlay = image_bgr.copy()

    # Green color for detected leaf
    overlay[leaf_mask] = (0, 255, 0)

    # Red color for detected disease spots
    affected_mask = spot_mask & leaf_mask
    overlay[affected_mask] = (0, 0, 255)

    # Blend original image and colored masks
    result = cv2.addWeighted(
        image_bgr,
        0.65,
        overlay,
        0.35,
        0
    )

    return result
