
from ultralytics import YOLO
import numpy as np
import cv2

# Load trained YOLO segmentation model
model = YOLO("models/leaf_lesion_yolo26_seg.pt")

# Predict on sample image
results = model.predict(
    source="sample_leaf.JPG",
    conf=0.25,
    save=True
)

result = results[0]

if result.masks is None:
    print("No segmentation masks detected.")
else:
    # Get masks and class IDs
    masks = result.masks.data.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    # Original image dimensions
    h, w = result.orig_shape

    leaf_mask = np.zeros((h, w), dtype=bool)
    spot_mask = np.zeros((h, w), dtype=bool)

    # Combine masks belonging to each class
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

    # Count pixels
    leaf_area = np.count_nonzero(leaf_mask)

    # Count only disease pixels inside leaf area
    disease_area = np.count_nonzero(spot_mask & leaf_mask)

    if leaf_area == 0:
        print("Leaf not detected. Cannot calculate severity.")
    else:
        severity = (disease_area / leaf_area) * 100

        print("\n--- PLANT DISEASE SEVERITY ---")
        print(f"Leaf Area: {leaf_area} pixels")
        print(f"Disease Area: {disease_area} pixels")
        print(f"Disease Severity: {severity:.2f}%")

        if severity <= 10:
            category = "Low"
        elif severity <= 30:
            category = "Moderate"
        else:
            category = "Severe"

        print(f"Severity Category: {category}")

