
import cv2
import numpy as np
from yolo_severity import analyze_severity

# Create a sample green leaf image
image = np.zeros((300, 300, 3), dtype=np.uint8)
image[:] = (0, 180, 0)  # Green background

# Add a brown spot to simulate a disease lesion
cv2.circle(image, (150, 150), 30, (0, 80, 120), -1)

# Calculate disease severity
result = calculate_severity(image)

print("Severity Test Result")
print("--------------------")
print("Severity:", result[0], "%")
print("Severity Level:", result[1])

