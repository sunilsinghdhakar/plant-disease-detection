
import torch
import torch.nn as nn
from torchvision import models

# 1. Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)

# 2. Load checkpoint
checkpoint = torch.load(
    "best_mobilenetv3.pth",
    map_location="cpu",
    weights_only=False
)

classes = checkpoint["classes"]
num_classes = len(classes)

# 3. Create MobileNetV3 Large
model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    num_classes
)

# 4. Load trained weights
model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()

# 5. Display model information
print("\nModel loaded successfully!")
print("Model: MobileNetV3 Large")
print("Total classes:", num_classes)
print("Best validation accuracy:", checkpoint["best_accuracy"])
print("Best epoch:", checkpoint["epoch"])

print("\nDisease Classes:")

for i, name in enumerate(classes):
    print(i, name)

print("\nModel is ready for prediction!")

