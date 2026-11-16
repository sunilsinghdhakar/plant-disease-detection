
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Device selection
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load trained checkpoint
checkpoint = torch.load(
    "best_mobilenetv3.pth",
    map_location="cpu",
    weights_only=False
)

classes = checkpoint["classes"]
num_classes = len(classes)

# Recreate model architecture
model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    num_classes
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load leaf image
image = Image.open("sample_leaf.JPG").convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_index = torch.max(
        probabilities, dim=1
    )

# Display result
predicted_class = classes[predicted_index.item()]
confidence_value = confidence.item() * 100

print("\n========== PLANT DISEASE DETECTION ==========")
print("Predicted Disease:", predicted_class)
print(f"Confidence: {confidence_value:.2f}%")
print("Device:", device)
print("=============================================")

