
import io
import cv2
import base64
import torch
import numpy as np
import torch.nn as nn

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import models, transforms

from yolo_severity import (
    analyze_severity,
    create_severity_overlay
)


# ==================================================
# 1. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="AI-Based Plant Disease Detection API",
    description=(
        "MobileNetV3 disease classification and "
        "YOLO segmentation-based severity analysis."
    ),
    version="1.1.0"
)


# ==================================================
# 2. DEVICE CONFIGURATION
# ==================================================

if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ==================================================
# 3. LOAD MOBILENETV3 MODEL
# ==================================================

try:

    checkpoint = torch.load(
        "models/best_mobilenetv3.pth",
        map_location="cpu",
        weights_only=False
    )

    classes = checkpoint["classes"]

    model = models.mobilenet_v3_large(weights=None)

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("MobileNetV3 loaded successfully!")

except Exception as e:

    raise RuntimeError(
        f"MobileNetV3 loading failed: {str(e)}"
    )


# ==================================================
# 4. IMAGE TRANSFORMATION
# ==================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==================================================
# 5. HOME API
# ==================================================

@app.get("/")
def home():

    return {
        "project": "AI-Based Plant Disease Detection",
        "status": "Running",
        "device": str(device),
        "models": [
            "MobileNetV3 Large",
            "YOLO Segmentation"
        ],
        "message": "API is running successfully!"
    }


# ==================================================
# 6. HEALTH CHECK API
# ==================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "mobilenet": "loaded",
        "yolo": "loaded",
        "device": str(device)
    }


# ==================================================
# 7. PLANT DISEASE PREDICTION API
# ==================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_types = [
        "image/jpeg",
        "image/png"
    ]

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are allowed."
        )

    try:

        # Read uploaded image
        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        # Maximum file size: 10 MB
        if len(contents) > 10 * 1024 * 1024:

            raise HTTPException(
                status_code=413,
                detail="Image size must be less than 10 MB."
            )

        # Decode image
        try:

            image = Image.open(
                io.BytesIO(contents)
            ).convert("RGB")

        except (UnidentifiedImageError, OSError):

            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image."
            )

        # ==================================================
        # 8. MOBILENETV3 DISEASE CLASSIFICATION
        # ==================================================

        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            top_probs, top_indices = torch.topk(
                probabilities,
                k=min(3, len(classes)),
                dim=1
            )

        predicted_index = top_indices[0][0].item()

        predicted_disease = classes[predicted_index]

        confidence = top_probs[0][0].item() * 100

        # Top 3 predictions
        top_predictions = []

        for probability, index in zip(
            top_probs[0],
            top_indices[0]
        ):

            top_predictions.append({

                "disease": classes[index.item()],

                "confidence": round(
                    probability.item() * 100,
                    2
                )

            })

        # ==================================================
        # 9. YOLO DISEASE SEGMENTATION
        # ==================================================

        image_np = np.array(image)

        image_bgr = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

        (
            leaf_mask,
            lesion_mask,
            severity_percent,
            severity_level
        ) = analyze_severity(image_bgr)

        overlay_base64 = None

        # ==================================================
        # 10. CREATE SEGMENTATION OVERLAY
        # ==================================================

        if leaf_mask is not None and lesion_mask is not None:

            overlay_image = create_severity_overlay(
                image_bgr,
                leaf_mask,
                lesion_mask
            )

            # Encode overlay as JPEG
            success, encoded_image = cv2.imencode(
                ".jpg",
                overlay_image,
                [cv2.IMWRITE_JPEG_QUALITY, 90]
            )

            if success:

                overlay_base64 = base64.b64encode(
                    encoded_image.tobytes()
                ).decode("utf-8")

        else:

            severity_percent = None

            severity_level = (
                "Leaf or disease area could not be detected."
            )

        # ==================================================
        # 11. FINAL API RESPONSE
        # ==================================================

        return {

            "success": True,

            "predicted_disease": predicted_disease,

            "confidence": round(confidence, 2),

            "top_3_predictions": top_predictions,

            "severity_percentage": severity_percent,

            "severity_level": severity_level,

            "overlay_image": overlay_base64,

            "device": str(device),

            "message": "Prediction completed successfully."

        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

    finally:

        await file.close()


# ==================================================
# 12. PROJECT INFORMATION API
# ==================================================

@app.get("/about")
def about():

    return {

        "project_name": (
            "AI-Based Plant Disease Detection "
            "and Severity Analysis System"
        ),

        "classification_model": "MobileNetV3 Large",

        "segmentation_model": "YOLO Segmentation",

        "supported_crops": [
            "Apple",
            "Corn",
            "Grape",
            "Potato",
            "Tomato"
        ],

        "prediction_features": [
            "Disease Classification",
            "Top 3 Predictions",
            "Confidence Score",
            "Leaf Segmentation",
            "Disease Spot Segmentation",
            "Disease Severity Estimation",
            "Segmentation Overlay Visualization"
        ]

    }
