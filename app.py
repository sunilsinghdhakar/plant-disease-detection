
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2

from PIL import Image
from torchvision import models, transforms
from yolo_severity import analyze_severity


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 AI-Based Plant Disease Detection")
st.write(
    "Plant disease classification and severity analysis "
    "using MobileNetV3 Large and YOLO Segmentation."
)

st.info("Powered by MobileNetV3 Large + YOLO Segmentation")


# ---------------- DEVICE ----------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------- LOAD MOBILENET ----------------

@st.cache_resource
def load_classifier():

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

    return model, classes


# ---------------- IMAGE TRANSFORMATION ----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------- SEGMENTATION OVERLAY ----------------

def create_overlay(image_bgr, leaf_mask, spot_mask):

    overlay = image_bgr.copy()

    overlay[leaf_mask] = (0, 255, 0)

    affected_mask = spot_mask & leaf_mask
    overlay[affected_mask] = (0, 0, 255)

    result = cv2.addWeighted(
        image_bgr,
        0.65,
        overlay,
        0.35,
        0
    )

    return cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )


# ---------------- LOAD MODEL ----------------

try:
    classifier, classes = load_classifier()
    st.success("MobileNetV3 model loaded successfully!")
    st.caption(f"Running on: {device}")

except Exception as e:
    st.error(f"MobileNetV3 loading error: {e}")
    st.stop()


# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload Plant Leaf Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Invalid image. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(
        image,
        caption="Uploaded Plant Leaf",
        width=500
    )

    if st.button(
        "🔍 Detect Plant Disease",
        use_container_width=True
    ):

        with st.spinner("Analyzing image using AI models..."):

            try:

                # -------- MOBILENET PREDICTION --------

                image_tensor = transform(image)
                image_tensor = image_tensor.unsqueeze(0).to(device)

                with torch.no_grad():

                    outputs = classifier(image_tensor)

                    probabilities = torch.softmax(
                        outputs,
                        dim=1
                    )

                    top_probs, top_indices = torch.topk(
                        probabilities,
                        k=min(3, len(classes)),
                        dim=1
                    )

                predicted_disease = classes[
                    top_indices[0][0].item()
                ]

                confidence = (
                    top_probs[0][0].item() * 100
                )

                # -------- TOP 3 PREDICTIONS --------

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

                # -------- YOLO SEVERITY ANALYSIS --------

                image_np = np.array(image)

                image_bgr = cv2.cvtColor(
                    image_np,
                    cv2.COLOR_RGB2BGR
                )

                (
                    leaf_mask,
                    spot_mask,
                    severity,
                    severity_level
                ) = analyze_severity(image_bgr)

                # -------- DISPLAY RESULTS --------

                st.success("Prediction Completed Successfully!")

                st.subheader("🧪 Disease Detection Result")

                st.write("**Predicted Disease:**", predicted_disease)

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

                st.subheader("Top 3 Predictions")

                for item in top_predictions:
                    st.write(
                        f'{item["disease"]} — '
                        f'{item["confidence"]:.2f}%'
                    )

                # -------- SEVERITY --------

                st.divider()
                st.subheader("🌿 YOLO Disease Severity Analysis")

                if severity is not None:

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Affected Leaf Area",
                            f"{severity:.2f}%"
                        )

                    with col2:
                        st.metric(
                            "Severity Level",
                            severity_level
                        )

                    st.progress(
                        min(max(int(severity), 0), 100)
                    )

                else:
                    st.warning(severity_level)

                # -------- VISUALIZATION --------

                st.divider()
                st.subheader("🎯 YOLO Segmentation Visualization")

                if leaf_mask is not None and spot_mask is not None:

                    overlay = create_overlay(
                        image_bgr,
                        leaf_mask,
                        spot_mask
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(
                            image,
                            caption="Original Leaf Image",
                            use_container_width=True
                        )

                    with col2:
                        st.image(
                            overlay,
                            caption="YOLO Segmentation Overlay",
                            use_container_width=True
                        )

                    st.caption(
                        "Green: Detected leaf area | "
                        "Red: Detected disease spots"
                    )

                else:
                    st.warning(
                        "YOLO could not detect the leaf or disease spots."
                    )

                st.info(
                    "AI predictions and severity estimates are "
                    "experimental and are not confirmed "
                    "agricultural diagnoses."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Powered by MobileNetV3 Large | YOLO Segmentation | "
    "AI-Based Plant Disease Detection Project"
)
