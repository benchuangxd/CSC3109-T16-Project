import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_DIR  = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "resnet18.pth"
CLASSES_PATH = MODEL_DIR / "classes.json"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def predict(image: Image.Image, model):
    tf = get_transform()
    x = tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze()
    pred_idx = probs.argmax().item()
    return CLASS_NAMES[pred_idx], probs[pred_idx].item(), {c: probs[i].item() for i, c in enumerate(CLASS_NAMES)}


# ── Streamlit UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="CSC3109-T16 Classifier", layout="centered")
st.title("CSC3109-T16 Aerial Image Classifier")
st.write("Upload an aerial image to classify it as **beach**, **ferry terminal**, **harbor**, or **river**.")

model = load_model()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width='stretch')

    with st.spinner("Classifying..."):
        label, confidence, all_probs = predict(img, model)

    st.success(f"**Prediction: {label}** ({confidence:.2%} confidence)")

    st.subheader("Confidence Scores")
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        st.progress(prob, text=f"{cls}: {prob:.2%}")
