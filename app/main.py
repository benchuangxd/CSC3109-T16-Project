import io
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import predict_image, get_inference_transform  # noqa: E402
from src.config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES      # noqa: E402

app = FastAPI(title="CSC3109-T16 Image Classifier")

MODEL_PATH = ROOT / "app" / "best_model.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
_model     = None


def load_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model weights not found at {MODEL_PATH}")

    from src.models import get_efficientnet_b0  # swap to your best model
    model = get_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    _model = model
    return model


@app.on_event("startup")
def startup():
    try:
        load_model()
        print(f"Model loaded from {MODEL_PATH} on {DEVICE}")
    except RuntimeError as e:
        print(f"Warning: {e}")


@app.get("/")
def root():
    return {"message": "CSC3109-T16 classifier is running", "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image")

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        model = load_model()
        label, confidence, all_probs = predict_image(
            tmp_path, model, CLASS_NAMES, DEVICE, IMAGE_SIZE
        )
    finally:
        os.unlink(tmp_path)

    return JSONResponse({
        "prediction":  label,
        "confidence":  round(confidence, 4),
        "all_probs":   {k: round(v, 4) for k, v in all_probs.items()},
    })
