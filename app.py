import io
import os

import torch
import torch.nn.functional as F
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MODEL_PATH = "emotion_detector.pth"

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Missing model file: {MODEL_PATH}")

    model = get_model(num_classes=len(EMOTIONS), freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded on {DEVICE}")

    yield

    model = None


app = FastAPI(title="Emotion Detector", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Upload a JPEG or PNG image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode image") from exc

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze().cpu().tolist()

    results = dict(zip(EMOTIONS, [round(probability, 4) for probability in probabilities]))
    dominant = max(results, key=results.get)
    return {
        "emotion": dominant,
        "confidence": results[dominant],
        "all_emotions": results,
    }


@app.get("/")
def index():
    return FileResponse("static/index.html")