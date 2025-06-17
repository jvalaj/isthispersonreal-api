from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch, io

MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"  # HF model card :contentReference[oaicite:0]{index=0}
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
id2label = {0: "fake", 1: "real"}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # --- load uploaded bytes into PIL ---
    img_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Cannot decode image")

    # --- run the SigLIP classifier ---
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    fake_prob, real_prob = float(probs[0]), float(probs[1])

    return {
        "is_real": real_prob > fake_prob,
        "confidence": max(real_prob, fake_prob)
    }
