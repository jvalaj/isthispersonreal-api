from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import imghdr
import io

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)
# Load model directly

processor = AutoImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()

    # Sanity check: is this a valid image?
    if imghdr.what(None, img_bytes) is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    # Convert bytes to PIL Image
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot process the uploaded image")

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

    # Decode predictions
    labels = model.config.id2label
    scores = {labels[i]: float(probs[i]) for i in range(len(probs))}

    is_real = scores.get("real", 0.0)
    is_fake = scores.get("fake", 0.0)

    return {
        "is_real": is_real > is_fake,
        "confidence": max(is_real, is_fake),
        "scores": scores,
    }
