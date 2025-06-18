import os
os.environ["HF_HOME"] = "/tmp/huggingface"  # Allow caching on Hugging Face Spaces

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + processor
processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
model = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

    labels = model.config.id2label
    scores = {labels[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "result": max(scores, key=scores.get),
        "confidence": max(scores.values()),
        "scores": scores,
    }
