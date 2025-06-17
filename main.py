from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import numpy as np
import cv2                              # already a DeepFace dependency

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()

    # --- NEW: turn bytes -> NumPy image array ---
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # BGR format expected by DeepFace
    if img is None:
        raise ValueError("Could not decode image")

    # DeepFace now gets a proper image array
    result = DeepFace.analyze(img, actions=("deepfake",))
    conf = result["deepfake"]["real"]
    return {"is_real": conf > 0.5, "confidence": conf}
