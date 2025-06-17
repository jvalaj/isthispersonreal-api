from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, requests, io, imghdr

HF_ENDPOINT = (
    "https://router.huggingface.co/hf-inference/models/prithivMLmods/deepfake-detector-model"        # ← note: no “-v1” in slug
)
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS  = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type":  "image/jpeg"
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # tighten later
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()

    # quick sanity check that it's an image
    if imghdr.what(None, img_bytes) is None:
        raise HTTPException(400, "Uploaded file is not a valid image")

    # ------ call the router endpoint ------
    resp = requests.post(HF_ENDPOINT, headers=HEADERS, data=img_bytes, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"HF error: {resp.text}")

    preds = resp.json()          # [{'label':'fake','score':0.91}, ...]
    fake = next(p["score"] for p in preds if p["label"] == "fake")
    real = 1 - fake
    return {"is_real": real > fake, "confidence": max(real, fake)}
