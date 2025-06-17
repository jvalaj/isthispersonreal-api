from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, requests, io
from PIL import Image   # only for a quick “is this an image?” sanity-check

HF_ENDPOINT = "https://api-inference.huggingface.co/models/prithivMLmods/deepfake-detector-model-v1"  # :contentReference[oaicite:0]{index=0}
HF_TOKEN    = os.getenv("HF_TOKEN")   # <-- you’ll add this in Render’s dashboard

headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Netlify URL later
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()

    # quick decode to be sure we actually got an image
    try:
        Image.open(io.BytesIO(img_bytes)).verify()
    except Exception:
        raise HTTPException(400, "Uploaded file is not a valid image")

    # ----- call the hosted model -----
    resp = requests.post(
        HF_ENDPOINT,
        headers=headers,
        data=img_bytes,
        timeout=30,
    )
    if resp.status_code != 200:
        raise HTTPException(502, f"HF API error: {resp.text}")

    predictions = resp.json()  # [{'label':'fake','score':0.91}, ...]
    if not isinstance(predictions, list):
        raise HTTPException(502, f"Unexpected HF response: {predictions}")

    fake_score = next((p["score"] for p in predictions if p["label"] == "fake"), None)
    if fake_score is None:
        raise HTTPException(502, "No 'fake' label in HF response")

    real_prob = 1 - fake_score
    return {"is_real": real_prob > fake_score, "confidence": max(real_prob, fake_score)}
