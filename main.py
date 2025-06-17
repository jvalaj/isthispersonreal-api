from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten later
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()
    result = DeepFace.analyze(img_bytes, actions=("deepfake",))
    conf = result["deepfake"]["real"]
    return {"is_real": conf > 0.5, "confidence": conf}
