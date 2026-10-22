from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision.models import resnet50, ResNet50_Weights
import torch
import io

app = FastAPI(title="ResNet-50 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
preprocess = weights.transforms()
categories = weights.meta["categories"]

@app.get("/")
def home():
    return {"status": "online", "model": "ResNet-50"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image")

    tensor = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        probabilities = torch.nn.functional.softmax(
            model(tensor), dim=1
        )[0]

    scores, indices = probabilities.topk(5)

    return {
        "predictions": [
            {
                "label": categories[index.item()],
                "confidence": round(score.item(), 6)
            }
            for score, index in zip(scores, indices)
        ]
    }
