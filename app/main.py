import io
from typing import List
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from app.inference import load_network, predict_digit

app = FastAPI(
    title = "Handwritten Digit Recognition",
    description = "The API that recognizes handwritten digits using a DeepConvNet trained on MNIST.",
    version = "0.1.0",
)

class PredictionResponse(BaseModel):
    predicted_digit: int = Field(..., ge=0, le=9, description="Predicted digit(0-9)")
    confidence: float = Field(..., ge=0, le=1.0, description="Confidence score")
    probabilities: List[float] = Field(..., description="Confidence score for each class")

@app.on_event("startup")
def _startup() -> None:
    load_networt()

@app.get("/")
def health_check() -> dict:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="Handwritten digit image file")):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload image file.")

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not load as an image.")

    result = predict_digit(image)
    return PredictionResponse(**result)
