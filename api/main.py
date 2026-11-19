
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tuned_xgboost_model.joblib"

# Load the complete preprocessing + prediction pipeline
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Last-Mile Delivery Duration Prediction API",
    description="Predicts delivery duration using a trained XGBoost pipeline.",
    version="1.0.0"
)


class DeliveryInput(BaseModel):
    city: str
    region_id: int
    aoi_type: int
    accept_hour: int = Field(ge=0, le=23)
    accept_dayofweek: int = Field(ge=0, le=6)
    accept_month: int = Field(ge=1, le=12)
    distance_km: float = Field(ge=0)
    lat: float
    lng: float


@app.get("/")
def home():
    return {
        "message": "Last-Mile Delivery Duration Prediction API is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict_delivery(data: DeliveryInput):
    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_duration_minutes": round(float(prediction), 2),
        "predicted_duration_hours": round(float(prediction) / 60, 2)
    }
