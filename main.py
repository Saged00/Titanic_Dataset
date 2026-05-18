from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "titanic_model.pkl"
STATIC_DIR = BASE_DIR / "static"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run Titanic.ipynb to create it.")

artifacts = joblib.load(MODEL_PATH)
model = artifacts["model"]
scaler = artifacts["scaler"]
sex_encoder = artifacts["sex_encoder"]
title_encoder = artifacts["title_encoder"]
embarked_encoder = artifacts["embarked_encoder"]
features = artifacts["features"]


app = FastAPI(title="Titanic Survival Predictor", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class PassengerInput(BaseModel):
    pclass: Literal[1, 2, 3] = Field(..., example=3)
    sex: Literal["male", "female"] = Field(..., example="male")
    age: float = Field(..., ge=0, le=120, example=22)
    title: Literal["Mr", "Mrs", "Miss", "Master", "Rare"] = Field(..., example="Mr")
    family_size: int = Field(..., ge=1, le=11, example=1)
    fare: float = Field(..., ge=0, example=7.25)
    embarked: Literal["C", "Q", "S"] = Field(..., example="S")


class PredictionOutput(BaseModel):
    survived: bool
    survived_label: str
    probability: float
    input_summary: dict


def build_features(data: PassengerInput) -> pd.DataFrame:
    sex_enc = int(sex_encoder.transform([data.sex])[0])
    title_enc = int(title_encoder.transform([data.title])[0])
    embarked_enc = int(embarked_encoder.transform([data.embarked])[0])

    # AgeBand: 0=child(0-12), 1=teen(12-18), 2=young adult(18-35), 3=adult(35-60), 4=senior(60+)
    age_band = int(np.digitize(data.age, [0, 12, 18, 35, 60, 100]) - 1)
    age_band = max(0, min(age_band, 4))

    # FareBand logic (matching notebook qcut buckets approx)
    # The notebook uses qcut, so we should ideally use the same boundaries if possible.
    # For simplicity, we keep the existing logic or approximate it.
    fare_band = int(np.digitize(data.fare, [0, 7.91, 14.454, 31.0]) - 1)
    fare_band = max(0, min(fare_band, 3))

    is_alone = int(data.family_size == 1)

    row = pd.DataFrame(
        [[data.pclass, sex_enc, data.age, age_band, title_enc,
          data.family_size, is_alone, data.fare, fare_band, embarked_enc]],
        columns=features
    )

    return pd.DataFrame(scaler.transform(row), columns=features)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    X = build_features(passenger)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return PredictionOutput(
        survived=bool(pred),
        survived_label="Survived" if pred == 1 else "Did not survive",
        probability=round(prob, 4),
        input_summary=passenger.model_dump(),
    )


@app.post("/predict/batch")
def predict_batch(passengers: list[PassengerInput]):
    if len(passengers) > 100:
        raise HTTPException(status_code=400, detail="Max 100 passengers per request.")

    results = []
    for i, p in enumerate(passengers):
        X = build_features(p)
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
        results.append({
            "index": i,
            "survived": bool(pred),
            "survived_label": "Survived" if pred == 1 else "Did not survive",
            "probability": round(prob, 4),
        })

    return {"count": len(results), "predictions": results}


@app.get("/model/info")
def model_info():
    return {
        "model": "XGBoostClassifier",
        "features": features,
        "params": artifacts["best_params"],
        "sex_classes": list(sex_encoder.classes_),
        "title_classes": list(title_encoder.classes_),
        "embarked_classes": list(embarked_encoder.classes_),
    }
