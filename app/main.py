"""Task 6 - serve the selected model as POST /predict.

Run:  uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
model = joblib.load(ROOT / "models" / "model.joblib")
meta = json.loads((ROOT / "models" / "best_run.json").read_text(encoding="utf-8"))
LOW_CONFIDENCE = 0.5  # below this, route the line to a human

app = FastAPI(title="Invoice line categoriser", version="1.0")


class PredictRequest(BaseModel):
    descriptions: list[str] = Field(..., min_length=1,
                                    examples=[["Rebar T16 16mm / حديد تسليح ١٦ مم", "رمل مغسول"]])


class Prediction(BaseModel):
    description: str
    category: str
    confidence: float
    needs_review: bool
    scores: dict[str, float]


class PredictResponse(BaseModel):
    model: str
    run_id: str
    predictions: list[Prediction]


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["name"], "run_id": meta["run_id"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    probs = model.predict_proba(req.descriptions)
    classes = model.classes_
    preds = []
    for text, p in zip(req.descriptions, probs):
        i = int(p.argmax())
        preds.append(Prediction(
            description=text, category=classes[i], confidence=round(float(p[i]), 3),
            needs_review=bool(p[i] < LOW_CONFIDENCE),
            scores={c: round(float(s), 3) for c, s in zip(classes, p)}))
    return PredictResponse(model=meta["name"], run_id=meta["run_id"], predictions=preds)
