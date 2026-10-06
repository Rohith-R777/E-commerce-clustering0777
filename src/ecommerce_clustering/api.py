"""FastAPI entry point for serving segment assignments."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="E-Commerce Customer Clustering API", version="0.1.0")


class CustomerFeatures(BaseModel):
    recency: float
    frequency: float
    monetary: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(features: CustomerFeatures) -> dict[str, str | float]:
    if min(features.recency, features.frequency, features.monetary) < 0:
        raise HTTPException(status_code=422, detail="Features must be non-negative")
    return {"profile": "unassigned", "recency": features.recency}
