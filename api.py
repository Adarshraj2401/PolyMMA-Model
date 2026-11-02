"""FastAPI service for the PolyMMA demo surrogate model.

Run from this folder with:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "polymma_surrogate.joblib"

app = FastAPI(
    title="PolyMMA Surrogate API",
    description=(
        "Demo prediction API for the PolyMMA dashboard. The included model is "
        "trained on synthetic illustrative data, not actual Aspen runs."
    ),
    version="0.1.0-demo",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your deployed frontend domain before production.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

bundle: dict[str, Any] | None = None


def load_bundle() -> dict[str, Any]:
    global bundle
    if bundle is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. Run `python train_model.py` first."
            )
        bundle = joblib.load(MODEL_PATH)
    return bundle


class PredictRequest(BaseModel):
    initial_temp_C: float = Field(ge=30.0, le=70.0, description="Initial reactor temperature in °C")
    target_temp_C: float = Field(ge=80.0, le=130.0, description="Target reactor temperature in °C")
    initiator_time_h: float = Field(ge=0.5, le=1.5, description="Initiator injection time in hours")
    batch_duration_h: float = Field(ge=1.0, le=3.0, description="Batch duration in hours")
    n_points: int = Field(default=41, ge=11, le=101, description="Number of points in returned profiles")

    @model_validator(mode="after")
    def check_initiator_before_batch_end(self):
        if self.initiator_time_h >= self.batch_duration_h:
            raise ValueError("Initiator injection time must be earlier than batch end.")
        return self


@app.get("/")
def root():
    return {
        "service": "PolyMMA Surrogate API",
        "status": "running",
        "docs": "/docs",
        "warning": "Demo model trained on synthetic data, not Aspen simulation runs.",
    }


@app.get("/health")
def health():
    try:
        b = load_bundle()
        return {"status": "ok", "model_loaded": True, "dataset_label": b["dataset_label"]}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metadata")
def metadata():
    b = load_bundle()
    return {
        "model_type": b["model_type"],
        "dataset_label": b["dataset_label"],
        "features": b["features"],
        "targets": b["targets"],
        "bounds": b["bounds"],
        "warning": b["warning"],
    }


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        b = load_bundle()
        features = b["features"]
        x = pd.DataFrame([[getattr(request, name) for name in features]], columns=features, dtype=float)
        predicted: dict[str, float] = {}
        uncertainty: dict[str, float] = {}

        # Use the fitted scaler and GPR directly to retrieve its posterior std.
        for target, pipeline in b["models"].items():
            scaler = pipeline.named_steps["scaler"]
            gpr = pipeline.named_steps["gpr"]
            x_scaled = scaler.transform(x)
            mean, std = gpr.predict(x_scaled, return_std=True)
            predicted[target] = float(mean[0])
            uncertainty[target] = float(std[0])

        # Keep predictions in physically sensible display ranges. This is a
        # guardrail for a demo model, not a substitute for process validation.
        predicted["conversion_pct"] = float(np.clip(predicted["conversion_pct"], 0.0, 100.0))
        predicted["resin_yield_kg"] = max(predicted["resin_yield_kg"], 0.0)
        predicted["dpn"] = max(predicted["dpn"], 0.01)
        predicted["dpw"] = max(predicted["dpw"], 0.01)
        predicted["pdi"] = max(predicted["pdi"], 1.0)
        predicted["heat_duty_MMkcal"] = max(predicted["heat_duty_MMkcal"], 0.0)
        tau = float(np.clip(predicted["curve_tau_h"], 0.05, 5.0))
        shape = float(np.clip(predicted["curve_shape"], 0.5, 3.0))

        batch = float(request.batch_duration_h)
        times = np.linspace(0.0, batch, request.n_points)
        denom = max(1.0 - np.exp(-((batch / tau) ** shape)), 1e-9)
        conversion_profile = []
        energy_profile = []
        total_conversion = predicted["conversion_pct"]
        total_energy = predicted["heat_duty_MMkcal"]
        for t in times:
            frac = (1.0 - np.exp(-((float(t) / tau) ** shape))) / denom
            frac = float(np.clip(frac, 0.0, 1.0))
            time_frac = float(t / batch) if batch else 0.0
            e_frac = 0.20 * time_frac + 0.80 * (time_frac**1.30)
            conversion_profile.append({"time_h": float(t), "conversion_pct": total_conversion * frac})
            energy_profile.append({
                "time_h": float(t),
                "cumulative_heat_duty_MMkcal": total_energy * e_frac,
            })

        return {
            "status": "ok",
            "result_type": "synthetic_surrogate_prediction",
            "warning": (
                "Illustrative prediction from a model trained on synthetic demo data. "
                "Not an Aspen simulation result and not suitable for design or operating decisions."
            ),
            "inputs": {name: getattr(request, name) for name in features},
            "outputs": {
                "conversion_pct": predicted["conversion_pct"],
                "resin_yield_kg": predicted["resin_yield_kg"],
                "dpn": predicted["dpn"],
                "dpw": predicted["dpw"],
                "pdi": predicted["pdi"],
                "heat_duty_MMkcal": predicted["heat_duty_MMkcal"],
            },
            "uncertainty_estimate": {
                k: uncertainty[k]
                for k in ["conversion_pct", "resin_yield_kg", "pdi", "heat_duty_MMkcal"]
            },
            "conversion_profile": conversion_profile,
            "energy_profile": energy_profile,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
