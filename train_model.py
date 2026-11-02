"""Train small Gaussian-process surrogate models on the demo dataset.

WARNING: the supplied dataset is synthetic. The fitted models are for a UI
proof-of-concept only and are not validated replacements for Aspen Plus.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["initial_temp_C", "target_temp_C", "initiator_time_h", "batch_duration_h"]
TARGETS = [
    "conversion_pct", "resin_yield_kg", "dpn", "dpw", "pdi",
    "heat_duty_MMkcal", "curve_tau_h", "curve_shape",
]


def make_model(seed: int = 42) -> Pipeline:
    # Standardising inputs is important because the four controls use
    # different units and numerical ranges.
    kernel = (
        ConstantKernel(1.0, (0.1, 10.0))
        * Matern(length_scale=np.ones(len(FEATURES)), length_scale_bounds=(0.15, 8.0), nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=seed,
    )
    return Pipeline([("scaler", StandardScaler()), ("gpr", gpr)])


def evaluate_holdout(frame: pd.DataFrame) -> dict:
    train_idx, test_idx = train_test_split(
        np.arange(len(frame)), test_size=0.20, random_state=42
    )
    x_train = frame.iloc[train_idx][FEATURES]
    x_test = frame.iloc[test_idx][FEATURES]
    metrics: dict[str, dict] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for target in TARGETS:
            model = make_model()
            model.fit(x_train, frame.iloc[train_idx][target])
            pred = model.predict(x_test)
            metrics[target] = {
                "mae": float(mean_absolute_error(frame.iloc[test_idx][target], pred)),
                "rmse": float(np.sqrt(mean_squared_error(frame.iloc[test_idx][target], pred))),
                "r2": float(r2_score(frame.iloc[test_idx][target], pred)),
            }
    return metrics


def main() -> None:
    data_path = DATA_DIR / "synthetic_aspen_runs.csv"
    if not data_path.exists():
        raise FileNotFoundError("Dataset not found. Run: python generate_synthetic_data.py")
    frame = pd.read_csv(data_path)
    missing = [c for c in FEATURES + TARGETS if c not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if len(frame) < 20:
        raise ValueError("At least 20 cases are recommended for this demo model.")

    # Hold-out metrics are reported only as a software sanity check on the
    # synthetic response surface. They do not establish real-process accuracy.
    validation_metrics = evaluate_holdout(frame)

    models = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for target in TARGETS:
            model = make_model()
            model.fit(frame[FEATURES], frame[target])
            models[target] = model

    metadata_path = DATA_DIR / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    bundle = {
        "model_type": "independent Gaussian Process Regression models",
        "dataset_label": metadata.get("dataset_label", "SYNTHETIC DEMO DATA"),
        "features": FEATURES,
        "targets": TARGETS,
        "bounds": metadata.get("bounds", {}),
        "models": models,
        "profile_method": "GPR predicts final conversion, curve time constant and shape; smooth Weibull profile is reconstructed from those predictions.",
        "warning": "Synthetic proof-of-concept only. Do not use for engineering decisions or claim Aspen-validated accuracy.",
    }
    model_path = MODEL_DIR / "polymma_surrogate.joblib"
    joblib.dump(bundle, model_path)

    report = {
        "dataset_label": bundle["dataset_label"],
        "cases": int(len(frame)),
        "features": FEATURES,
        "targets": TARGETS,
        "holdout_fraction": 0.20,
        "holdout_metrics_on_synthetic_data_only": validation_metrics,
        "warning": bundle["warning"],
        "model_file": str(model_path.relative_to(ROOT)),
    }
    (MODEL_DIR / "training_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved model: {model_path}")


if __name__ == "__main__":
    main()
