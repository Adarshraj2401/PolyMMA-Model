"""Generate a clearly-labelled synthetic PolyMMA demo dataset.

The only measured anchor used here is the R-101 baseline reported in the
project report. All other cases and all conversion/energy time profiles are
synthetic illustrations, NOT Aspen Plus simulation results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

FEATURES = ["initial_temp_C", "target_temp_C", "initiator_time_h", "batch_duration_h"]
BOUNDS = {
    "initial_temp_C": [30.0, 70.0],
    "target_temp_C": [80.0, 130.0],
    "initiator_time_h": [0.5, 1.5],
    "batch_duration_h": [1.0, 3.0],
}
BASELINE = {
    "initial_temp_C": 40.0,
    "target_temp_C": 105.0,
    "initiator_time_h": 1.01,
    "batch_duration_h": 2.00,
}
BASELINE_OUTPUTS = {
    "conversion_pct": 89.91,
    "resin_yield_kg": 3034.5,
    "dpn": 71.14,
    "dpw": 227.26,
    "pdi": 3.195,
    "heat_duty_MMkcal": 0.508,
}


def process_response(x: dict[str, float]) -> dict[str, float]:
    """Physics-inspired response surface calibrated to the reported baseline.

    This is a demo surrogate generator, not a kinetic model and not Aspen.
    Terms encode plausible smooth trends and trade-offs while exactly matching
    the supplied baseline KPIs.
    """
    ti = float(x["initial_temp_C"])
    tf = float(x["target_temp_C"])
    inj = float(x["initiator_time_h"])
    batch = float(x["batch_duration_h"])

    dti, dtf, dinj, db = ti - 40.0, tf - 105.0, inj - 1.01, batch - 2.0

    conversion = (
        89.91
        + 0.050 * dti
        + 0.080 * dtf
        + 4.00 * db
        - 0.80 * dinj
        - 0.0012 * dtf**2
        - 0.0006 * dti**2
        + 0.0004 * dti * dtf
    )
    conversion = float(np.clip(conversion, 50.0, 98.5))

    # Resin mass tracks conversion, with a small batch-duration contribution.
    resin_yield = 3034.5 * (conversion / 89.91) * (1.0 + 0.010 * db)

    # Molecular-weight indicators: smooth illustrative dependencies.
    pdi = (
        3.195
        + 0.009 * (dtf / 25.0) ** 2
        + 0.006 * (dti / 20.0) ** 2
        + 0.010 * (dinj / 0.5) ** 2
        + 0.004 * db**2
        - 0.0015 * ((conversion - 89.91) / 10.0)
    )
    chain_factor = np.exp(-0.0020 * dtf - 0.0015 * dti + 0.010 * dinj + 0.004 * db)
    dpn = 71.14 * chain_factor
    # Calibrated separately to the reported rounded baseline DPw.
    dpw = 227.26 * chain_factor * (pdi / 3.195)

    # Positive cumulative heat-duty proxy, exactly 0.508 MMkcal at baseline.
    heat = (
        0.508
        + 0.0010 * dti
        + 0.0014 * dtf
        + 0.040 * db
        - 0.008 * dinj
        + 0.000020 * dtf**2
    )
    heat = max(float(heat), 0.05)

    # Parameters used to reconstruct a smooth time-dependent conversion curve.
    tau = 0.58 * np.exp(-0.012 * dti - 0.010 * dtf + 0.20 * dinj)
    tau = float(np.clip(tau, 0.12, 2.0))
    shape = float(np.clip(1.55 + 0.08 * dinj - 0.001 * dtf + 0.0005 * dti, 1.1, 2.2))

    return {
        "conversion_pct": conversion,
        "resin_yield_kg": float(resin_yield),
        "dpn": float(dpn),
        "dpw": float(dpw),
        "pdi": float(pdi),
        "heat_duty_MMkcal": heat,
        "curve_tau_h": tau,
        "curve_shape": shape,
    }


def sample_design(n: int = 50, seed: int = 2026) -> list[dict[str, float]]:
    """Create 1 reported baseline anchor + 49 space-filling synthetic cases."""
    rows = [BASELINE.copy()]
    sampler = qmc.LatinHypercube(d=4, seed=seed, optimization="random-cd")
    # Rejection-sample invalid cases where injection occurs after batch end.
    candidates = sampler.random(n=400)
    mins = np.array([BOUNDS[k][0] for k in FEATURES])
    maxs = np.array([BOUNDS[k][1] for k in FEATURES])
    scaled = qmc.scale(candidates, mins, maxs)
    for values in scaled:
        x = dict(zip(FEATURES, map(float, values)))
        if x["batch_duration_h"] <= x["initiator_time_h"] + 0.10:
            continue
        # Avoid duplicating the baseline anchor.
        if np.linalg.norm((np.array(values) - np.array(list(BASELINE.values()))) / (maxs - mins)) < 0.03:
            continue
        rows.append(x)
        if len(rows) == n:
            break
    if len(rows) != n:
        raise RuntimeError(f"Could only generate {len(rows)} valid cases; expected {n}.")
    return rows


def main() -> None:
    cases = sample_design(50)
    run_rows: list[dict] = []
    profile_rows: list[dict] = []

    for i, x in enumerate(cases, start=1):
        y = process_response(x)
        source = "reported_baseline_anchor" if i == 1 else "synthetic_physics_inspired_case"
        row = {"run_id": f"RUN-{i:03d}", "data_source": source, **x, **y}
        run_rows.append(row)

        # All profiles are synthetic, including the baseline profile: the
        # project report provides final KPIs, not the underlying time series.
        times = np.linspace(0.0, x["batch_duration_h"], 41)
        denom = 1.0 - np.exp(-((x["batch_duration_h"] / y["curve_tau_h"]) ** y["curve_shape"]))
        denom = max(float(denom), 1e-9)
        for t in times:
            frac = (1.0 - np.exp(-((t / y["curve_tau_h"]) ** y["curve_shape"]))) / denom
            conv_t = y["conversion_pct"] * float(np.clip(frac, 0.0, 1.0))
            time_frac = float(t / x["batch_duration_h"]) if x["batch_duration_h"] else 0.0
            energy_frac = 0.20 * time_frac + 0.80 * (time_frac**1.30)
            profile_rows.append({
                "run_id": f"RUN-{i:03d}",
                "data_source": "synthetic_profile",
                "time_h": float(t),
                "conversion_pct": conv_t,
                "cumulative_heat_duty_MMkcal": y["heat_duty_MMkcal"] * energy_frac,
            })

    runs = pd.DataFrame(run_rows)
    profiles = pd.DataFrame(profile_rows)
    runs.to_csv(DATA_DIR / "synthetic_aspen_runs.csv", index=False, float_format="%.6f")
    profiles.to_csv(DATA_DIR / "synthetic_time_profiles.csv", index=False, float_format="%.6f")

    metadata = {
        "dataset_label": "SYNTHETIC DEMO DATA - NOT ASPEN SIMULATION OUTPUT",
        "number_of_cases": len(runs),
        "profiles_per_case": 41,
        "features": FEATURES,
        "bounds": BOUNDS,
        "reported_baseline_anchor": BASELINE,
        "reported_baseline_outputs": BASELINE_OUTPUTS,
        "notes": [
            "The R-101 baseline KPI values are taken from the user's Bi-Weekly Report No. 6.",
            "The report's example current-configuration metrics have no associated input settings and were not used as a training point.",
            "All non-baseline cases are synthetic physics-inspired illustrations, not Aspen runs.",
            "All time-series profiles, including the baseline profile, are synthetic because measured profiles were not supplied.",
            "Do not report validation scores on this dataset as real Aspen prediction accuracy. Replace with actual Aspen runs before engineering use.",
        ],
    }
    (DATA_DIR / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Generated {len(runs)} cases and {len(profiles)} profile points.")
    print(runs.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
