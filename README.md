# PolyMMA Surrogate Model — Demonstration Package

> **IMPORTANT DATA LABEL:** This package is a proof-of-concept using synthetic, physics-inspired illustrative data. It is **not trained on 50 Aspen Plus runs**. The only measured reference point used is the R-101 baseline reported in Bi-Weekly Report No. 6. Do not present its predictions or validation metrics as experimentally/Aspen-validated results, and do not use it for engineering or operating decisions.

## What is included

- `data/synthetic_aspen_runs.csv` — 50 operating-condition cases and final KPI values. One row is anchored to the reported baseline; the other 49 are synthetic.
- `data/synthetic_time_profiles.csv` — 41 conversion/energy profile points for each case. **All** profiles, including the baseline profile, are synthetic because the measured time-series data was not supplied.
- `data/dataset_metadata.json` — source labels, input ranges and limitations.
- `train_model.py` — trains Gaussian Process Regression models for the final KPIs and curve parameters.
- `models/polymma_surrogate.joblib` — trained model bundle.
- `models/training_metrics.json` — 20% hold-out metrics on the synthetic dataset only.
- `api.py` — FastAPI service with `/predict`, `/health` and `/metadata` endpoints.

## Confirmed baseline anchor

The project report gives the R-101 baseline as:

| Variable | Value |
|---|---:|
| Initial temperature | 40 °C |
| Target temperature | 105 °C |
| Batch duration | 2.00 h |
| MMA conversion | 89.91% |
| Resin yield | 3,034.5 kg/batch |
| DPn | 71.14 |
| DPw | 227.26 |
| PDI | 3.195 |
| Cumulative heat duty | 0.508 MMkcal/batch |

The baseline initiator injection time used in this demo is 1.01 h, based on the project's previously documented R-101 setup. Other fixed baseline conditions (e.g. 3 bar pressure, 0.30 h temperature ramp, 3,000 kg/h initiator pulse for 0.01 h) are not model inputs in this first prototype.

The report also contains an example comparison case (87.95% conversion, 2,968.9 kg resin, PDI 3.200, 0.469 MMkcal), but it does not specify the corresponding four input settings. That unpaired example was therefore **not** used as a training observation.

## 1. Run on Linux (e.g. Arch) or macOS

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_synthetic_data.py
python train_model.py
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` to test the API interactively.

## 2. Run on Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
pip install -r requirements.txt
py generate_synthetic_data.py
py train_model.py
py -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The API does not require Aspen to be installed. It loads the saved ML artifact.

## 3. Test a prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "initial_temp_C": 44,
    "target_temp_C": 95,
    "initiator_time_h": 1.10,
    "batch_duration_h": 1.6,
    "n_points": 41
  }'
```

The response includes:

- `outputs`: conversion, resin yield, DPn, DPw, PDI and heat duty
- `conversion_profile`: time/conversion points for the graph
- `energy_profile`: time/cumulative-energy points
- `uncertainty_estimate`: approximate GPR posterior standard deviations
- `result_type`: always labels the result as a synthetic surrogate prediction

## 4. Frontend integration example

```javascript
async function predictPolyMMA(values) {
  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      initial_temp_C: values.initialTemperature,
      target_temp_C: values.targetTemperature,
      initiator_time_h: values.initiatorTime,
      batch_duration_h: values.batchDuration,
      n_points: 41
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail ?? "Prediction failed");
  }
  return await response.json();
}
```

Call this function when the user changes a slider (preferably debounce requests by 200–400 ms). Use `result.conversion_profile` as the graph's x/y series and display `result.outputs` in the KPI cards. Keep the `result_type`/warning visible during the demo.

## 5. Replace synthetic data with real Aspen runs

For a defensible surrogate model:

1. Run Aspen at a space-filling set of operating conditions (Latin Hypercube or a designed experiment), with all other process settings held consistent.
2. Export one row per simulation with the four inputs and final KPIs.
3. Export conversion/time and cumulative-energy/time profiles for every run, using consistent definitions and units.
4. Replace the synthetic CSVs with the measured Aspen data, preserving column names.
5. Re-run `python train_model.py`.
6. Evaluate the model on held-out **Aspen runs**, not random time points from the same run. Add more Aspen runs where error or prediction uncertainty is high.

With only 30–50 real runs, model accuracy is not guaranteed. Gaussian Processes are selected as a reasonable small-data baseline, but the model must be checked against held-out simulation cases before claiming reliable predictions.

## Assumptions and limits

- This initial model uses four inputs: initial temperature, target temperature, initiator injection time and batch duration.
- Pressure, ramp duration, initiator mass flow and pulse duration are held fixed at the project's baseline settings and are not exposed as inputs.
- The generated response surface is deliberately smooth and plausible-looking for a tomorrow-facing UI concept; it is not derived from polymerisation kinetics or a validated first-principles model.
- The synthetic profile reconstructs a smooth conversion curve from model-predicted final conversion, time constant and shape. It is not an Aspen-exported conversion trajectory.
- Do not call the synthetic optimum an Aspen-optimised configuration. Use the real sensitivity-analysis optimum once its exact four input settings and outputs are supplied.
