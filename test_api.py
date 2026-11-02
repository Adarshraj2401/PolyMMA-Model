"""Quick local smoke test; run while the API server is running."""
import json
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8000/predict"
PAYLOAD = {
    "initial_temp_C": 44,
    "target_temp_C": 95,
    "initiator_time_h": 1.10,
    "batch_duration_h": 1.6,
    "n_points": 41,
}
request = urllib.request.Request(
    URL,
    data=json.dumps(PAYLOAD).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.load(response)
except urllib.error.URLError as exc:
    raise SystemExit(f"Could not contact API at {URL}: {exc}. Is uvicorn running?") from exc

assert result["status"] == "ok"
assert len(result["conversion_profile"]) == PAYLOAD["n_points"]
assert abs(result["conversion_profile"][-1]["conversion_pct"] - result["outputs"]["conversion_pct"]) < 1e-5
print(json.dumps({
    "result_type": result["result_type"],
    "outputs": result["outputs"],
    "profile_points": len(result["conversion_profile"]),
    "warning": result["warning"],
}, indent=2))
