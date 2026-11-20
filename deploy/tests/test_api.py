import sys, json
from pathlib import Path
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from app.main import app
from app.engine import Engine

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c: yield c

def test_health_and_ui(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    r = client.get("/"); assert r.status_code == 200 and "Estimate risk" in r.text
    assert client.get("/docs").status_code == 200

def test_engine_matches_notebook_exactly():
    """60 held-out respondents: serving code must reproduce the notebook's calibrated probabilities bit-for-bit."""
    e = Engine(); ref = pd.read_pickle(HERE / "reference_rows.pkl")
    got = e.predict_clean(ref.drop(columns=["_expected_p", "_y"]))
    assert np.abs(got - ref["_expected_p"].values).max() < 1e-9

def test_minimal_and_full_payload(client):
    r = client.post("/predict", json={"age": 45, "sex": "male", "residence": "urban"}); assert r.status_code == 200
    j = r.json(); assert 0 < j["risk_probability"] < 1 and j["reliability"] == "low" and j["warnings"]
    r = client.post("/predict", json={"age": 62, "sex": "female", "residence": "urban", "height_cm": 156, "weight_kg": 88, "waist_cm": 112, "hip_cm": 114,
        "systolic_bp": 158, "diastolic_bp": 96, "heart_rate": 90, "on_bp_medication": True, "bp_ever_measured": True, "told_raised_bp": True})
    j = r.json(); assert r.status_code == 200 and j["decision"] == "refer_for_confirmatory_test" and j["reliability"] == "higher"
    assert j["drivers"]["increase"], "expected at least one risk-raising driver"

def test_monotone_sense(client):
    base = {"age": 40, "sex": "female", "residence": "urban", "height_cm": 160, "weight_kg": 60, "waist_cm": 75, "hip_cm": 95, "systolic_bp": 115, "diastolic_bp": 75, "heart_rate": 70}
    old = dict(base, age=65, waist_cm=105, weight_kg=85)
    assert client.post("/predict", json=old).json()["risk_probability"] > client.post("/predict", json=base).json()["risk_probability"]

@pytest.mark.parametrize("bad", [
    {"age": 5, "sex": "male", "residence": "urban"}, {"age": 40, "sex": "x", "residence": "urban"}, {"age": 40, "sex": "male"},
    {"age": 40, "sex": "male", "residence": "urban", "height_cm": 888},
    {"age": 40, "sex": "male", "residence": "urban", "systolic_bp": 100, "diastolic_bp": 110},
    {"age": 40, "sex": "male", "residence": "urban", "bp_ever_measured": False, "told_raised_bp": True},
    {"age": 40, "sex": "male", "residence": "urban", "surprise_field": 1}])
def test_rejects_bad_input(client, bad): assert client.post("/predict", json=bad).status_code == 422

def test_leakage_items_cannot_be_supplied(client):
    for code in ("B5", "H7a", "H8", "H20d", "PSU_Cluster", "B8csv"):
        r = client.post("/predict", json={"age": 40, "sex": "male", "residence": "urban", "extra_items": {code: 1}})
        assert r.status_code == 422 and "not a model input" in r.json()["detail"], code

def test_extra_items_and_batch(client):
    ok = client.post("/predict", json={"age": 40, "sex": "male", "residence": "rural", "extra_items": {"D8": 3, "P1": True}}); assert ok.status_code == 200
    b = client.post("/predict/batch", json={"patients": [{"age": 30, "sex": "male", "residence": "urban"}, {"age": 60, "sex": "female", "residence": "rural"}]}).json()
    assert b["n"] == 2
    assert client.post("/predict", json={"age": 40, "sex": "male", "residence": "rural", "extra_items": {"M12": 70}}).status_code == 200

def test_model_info(client):
    j = client.get("/model-info").json(); assert j["champion"] == "Elastic-net LR" and abs(j["referral_threshold"] - 1/11) < 1e-6 and j["limitations"]
    s = client.get("/schema").json(); assert "B5" not in s["extra_item_codes"] and "D8" in s["extra_item_codes"]


def test_metrics_summary_is_aggregate_only(client):
    before = client.get("/metrics-summary").json()["requests_scored"]
    client.post("/predict", json={"age": 61, "sex": "female", "residence": "urban", "weight_kg": 90, "height_cm": 160, "waist_cm": 108})
    client.post("/predict", json={"age": 20, "sex": "male", "residence": "rural"})
    client.post("/predict", json={"age": 20, "sex": "male", "residence": "rural", "diastolic_bp": 90, "systolic_bp": 80})          # 422
    s = client.get("/metrics-summary").json()
    assert s["requests_scored"] == before + 2 and s["validation_errors"] >= 1
    assert sum(b["count"] for b in s["risk_histogram"]) == s["requests_scored"] and sum(s["measurements_supplied"].values()) == s["requests_scored"]
    assert set(s) == {"since_unix", "uptime_s", "requests_scored", "validation_errors", "referred", "referral_rate", "mean_risk", "training_base_rate", "referral_threshold",
                      "mean_latency_ms", "max_latency_ms", "measurements_supplied", "risk_histogram", "reliability", "note"}      # aggregate keys only: no per-patient fields
    assert set(s) >= {"referral_rate", "mean_risk", "mean_latency_ms", "training_base_rate"}
