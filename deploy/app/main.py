from __future__ import annotations
import logging, os, time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .schemas import BatchRequest, Patient, Prediction
from .service import Scorer
from .telemetry import Telemetry

log = logging.getLogger("diabetes-api"); logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).resolve().parent / "static"
@asynccontextmanager
async def lifespan(_):
    global scorer
    scorer = Scorer(); log.info("model '%s' loaded, %d features, referral threshold %.4f", scorer.eng.champion, len(scorer.eng.features), scorer.threshold)
    yield

app = FastAPI(lifespan=lifespan, title="Diabetes Screening API (WHO STEPS Kenya 2015)", version="1.0.0",
              description="Scores adults for the probability of diabetes (fasting glucose ≥ 7.0 mmol/L or on treatment) from cheap, non-laboratory inputs. Research prototype.")
scorer: Scorer | None = None
telemetry = Telemetry()
API_KEY = os.environ.get("API_KEY")            # optional: set to require the X-API-Key header on /predict

def guard(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY: raise HTTPException(401, "missing or invalid X-API-Key")

@app.get("/health", tags=["ops"])
def health(): return {"status": "ok" if scorer else "starting", "model_loaded": scorer is not None}

@app.get("/model-info", tags=["ops"])
def model_info():
    c = scorer.eng.card; ops = scorer.ops
    return {"champion": scorer.eng.champion, "n_features": len(scorer.eng.features), "referral_threshold": scorer.threshold,
            "assumed_cost_ratio_FN_to_FP": scorer.eng.bundle_info["cost_ratio"][0], "validation": c["validation"].get(scorer.eng.champion), "test": c["test"].get(scorer.eng.champion + " ← CHAMPION"),
            "baselines_test": {k: v for k, v in c["test"].items() if k.startswith(("Dummy", "Clinical"))},
            "operating_points": {k: v for k, v in ops.items() if k in ("frozen_refer_threshold", "elevated_threshold_80pct_sens_on_validation", "n_pos")},
            "base_rate_in_training_survey": scorer.base,
            "limitations": ["Trained on 2,857 people with 81 positives; held-out sets have 17-18 positives each, so all metrics are imprecise.",
                            "Discrimination is moderate (test ROC-AUC ≈ 0.69); age and sex alone give almost the same ranking as the full model.",
                            "At the frozen threshold most diabetics are NOT flagged (test sensitivity about one in three); it is a high-specificity rule.",
                            "Cross-sectional 2015 Kenyan survey; no external validation."]}

@app.get("/metrics-summary", tags=["ops"])
def metrics_summary():
    """Aggregate serving statistics (counts, referral rate, risk histogram, latency). Contains no patient data; polled by scripts/log_serving_stats.py."""
    return telemetry.snapshot(scorer.base, scorer.threshold)

@app.get("/schema", tags=["ops"])
def schema():
    return {"required": ["age", "sex", "residence"], "key_measurements": ["height_cm", "weight_kg", "waist_cm", "hip_cm", "systolic_bp", "heart_rate"],
            "education_levels": scorer.eng.meta["string_levels"]["C5_Highest_Education"], "marital_status_levels": scorer.eng.meta["string_levels"]["C7_Marital_Status"],
            "extra_item_codes": scorer.extra, "note": "extra_items values: numbers as coded in the STEPS questionnaire, or true/false for yes/no items"}

@app.post("/predict", response_model=Prediction, tags=["scoring"], dependencies=[Depends(guard)])
def predict(patient: Patient):
    t0 = time.perf_counter()
    try: out = scorer.predict(patient)
    except ValueError as e: telemetry.error(); raise HTTPException(422, str(e))
    telemetry.record(out["risk_probability"], out["decision"], out["measurements_supplied"], out["reliability"], time.perf_counter() - t0)
    log.info("scored: decision=%s p=%.4f measurements=%s", out["decision"], out["risk_probability"], out["measurements_supplied"])   # no inputs are logged or stored
    return out

@app.post("/predict/batch", tags=["scoring"], dependencies=[Depends(guard)])
def predict_batch(req: BatchRequest):
    res = []
    for i, p in enumerate(req.patients):
        t0 = time.perf_counter()
        try: o = scorer.predict(p)
        except ValueError as e: telemetry.error(); raise HTTPException(422, f"patient {i}: {e}")
        telemetry.record(o["risk_probability"], o["decision"], o["measurements_supplied"], o["reliability"], time.perf_counter() - t0); res.append(o)
    return {"n": len(res), "results": res}

app.mount("/static", StaticFiles(directory=STATIC), name="static")
@app.get("/", include_in_schema=False)
def ui(): return FileResponse(STATIC / "index.html")
