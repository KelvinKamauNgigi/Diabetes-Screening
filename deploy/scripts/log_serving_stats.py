#!/usr/bin/env python
"""Poll the running API's aggregate counters (GET /metrics-summary) and log them to MLflow as a time series.

No patient data passes through here: the API only keeps counts, a risk histogram and latency (see app/telemetry.py).

    python scripts/log_serving_stats.py --api http://localhost:8000 --interval 30          # run until Ctrl-C
    python scripts/log_serving_stats.py --api http://localhost:8000 --once                 # one snapshot (cron-friendly)

Read the drift columns with care: a live demo is not a random sample of Kenyan adults, so `mean_risk_vs_training_base_rate` only means
something once the service sees the population it is meant to screen.
"""
from __future__ import annotations
import argparse, json, os, time, urllib.request
import mlflow


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r: return json.load(r)


def log_snapshot(s: dict, step: int, prev: dict | None) -> None:
    n = s["requests_scored"]; d_n = n - (prev["requests_scored"] if prev else 0)
    m = {"requests_scored": n, "requests_in_interval": d_n, "validation_errors": s["validation_errors"], "referred": s["referred"], "max_latency_ms": s["max_latency_ms"]}
    if n:
        low = s["measurements_supplied"]; share_low = sum(low[k] for k in ("0", "1", "2")) / n
        m.update({"referral_rate": s["referral_rate"], "mean_risk": s["mean_risk"], "mean_risk_vs_training_base_rate": s["mean_risk"] / s["training_base_rate"],
                  "mean_latency_ms": s["mean_latency_ms"], "share_with_fewer_than_3_measurements": share_low})
    mlflow.log_metrics(m, step=step)
    mlflow.log_dict(s, "latest_snapshot.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000"); ap.add_argument("--interval", type=float, default=30.0); ap.add_argument("--once", action="store_true")
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")); ap.add_argument("--experiment", default="diabetes-screening-serving")
    a = ap.parse_args(); mlflow.set_tracking_uri(a.tracking_uri); mlflow.set_experiment(a.experiment)
    info = get(a.api + "/model-info")
    with mlflow.start_run(run_name=f"serving-{time.strftime('%Y%m%d-%H%M')}"):
        mlflow.set_tags({"api": a.api, "model": info["champion"], "stage": "serving"})
        mlflow.log_params({"referral_threshold": info["referral_threshold"], "training_base_rate": info["base_rate_in_training_survey"], "poll_interval_s": a.interval})
        step, prev = 0, None
        while True:
            s = get(a.api + "/metrics-summary"); log_snapshot(s, step, prev); prev = s; step += 1
            print(f"[{time.strftime('%H:%M:%S')}] scored={s['requests_scored']} referral_rate={s['referral_rate']} mean_risk={s['mean_risk']}", flush=True)
            if a.once: break
            time.sleep(a.interval)


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("stopped")
