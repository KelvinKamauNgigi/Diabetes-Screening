"""Aggregate-only serving telemetry (no inputs, no identifiers are ever kept): counts, a risk histogram and latency.

Counters live in process memory, so the container runs a single uvicorn worker; `scripts/log_serving_stats.py` polls
GET /metrics-summary and writes the deltas to MLflow so drift against the training survey can be tracked over time.
"""
from __future__ import annotations
import threading, time
from collections import Counter

RISK_BINS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.0909, 0.15, 0.25, 1.0001]


class Telemetry:
    def __init__(self) -> None:
        self._lock = threading.Lock(); self.started = time.time(); self.reset()

    def reset(self) -> None:
        self.n = 0; self.refer = 0; self.risk_sum = 0.0; self.lat_sum = 0.0; self.lat_max = 0.0; self.errors = 0
        self.meas = Counter(); self.bins = Counter(); self.reliability = Counter()

    def record(self, prob: float, decision: str, n_meas: str, reliability: str, seconds: float) -> None:
        k = int(str(n_meas).split(" ")[0])
        b = next(i for i in range(len(RISK_BINS) - 1) if RISK_BINS[i] <= prob < RISK_BINS[i + 1])
        with self._lock:
            self.n += 1; self.risk_sum += prob; self.lat_sum += seconds; self.lat_max = max(self.lat_max, seconds)
            self.refer += decision == "refer_for_confirmatory_test"; self.meas[k] += 1; self.bins[b] += 1; self.reliability[reliability] += 1

    def error(self) -> None:
        with self._lock: self.errors += 1

    def snapshot(self, base_rate: float, threshold: float) -> dict:
        with self._lock:
            n = self.n
            return {"since_unix": self.started, "uptime_s": round(time.time() - self.started, 1), "requests_scored": n, "validation_errors": self.errors,
                    "referred": self.refer, "referral_rate": (self.refer / n) if n else None,
                    "mean_risk": (self.risk_sum / n) if n else None, "training_base_rate": base_rate, "referral_threshold": threshold,
                    "mean_latency_ms": (1000 * self.lat_sum / n) if n else None, "max_latency_ms": 1000 * self.lat_max,
                    "measurements_supplied": {str(k): self.meas.get(k, 0) for k in range(7)},
                    "risk_histogram": [{"lo": RISK_BINS[i], "hi": min(RISK_BINS[i + 1], 1.0), "count": self.bins.get(i, 0)} for i in range(len(RISK_BINS) - 1)],
                    "reliability": dict(self.reliability),
                    "note": "aggregates only; no patient inputs are stored"}
