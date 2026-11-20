from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from .engine import Engine, MODEL_DIR
from .schemas import Patient

DISCLAIMER = ("Research prototype built from the WHO STEPS Kenya 2015 survey for a university capstone. It estimates population-level risk, is not a diagnostic "
              "device, and must not replace a fasting plasma glucose test or clinical judgement.")
CORE_MEASURES = ["height_cm", "weight_kg", "waist_cm", "hip_cm", "systolic_bp", "heart_rate"]

class Scorer:
    def __init__(self, model_dir: Path | None = None):
        model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.eng = Engine(model_dir)
        self.ops = json.load(open(model_dir / "operating_points.json"))
        self.q = np.array(self.ops["reference_quantiles"]); self.base = float(self.ops["base_rate"])
        self.threshold = float(os.environ.get("REFERRAL_THRESHOLD", self.eng.threshold))
        self.extra = self.eng.allowed_extra_items()
        self.allowed_obj = {c for c in self.eng.obj_cols} | {"C1_Sex"}

    # ---- friendly payload -> the "clean" questionnaire frame the notebook's pipeline expects ----
    def to_frame(self, p: Patient):
        e = self.eng; col = e.col; F = e.blank_frame(); supplied: set[str] = set(); used: dict[str, object] = {}
        def put(code, val):
            c = col(code); F.at[0, c] = (float(val) if not isinstance(val, str) else val); supplied.add(code); used[code] = val
        yn = lambda b: 1.0 if b else 0.0
        put("C3", p.age); put("C1", "Women" if p.sex == "female" else "Men"); F.at[0, "Female"] = yn(p.sex == "female"); F.at[0, "Urban"] = yn(p.residence == "urban")
        supplied |= {"X2a"}
        for code, v in (("M11", p.height_cm), ("M12", p.weight_kg), ("M14", p.waist_cm), ("M15", p.hip_cm)):
            if v is not None: put(code, v)
        if p.systolic_bp is not None:
            for c in ("M4a", "M5a", "M6a"): put(c, p.systolic_bp)
        if p.diastolic_bp is not None:
            for c in ("M4b", "M5b", "M6b"): put(c, p.diastolic_bp)
        if p.heart_rate is not None:
            for c in ("M16a", "M16b", "M16c"): put(c, p.heart_rate)
        for code, v in (("M7", p.on_bp_medication), ("H12", p.cholesterol_ever_measured), ("H17", p.history_heart_attack_or_stroke),
                        ("T1", p.current_smoker), ("A1", p.ever_drank_alcohol)):
            if v is not None: put(code, yn(v))
        if p.bp_ever_measured is not None: put("H1", yn(p.bp_ever_measured))
        elif p.told_raised_bp is not None: put("H1", 1.0)
        if p.told_raised_bp is not None: put("H2a", yn(p.told_raised_bp))
        if p.years_schooling is not None: put("C4", p.years_schooling)
        if p.education is not None: put("C5", p.education)
        if p.marital_status is not None: put("C7", p.marital_status)
        for raw_code, v in p.extra_items.items():
            code = raw_code.strip()
            code = next((k for k in self.extra if k.lower() == code.lower()), None)
            if code is None: raise ValueError(f"'{raw_code}' is not a model input (unknown item, or excluded to prevent target leakage)")
            if code in supplied: raise ValueError(f"'{raw_code}' is already set by a named field")
            c = col(code)
            if c in self.allowed_obj:
                if not isinstance(v, str): raise ValueError(f"'{raw_code}' expects text")
                put(code, v)
            else:
                if isinstance(v, str): v = {"yes": 1.0, "no": 0.0}.get(v.lower(), None)
                if v is None: raise ValueError(f"'{raw_code}' expects a number or true/false")
                v = float(v)
                if c in e.bin_cols and v not in (0.0, 1.0): raise ValueError(f"'{raw_code}' is a yes/no item: use true/false")
                put(code, v)
        return F, supplied

    def warnings_for(self, p: Patient, feats: pd.Series, n_meas: int) -> list[str]:
        w = []; tr = self.eng.meta["train_ranges"]
        if p.age > 70 or p.age < 18: w.append("Age is outside the range the model was trained on (16-70); treat the estimate with extra caution.")
        for k, lab in (("bmi", "BMI"), ("waist", "Waist"), ("whr", "Waist-to-hip ratio"), ("sbp", "Systolic BP"), ("hr", "Heart rate")):
            v = feats.get(k)
            if pd.notna(v) and (v < tr[k][0] or v > tr[k][1]): w.append(f"{lab} ({v:.1f}) is outside the central 98% of the training survey; the estimate is an extrapolation.")
        if n_meas < 3: w.append("Fewer than 3 of the 6 key measurements were supplied: the estimate then rests mostly on age and sex and cannot be personalised.")
        return w

    def predict(self, p: Patient) -> dict:
        F, supplied = self.to_frame(p)
        X = self.eng.features_from_clean(F); prob = float(self.eng.predict_features(X)[0])
        n_meas = sum(v is not None for v in (p.height_cm, p.weight_kg, p.waist_cm, p.hip_cm, p.systolic_bp, p.heart_rate))
        pct = float((self.q <= prob).mean() * 100)
        row = X.iloc[0]
        return {
            "risk_probability": round(prob, 4), "risk_percent": round(100 * prob, 1),
            "decision": "refer_for_confirmatory_test" if prob >= self.threshold else "below_referral_threshold",
            "referral_threshold": round(self.threshold, 4), "times_average_risk": round(prob / self.base, 2), "percentile_vs_survey": round(pct, 0),
            "measurements_supplied": f"{n_meas} of {len(CORE_MEASURES)}", "reliability": "low" if n_meas < 3 else ("moderate" if n_meas < 6 else "higher"),
            "derived": {k: (None if pd.isna(row[k]) else round(float(row[k]), 2)) for k in ("bmi", "whr", "sbp", "dbp", "pulse_pressure", "hr")},
            "drivers": self.eng.drivers(X, supplied), "warnings": self.warnings_for(p, row, n_meas),
            "model": {"name": self.eng.champion, "training_data": "WHO STEPS Kenya 2015", "sklearn": self.eng.bundle_info["sklearn"]}, "disclaimer": DISCLAIMER}
