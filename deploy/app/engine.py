"""Inference engine: turns questionnaire answers into the exact feature frame the notebook built, then scores it.

The functions `apply_structural`, `build_features` and `finalize` are the notebook's own code (sections 8.2), parameterised by the
metadata saved with the model, so training-time and serving-time preprocessing cannot drift apart.
"""
from __future__ import annotations
import json, os, re, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd

warnings.filterwarnings("ignore")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))

# which raw questionnaire items each engineered feature is built from (used only to explain a score)
FEATURE_SOURCES = {
    "age": ["C3"], "female": ["C1"], "urban": ["X2a"], "schooling_years": ["C4"], "bmi": ["M11", "M12"], "waist": ["M14"], "whr": ["M14", "M15"],
    "sbp": ["M4a", "M5a", "M6a"], "dbp": ["M4b", "M5b", "M6b"], "hr": ["M16a", "M16b", "M16c"], "pulse_pressure": ["M4a", "M5a", "M6a", "M4b", "M5b", "M6b"],
    "met_min_week": ["P2", "P3a", "P3b", "P5", "P6a", "P6b", "P8", "P9a", "P9b"], "sitting_min_day": ["P16a", "P16b"],
    "fruit_veg_servings_day": ["D1", "D2", "D3", "D4"],
}
LABELS = {
    "age": "Age", "female": "Sex (female)", "urban": "Urban residence", "bmi": "BMI", "waist": "Waist circumference", "whr": "Waist-to-hip ratio",
    "sbp": "Systolic blood pressure", "dbp": "Diastolic blood pressure", "hr": "Resting heart rate", "pulse_pressure": "Pulse pressure",
    "schooling_years": "Years of schooling", "met_min_week": "Physical activity", "sitting_min_day": "Sitting time", "fruit_veg_servings_day": "Fruit & vegetable intake",
}


class Engine:
    def __init__(self, model_dir: Path = MODEL_DIR):
        b = joblib.load(model_dir / "model_bundle.joblib")
        self.meta = json.load(open(model_dir / "meta.json")); self.card = json.load(open(model_dir / "model_card.json"))
        self.bundle_info = {k: b[k] for k in ("champion", "threshold", "cost_ratio", "split_seed", "sklearn")}
        self.champion, self.threshold = b["champion"], float(b["threshold"])
        self.models = b["calibrated_models"]
        for m in self.models.values():                       # the bundle was saved with a Windows joblib cache path; serving never fits, so drop it
            for cc in m.calibrated_classifiers_: cc.estimator.memory = None
        self.features = list(b["features"]); self.struct_rules = b["struct_rules"]; self.universe = list(b["universe"])
        self.orig_cols = self.meta["ORIG_COLS"]
        self.level_cols, self.obj_cols = self.meta["LEVEL_COLS"], self.meta["OBJ_COLS"]
        self.cat_num, self.bin_cols, self.consumed = self.meta["CAT_NUM"], set(self.meta["BIN_COLS"]), set(self.meta["CONSUMED"])
        self._code2col = {}
        for c in self.orig_cols:
            self._code2col.setdefault(c.split("_")[0], []).append(c)
        est = self._lr_pipeline()
        self._prep, self._coef = (est["prep"], est["model"].coef_[0]) if est is not None else (None, None)
        self._tnames = list(self._prep.get_feature_names_out()) if est is not None else []

    # ---------- schema helpers ----------
    def col(self, code: str) -> str:
        m = self._code2col.get(code, [])
        if len(m) != 1: raise KeyError(f"unknown questionnaire item '{code}'")
        return m[0]

    def allowed_extra_items(self) -> dict[str, str]:
        """questionnaire code -> column name for every item that is a legitimate model input"""
        return {c.split("_")[0]: c for c in self.universe if c.split("_")[0] in self._code2col and len(self._code2col[c.split("_")[0]]) == 1}

    def _lr_pipeline(self):
        try:
            return self.models[self.champion].calibrated_classifiers_[0].estimator
        except Exception:
            return None

    # ---------- notebook code (section 8.2), verbatim logic ----------
    def blank_frame(self) -> pd.DataFrame:
        cols = self.universe + ["Female", "Urban"]
        data = {c: pd.Series([np.nan], dtype=object if (c in self.obj_cols or c == "C1_Sex") else "float64") for c in cols}
        return pd.DataFrame(data)

    def _apply_structural(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        for c, (g, S, kind) in self.struct_rules.items():
            m = d[c].isna() & d[g].isin(S)
            d.loc[m, c] = 0.0 if kind == "zero" else -1.0
        return d

    def build_features(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        col = self.col
        d = self._apply_structural(df_clean[self.universe + ["Female", "Urban"]])
        f = pd.DataFrame(index=d.index)
        f["age"], f["female"], f["urban"], f["schooling_years"] = d["C3_Age_Years"], d["Female"], d["Urban"], d["C4_Years_Of_Schooling"]
        h, w, wa, hi = d[col("M11")], d[col("M12")], d[col("M14")], d[col("M15")]
        f["bmi"] = (w / (h / 100) ** 2).where(lambda s: s.between(12, 70)); f["waist"] = wa; f["whr"] = (wa / hi).where(lambda s: s.between(0.5, 1.5))
        C = lambda *codes: [col(c) for c in codes]
        f["sbp"] = d[C("M5a", "M6a")].mean(axis=1).fillna(d[col("M4a")]); f["dbp"] = d[C("M5b", "M6b")].mean(axis=1).fillna(d[col("M4b")])
        f["hr"] = d[C("M16a", "M16b", "M16c")].mean(axis=1); f["pulse_pressure"] = f["sbp"] - f["dbp"]
        f["step2_incomplete"] = f[["bmi", "waist", "whr", "sbp", "dbp"]].isna().any(axis=1).astype(float)
        mins = lambda hc, mc: d[col(hc)] * 60 + d[col(mc)]
        vig, mod, trv = d[col("P2")] * mins("P3a", "P3b"), d[col("P5")] * mins("P6a", "P6b"), d[col("P8")] * mins("P9a", "P9b")
        f["met_min_week"] = 8 * vig + 4 * (mod + trv); f["sitting_min_day"] = mins("P16a", "P16b")
        f["fruit_veg_servings_day"] = (d[col("D1")] * d[col("D2")] + d[col("D3")] * d[col("D4")]) / 7
        for c in self.universe:
            if c not in self.consumed: f[c] = d[c]
        return f

    def finalize(self, f: pd.DataFrame) -> pd.DataFrame:
        f = f.copy()
        for c in self.level_cols:
            isbin = c in self.bin_cols
            f[c] = f[c].map(lambda v: np.nan if pd.isna(v) else ("not_applicable" if v == -1 else (("yes" if v == 1 else "no") if isbin else str(int(v)))))
        for c in self.cat_num: f[c] = f[c].map(lambda v: np.nan if pd.isna(v) else str(int(v)))
        for c in self.obj_cols: f[c] = f[c].astype(object).where(f[c].notna(), np.nan)
        return f

    def features_from_clean(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        return self.finalize(self.build_features(df_clean))[self.features]

    # ---------- scoring ----------
    def predict_features(self, F: pd.DataFrame) -> np.ndarray:
        if self.champion == "Soft-vote ensemble":
            return np.mean([m.predict_proba(F)[:, 1] for m in self.models.values()], axis=0)
        return self.models[self.champion].predict_proba(F)[:, 1]

    def predict_clean(self, df_clean: pd.DataFrame) -> np.ndarray:
        return self.predict_features(self.features_from_clean(df_clean))

    def _source_of(self, tname: str) -> tuple[str, bool]:
        n = re.sub(r"^(meas|num|ord|bin|cat|ind)__", "", tname); ind = tname.startswith("ind__")
        if ind: n = n.replace("missingindicator_", "")
        best = max((f for f in self.features if n == f or n.startswith(f + "_")), key=len, default=n)
        return best, ind

    def drivers(self, F: pd.DataFrame, supplied_codes: set[str], top: int = 5) -> dict:
        """Per-feature contribution to the log-odds for ONE row, restricted to items the user actually supplied."""
        if self._prep is None: return {"increase": [], "decrease": []}
        x = np.asarray(self._prep.transform(F))[0]; contrib = x * self._coef; agg: dict[str, float] = {}
        for tn, v in zip(self._tnames, contrib):
            src, ind = self._source_of(tn)
            if ind: continue
            agg[src] = agg.get(src, 0.0) + float(v)
        def supplied(feat):
            codes = FEATURE_SOURCES.get(feat, [feat.split("_")[0]])
            return any(c in supplied_codes for c in codes)
        rows = [(LABELS.get(k, re.sub(r"^[A-Z]\d+[a-z]*_", "", k).replace("_", " ")), v) for k, v in agg.items() if supplied(k) and abs(v) > 1e-6]
        rows.sort(key=lambda t: t[1], reverse=True)
        return {"increase": [{"factor": k, "log_odds": round(v, 3)} for k, v in rows if v > 0][:top],
                "decrease": [{"factor": k, "log_odds": round(v, 3)} for k, v in rows[::-1] if v < 0][:top]}
