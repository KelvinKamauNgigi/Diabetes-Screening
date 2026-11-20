"""MLflow "models-from-code" definition of the screening model, so the registry holds a loadable, versioned, signed artefact.

Input  : one row per person with the same friendly fields as POST /predict (only age, sex, residence are required).
Output : risk_probability, decision, reliability, measurements_supplied  (identical to the API, because it calls the same Scorer).
"""
from __future__ import annotations
import math, os
import mlflow
import pandas as pd
from mlflow.models import set_model
from mlflow.types import ColSpec, Schema

INT_FIELDS = {"age", "years_schooling"}
BOOL_FIELDS = {"on_bp_medication", "bp_ever_measured", "told_raised_bp", "cholesterol_ever_measured", "history_heart_attack_or_stroke", "current_smoker", "ever_drank_alcohol"}
FLOAT_FIELDS = {"height_cm", "weight_kg", "waist_cm", "hip_cm", "systolic_bp", "diastolic_bp", "heart_rate"}
STR_FIELDS = {"education", "marital_status"}


def input_schema() -> Schema:
    cols = [ColSpec("long", "age"), ColSpec("string", "sex"), ColSpec("string", "residence")]
    cols += [ColSpec("double", f, required=False) for f in sorted(FLOAT_FIELDS)]
    cols += [ColSpec("boolean", f, required=False) for f in sorted(BOOL_FIELDS)]
    cols += [ColSpec("double", "years_schooling", required=False)]       # whole-number fields are declared double so blanks (NaN) are accepted: pass 8.0, not 8
    cols += [ColSpec("string", f, required=False) for f in sorted(STR_FIELDS)]
    return Schema(cols)


def _clean(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA: continue
        out[k] = int(v) if k in INT_FIELDS else bool(v) if k in BOOL_FIELDS else float(v) if k in FLOAT_FIELDS else str(v)
    return out


class ScreeningModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        os.environ["MODEL_DIR"] = context.artifacts["model_dir"]            # read by app.engine at import time
        from app.service import Scorer
        self.scorer = Scorer(context.artifacts["model_dir"])

    def predict(self, context, model_input, params=None):
        from app.schemas import Patient
        rows = []
        for r in pd.DataFrame(model_input).to_dict("records"):
            o = self.scorer.predict(Patient(**_clean(r)))
            rows.append({k: o[k] for k in ("risk_probability", "decision", "reliability", "measurements_supplied")})
        return pd.DataFrame(rows)


set_model(ScreeningModel())
