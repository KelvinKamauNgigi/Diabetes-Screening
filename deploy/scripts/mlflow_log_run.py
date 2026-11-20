#!/usr/bin/env python
"""Record the trained model in MLflow: parameters, metrics, artefacts, a signed pyfunc model and a registry entry with a `champion` alias.

This back-fills the run from the saved bundle and model card, so it needs neither the training data nor the notebook.
To log a *live* training run (with the cross-validation search results), use scripts/notebook_mlflow_cell.py inside the notebook instead.

    python scripts/mlflow_log_run.py                      # local sqlite store in ./mlruns
    MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/mlflow_log_run.py   # the docker-compose server
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, tempfile, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

DESCRIPTION = ("Diabetes screening model trained on the WHO STEPS Kenya 2015 survey (4,090 labelled adults, 116 positives = 2.8%). Target: on diabetes medication or "
               "fasting plasma glucose >= 7.0 mmol/L. PSU-grouped, stratified 70/15/15 split; leakage-prone items excluded; all fitting on the training split only.")
HP_KEYS = {"Elastic-net LR": ["C", "l1_ratio", "penalty", "solver", "max_iter", "class_weight"],
           "Random forest": ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "max_samples", "class_weight"],
           "LightGBM": ["n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree", "reg_lambda", "scale_pos_weight"]}
EXAMPLE = pd.DataFrame([{"age": 54, "sex": "female", "residence": "urban", "height_cm": 158.0, "weight_kg": 82.0, "waist_cm": 102.0, "hip_cm": 111.0,
                         "systolic_bp": 146.0, "diastolic_bp": 92.0, "heart_rate": 84.0, "told_raised_bp": True}])


def norm(name: str) -> str: return name.replace(" ← CHAMPION", "").strip()
def slug(name: str) -> str: return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def champion_coefficients(bundle) -> pd.DataFrame | None:
    """Mean standardised coefficient across the calibration folds' fitted pipelines (only meaningful for the linear champion)."""
    m = bundle["calibrated_models"][bundle["champion"]]; rows = []
    for cc in m.calibrated_classifiers_:
        pipe = cc.estimator
        if not hasattr(pipe["model"], "coef_"): return None
        rows.append(pd.Series(pipe["model"].coef_[0], index=pipe["prep"].get_feature_names_out()))
    d = pd.concat(rows, axis=1); return pd.DataFrame({"coef_mean": d.mean(axis=1), "coef_sd_across_folds": d.std(axis=1)}).sort_values("coef_mean", key=abs, ascending=False)


def metric_figure(card: dict, champion: str):
    names = [n for n in card["validation"]]; fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for a, met in zip(ax, ("ROC-AUC", "PR-AUC")):
        v = [card["validation"][n][met] for n in names]; t = [card["test"].get(n, card["test"].get(n + " ← CHAMPION"))[met] for n in names]
        x = np.arange(len(names)); a.bar(x - 0.2, v, 0.4, label="validation"); a.bar(x + 0.2, t, 0.4, label="test")
        a.set_xticks(x); a.set_xticklabels([n.replace(" (", "\n(") for n in names], rotation=35, ha="right", fontsize=8); a.set_title(met); a.legend(fontsize=8)
    fig.suptitle("17 validation / 18 test positives: differences between models are within noise", fontsize=9); fig.tight_layout(); return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "model")); ap.add_argument("--experiment", default="diabetes-screening-steps2015")
    ap.add_argument("--registered-name", default="diabetes-screening-steps2015"); ap.add_argument("--run-name", default=None)
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}"))
    a = ap.parse_args(); md = Path(a.model_dir)
    if a.tracking_uri.startswith("sqlite:///"): Path(a.tracking_uri[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(a.tracking_uri); mlflow.set_experiment(a.experiment)

    bundle = joblib.load(md / "model_bundle.joblib"); card = json.load(open(md / "model_card.json")); ops = json.load(open(md / "operating_points.json")); meta = json.load(open(md / "meta.json"))
    champ = bundle["champion"]; val = card["validation"]; test = {norm(k): v for k, v in card["test"].items()}
    sha = hashlib.sha256((md / "model_bundle.joblib").read_bytes()).hexdigest()

    with mlflow.start_run(run_name=a.run_name or f"final-{slug(champ)}-seed{bundle['split_seed']}", description=DESCRIPTION) as parent:
        mlflow.set_tags({"stage": "final", "source": "backfill-from-bundle", "champion": champ, "dataset": "WHO STEPS Kenya 2015", "unit_of_split": "PSU (cluster)",
                         "target": "diabetes_medication OR fasting_glucose>=7.0", "python": platform.python_version(), "bundle_sha256": sha[:16]})
        mlflow.log_params({"split_seed": bundle["split_seed"], "split": "70/15/15 StratifiedGroupKFold(20) by PSU", "n_train": meta["n_train"], "n_features": card["n_features"],
                           "cost_ratio_FN_to_FP": f"{bundle['cost_ratio'][0]}:{bundle['cost_ratio'][1]}", "referral_threshold": round(float(bundle["threshold"]), 6),
                           "calibration": "Platt (sigmoid) on out-of-fold predictions", "sklearn": bundle["sklearn"], "bundle_sha256": sha,
                           "n_positives_val": ops["n_pos"]["validation"], "n_positives_test": ops["n_pos"]["test"], "train_base_rate": round(meta["base_rate_train"], 5)})
        for split, tbl in (("val", val), ("test", test)):
            for met, v in tbl[champ].items(): mlflow.log_metric(f"{split}_{slug(met)}", v)
        for split in ("validation", "test"):
            for k, v in ops["frozen_refer_threshold"][split].items():
                if k != "threshold": mlflow.log_metric(f"{split[:4] if split=='test' else 'val'}_frozen_{k}", v)
            for k, v in ops["elevated_threshold_80pct_sens_on_validation"][split].items():
                if k != "threshold": mlflow.log_metric(f"{'test' if split=='test' else 'val'}_elev80_{k}", v)
        mlflow.log_metric("elev80_threshold", ops["elevated_threshold_80pct_sens_on_validation"]["validation"]["threshold"])

        # one nested run per candidate model, so they can be compared side by side in the UI
        for name in val:
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tags({"model": name, "is_champion": str(name == champ), "kind": "baseline" if name.startswith(("Dummy", "Clinical")) else "candidate"})
                if name in bundle["calibrated_models"]:
                    est = bundle["calibrated_models"][name].calibrated_classifiers_[0].estimator["model"]; hp = est.get_params()
                    mlflow.log_params({f"hp_{k}": hp[k] for k in HP_KEYS.get(name, []) if k in hp})
                for met, v in val[name].items(): mlflow.log_metric(f"val_{slug(met)}", v)
                for met, v in test.get(name, {}).items(): mlflow.log_metric(f"test_{slug(met)}", v)

        # artefacts
        for f in ("model_card.json", "operating_points.json", "meta.json"): mlflow.log_artifact(str(md / f), "model_documents")
        mlflow.log_text("\n".join(bundle["features"]), "model_documents/features.txt")
        mlflow.log_dict({"dropped_80pct_missing": bundle["drop80"], "structural_missingness_rules": {k: [v[0], [x if isinstance(x, str) else float(x) for x in v[1]], v[2]] for k, v in bundle["struct_rules"].items()}},
                        "model_documents/missingness_decisions.json")
        co = champion_coefficients(bundle)
        if co is not None:
            mlflow.log_text(co.round(5).to_csv(), "model_documents/champion_coefficients.csv")
        fig = metric_figure(card, champ); mlflow.log_figure(fig, "figures/model_comparison.png"); plt.close(fig)

        # signed, registered model (the same Scorer the API runs)
        out_example = None
        from mlops.screening_model import input_schema
        from mlflow.models.signature import ModelSignature
        from mlflow.types import ColSpec, Schema
        sig = ModelSignature(inputs=input_schema(), outputs=Schema([ColSpec("double", "risk_probability"), ColSpec("string", "decision"), ColSpec("string", "reliability"), ColSpec("string", "measurements_supplied")]))
        info = mlflow.pyfunc.log_model(name="model", python_model=str(ROOT / "mlops" / "screening_model.py"), artifacts={"model_dir": str(md)}, code_paths=[str(ROOT / "app"), str(ROOT / "mlops")],
                                       signature=sig, input_example=EXAMPLE, registered_model_name=a.registered_name,
                                       pip_requirements=["scikit-learn==" + bundle["sklearn"], "pandas==2.2.3", "numpy==2.1.0", "scipy==1.18.1", "lightgbm==4.5.0", "joblib==1.6.0", "pydantic==2.14.4"])
        client = MlflowClient(); v = info.registered_model_version
        client.set_registered_model_alias(a.registered_name, "champion", v)
        client.set_model_version_tag(a.registered_name, v, "test_roc_auc", str(test[champ]["ROC-AUC"])); client.set_model_version_tag(a.registered_name, v, "test_pr_auc", str(test[champ]["PR-AUC"]))
        client.set_model_version_tag(a.registered_name, v, "caveat", "18 test positives; discrimination moderate; research prototype, not a diagnostic device")
        client.update_registered_model(a.registered_name, description=DESCRIPTION)

        # smoke test: load back from the registry alias and score the example
        loaded = mlflow.pyfunc.load_model(f"models:/{a.registered_name}@champion"); pred = loaded.predict(EXAMPLE)
        mlflow.log_metric("registry_smoke_test_risk", float(pred.risk_probability.iloc[0]))
        print(f"run_id={parent.info.run_id}  registered {a.registered_name} v{v} @champion  | example risk={pred.risk_probability.iloc[0]:.4f} ({pred.decision.iloc[0]})")
    print(f"tracking store: {a.tracking_uri}\nview with:  mlflow ui --backend-store-uri {a.tracking_uri}")


if __name__ == "__main__": main()
