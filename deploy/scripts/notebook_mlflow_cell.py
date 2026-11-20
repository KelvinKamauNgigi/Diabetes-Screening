# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# MLflow tracking cell — paste into the notebook as the LAST code cell (after the cell that writes diabetes_model/model_bundle.joblib).
# It records this training run: every cross-validation candidate, the tuned hyper-parameters, validation/test metrics, the
# frozen threshold, the bundle itself, and registers the champion. Needs:  pip install mlflow
# Re-run the notebook and this cell again, and MLflow lists the new run next to the old one for comparison.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
import hashlib, os, platform, tempfile
import mlflow
from pathlib import Path

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")   # docker-compose server: http://localhost:5000
if TRACKING_URI.startswith("sqlite:///"): Path(TRACKING_URI[10:]).parent.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("diabetes-screening-steps2015")

def _clean(d): return {k.replace("model__", "").replace("param_", ""): (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in d.items()}
def _slug(s): return "".join(ch if ch.isalnum() else "_" for ch in str(s)).strip("_").lower()
_TEST = TEST.rename(index=lambda s: str(s).replace(" ← CHAMPION", ""))      # the notebook marks the champion row in TEST
_sha = hashlib.sha256(Path("diabetes_model/model_bundle.joblib").read_bytes()).hexdigest()

with mlflow.start_run(run_name=f"notebook-{CHAMPION}-seed{int(best_seed)}") as run:
    mlflow.set_tags({"stage": "training", "source": "notebook", "champion": CHAMPION, "dataset": "WHO STEPS Kenya 2015", "unit_of_split": "PSU (cluster)",
                     "python": platform.python_version(), "bundle_sha256": _sha[:16]})
    mlflow.log_params({"split_seed": int(best_seed), "random_state": SEED, "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte), "n_pos_train": int(ytr.sum()),
                       "n_pos_val": int(yva.sum()), "n_pos_test": int(yte.sum()), "n_features": len(FEATURES), "cost_ratio_FN_to_FP": f"{C_FN}:{C_FP}",
                       "referral_threshold": round(float(THRESHOLD), 6), "cv_scoring": "average_precision (PR-AUC)", "calibration": "Platt on out-of-fold predictions",
                       "sklearn": sklearn.__version__, "bundle_sha256": _sha})

    # 1) every hyper-parameter candidate the searches tried, grouped under one run per model family
    searches = {}
    if "lr_search" in globals(): searches["Elastic-net LR (grid)"] = pd.DataFrame(lr_search.cv_results_)
    if "rf_search" in globals(): searches["Random forest (random search)"] = pd.DataFrame(rf_search.cv_results_)
    if "study" in globals():
        t = study.trials_dataframe(); t = t[t.state == "COMPLETE"]
        searches["LightGBM (Optuna TPE)"] = pd.DataFrame({"params": [{k: v for k, v in tr.params.items()} for tr in study.trials if tr.state.name == "COMPLETE"],
                                                         "mean_test_score": [tr.value for tr in study.trials if tr.state.name == "COMPLETE"]})
    for name, df in searches.items():
        with mlflow.start_run(run_name=f"search: {name}", nested=True):
            mlflow.set_tag("kind", "hyperparameter_search")
            score = df["mean_test_score"]; mlflow.log_metrics({"cv_pr_auc_best": float(score.max()), "cv_pr_auc_median": float(score.median()), "n_candidates": float(len(df))})
            with tempfile.TemporaryDirectory() as d:
                df.to_csv(Path(d) / "cv_results.csv", index=False); mlflow.log_artifact(str(Path(d) / "cv_results.csv"), "search")
            for i, r in df.sort_values("mean_test_score", ascending=False).head(10).reset_index(drop=True).iterrows():           # top-10 candidates as their own runs
                with mlflow.start_run(run_name=f"candidate {i+1}", nested=True):
                    mlflow.log_params(_clean(r["params"]) if "params" in r else {}); mlflow.log_metric("cv_pr_auc_mean", float(r["mean_test_score"]))
                    if "std_test_score" in r: mlflow.log_metric("cv_pr_auc_std", float(r["std_test_score"]))

    # 2) final models: validation and test metrics side by side
    for name in VAL.index:
        with mlflow.start_run(run_name=str(name), nested=True):
            mlflow.set_tags({"model": str(name), "is_champion": str(name == CHAMPION), "kind": "final_model"})
            for m in ("ROC-AUC", "PR-AUC", "Brier"):
                mlflow.log_metric(f"val_{_slug(m)}", float(VAL.loc[name, m]))
                if name in _TEST.index: mlflow.log_metric(f"test_{_slug(m)}", float(_TEST.loc[name, m]))
    for m in ("ROC-AUC", "PR-AUC", "Brier"):
        mlflow.log_metric(f"val_{_slug(m)}", float(VAL.loc[CHAMPION, m])); mlflow.log_metric(f"test_{_slug(m)}", float(_TEST.loc[CHAMPION, m]))

    # the frozen referral rule evaluated on the held-out test set (variable `op` from the threshold cell)
    if "op" in globals():
        for k in ("referred_pct", "sensitivity", "specificity", "PPV", "NPV", "TP", "FN", "FP", "TN"): mlflow.log_metric(f"test_frozen_{k}", float(op[k]))

    # 3) artefacts: the exact files the API serves
    mlflow.log_artifact("diabetes_model/model_bundle.joblib", "model_files"); mlflow.log_artifact("diabetes_model/model_card.json", "model_files")
    mlflow.log_text("\n".join(FEATURES), "model_files/features.txt")
    print(f"MLflow run {run.info.run_id} logged to {TRACKING_URI}")
