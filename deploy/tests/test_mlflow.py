"""MLflow integration: the backfill script must register a model whose predictions equal the API's (same Scorer), with a champion alias."""
import subprocess, sys
from pathlib import Path
import pandas as pd, pytest
mlflow = pytest.importorskip("mlflow")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.schemas import Patient
from app.service import Scorer

PAYLOADS = [
    {"age": 54, "sex": "female", "residence": "urban", "height_cm": 158.0, "weight_kg": 82.0, "waist_cm": 102.0, "hip_cm": 111.0, "systolic_bp": 146.0, "diastolic_bp": 92.0, "heart_rate": 84.0, "told_raised_bp": True},
    {"age": 30, "sex": "male", "residence": "rural"},
    {"age": 66, "sex": "male", "residence": "urban", "waist_cm": 118.0, "systolic_bp": 171.0, "on_bp_medication": True, "education": "Primary school completed", "years_schooling": 8.0},
    {"age": 41, "sex": "female", "residence": "rural", "weight_kg": 61.0, "height_cm": 162.0, "current_smoker": False},
]

@pytest.fixture(scope="module")
def uri(tmp_path_factory):
    d = tmp_path_factory.mktemp("mlf"); u = f"sqlite:///{d / 'mlflow.db'}"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "mlflow_log_run.py"), "--tracking-uri", u], cwd=d, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return u

def test_registry_champion_matches_api(uri):
    mlflow.set_tracking_uri(uri)
    model = mlflow.pyfunc.load_model("models:/diabetes-screening-steps2015@champion")
    got = model.predict(pd.DataFrame(PAYLOADS))
    sc = Scorer(); want = [sc.predict(Patient(**p)) for p in PAYLOADS]
    assert [round(x, 6) for x in got.risk_probability] == [round(w["risk_probability"], 6) for w in want]
    assert list(got.decision) == [w["decision"] for w in want]

def test_runs_and_metrics_recorded(uri):
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["diabetes-screening-steps2015"])
    parent = runs[runs["tags.mlflow.parentRunId"].isna()].iloc[0]
    assert abs(parent["metrics.test_roc_auc"] - 0.6903) < 1e-9 and abs(parent["metrics.val_pr_auc"] - 0.1542) < 1e-9
    assert len(runs) == 7            # 1 parent + 6 models (2 baselines + 4 candidates)
    mv = mlflow.MlflowClient().get_model_version_by_alias("diabetes-screening-steps2015", "champion")
    assert mv.tags["test_roc_auc"] == "0.6903"
