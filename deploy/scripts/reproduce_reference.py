"""Re-run the notebook's data-preparation cells (sections 1-10) in the pinned environment and verify that the saved
model bundle reproduces the metrics in model_card.json. Dumps metadata needed by the API."""
import sys, json, re, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import nbformat as nbf, joblib, numpy as np, pandas as pd
NB, CSV, BUNDLE, CARD = sys.argv[1:5]
nb = nbf.read(NB, 4)
ns = {"display": lambda *a, **k: None}
for c in nb.cells:
    if c.cell_type != "code": continue
    if "GridSearchCV(lr_pipe" in c.source: break
    src = c.source.replace("%matplotlib inline\n", "")
    src = re.sub(r'DATA_PATH = ".*?"', f'DATA_PATH = r"{CSV}"', src, count=1)
    src = src.replace("plt.show()", "plt.close('all')")
    exec(compile(src, "<nb>", "exec"), ns)
b = joblib.load(BUNDLE); card = json.load(open(CARD))
assert list(ns["FEATURES"]) == list(b["features"]), "feature list differs from bundle"
assert ns["STRUCT_RULES"].keys() == b["struct_rules"].keys() and sorted(ns["DROP80"]) == sorted(b["drop80"]) and ns["UNIVERSE"] == b["universe"]
print("feature list / struct rules / drop list / universe: IDENTICAL to bundle;  split seed", ns["best_seed"], "bundle", b["split_seed"])
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
m = b["calibrated_models"]["Elastic-net LR"]
for nm, F, y, key in (("validation", ns["Fva"], ns["yva"], "validation"), ("test", ns["Fte"], ns["yte"], "test")):
    p = m.predict_proba(F)[:, 1]
    got = (round(roc_auc_score(y, p), 4), round(average_precision_score(y, p), 4), round(brier_score_loss(y, p), 4))
    ck = "Elastic-net LR" if key == "validation" else "Elastic-net LR ← CHAMPION"
    want = tuple(card[key][ck][k] for k in ("ROC-AUC", "PR-AUC", "Brier"))
    print(nm, "reproduced", got, "card", want, "MATCH" if np.allclose(got, want, atol=2e-4) else "**MISMATCH**")
# metadata for the service
meta = {"LEVEL_COLS": ns["LEVEL_COLS"], "OBJ_COLS": ns["OBJ_COLS"], "CAT_NUM": ns["CAT_NUM"], "BIN_COLS": ns["BIN_COLS"], "CONSUMED": ns["CONSUMED"],
        "ORIG_COLS": ns["ORIG_COLS"], "ID_COLS": ns["ID_COLS"], "RANGES": {k: list(v) for k, v in ns["RANGES"].items()},
        "string_levels": {c: sorted(ns["Xall"][c].dropna().unique().tolist()) for c in ["C5_Highest_Education", "C6_Ethnic_Background", "C7_Marital_Status", "I6_Interview_Language"]},
        "train_ranges": {c: [float(ns["Ftr"][c].quantile(.01)), float(ns["Ftr"][c].quantile(.99))] for c in ["age", "bmi", "waist", "whr", "sbp", "dbp", "hr"]},
        "base_rate_train": float(ns["ytr"].mean()), "n_train": int(len(ns["ytr"]))}
json.dump(meta, open("meta.json", "w"), indent=1); print("meta.json written;", {k: (len(v) if hasattr(v, '__len__') else v) for k, v in meta.items()})
# reference rows (clean-domain inputs + features + expected probabilities) for the API parity test
ref = ns["Xte_raw"].iloc[:60].copy(); ref["_expected_p"] = m.predict_proba(ns["Fte"].iloc[:60])[:, 1]; ref["_y"] = ns["yte"].iloc[:60].values
ref.to_pickle("reference_rows.pkl")
ns["Xte_raw"].to_pickle("test_clean_rows.pkl"); ns["yte"].to_pickle("test_y.pkl"); ns["Xva_raw"].to_pickle("val_clean_rows.pkl"); ns["yva"].to_pickle("val_y.pkl")
