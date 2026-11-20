# AI_USE.md: DSA 8401 Capstone (draft prepared with Claude; edit so it states what *you* actually did)

## Tools
Claude (Anthropic) in an agentic coding/analysis workspace, used from the proposal stage to deployment.

## What AI did
| Stage | AI contribution | Human oversight |
|---|---|---|
| Dataset triage | Compared candidate datasets, advised STEPS 2015 and against merging DHS | **KK chose** the dataset and had it approved |
| Data mapping | Matched 233 CSV columns to questionnaire sections (`Kenya2015.csv`); two columns mapped by value pattern (lipids `b8`/`b17`, device ID `m13`) and flagged as inferred | KK supplied the questionnaire and checked the file |
| Modelling | Wrote the notebook: target definition, cleaning, PSU-grouped 70/15/15 split, missing-data audit, leakage controls, three models + ensemble, calibration, threshold, SHAP | **KK reviewed and re-ran** the notebook on his machine; his run produced the model that is deployed |
| Deployment | Wrote the FastAPI service, test page, tests, Dockerfile, README | KK to run the container and confirm behaviour |
| Experiment tracking | Wrote the MLflow logging scripts, the registered-model wrapper, the notebook logging cell, the aggregate-only serving monitor and the MLflow docker service | KK to run `docker compose --profile mlflow up --build -d` and check the runs match his notebook output |

## How AI output was checked
* Notebook executed end to end; leakage proven three ways (label-shuffle canary, ablation, structural audit).
* Serving code reproduces the notebook's probabilities to 1e-9 on 60 held-out people, and the saved model reproduces the reported validation/test metrics.
* Automated tests (`pytest`): input validation, leakage-item rejection, minimal/full payloads, monotonicity sanity check, aggregate-only metrics endpoint, and that the MLflow-registered model scores identically to the API.
* MLflow metrics were compared against `model_card.json` (e.g. test ROC-AUC 0.6903) and the registered model's SHA-256 against the bundle.

## Known limitations of AI assistance
* Judgement calls (the 20-80% missing band, the 10:1 cost ratio) were AI-proposed defaults and are documented as assumptions.
* Neither Docker image (API, MLflow) was built by the AI (no daemon available there); the MLflow server and scripts were tested over HTTP on localhost instead.

## Declaration (KK to complete)
Parts I wrote or changed myself: ______________________ ; decisions I made: ______________________ ; I understand and can explain every part of the submitted code: yes / no.
