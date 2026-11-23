
# AI_USE.md

## Tools

Claude (Anthropic) in an agentic coding/analysis workspace, used during columns mapping (233 were renamed), deciding the best method to handle missing data, and choosing the best models which will fit the data.

## What AI did

| Stage               | AI contribution                                                                                                                                                                                                                | Human oversight                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Dataset triage      | Compared candidate datasets, advised STEPS 2015 and against merging DHS                                                                                                                                                        | **I chose** the dataset and had it approved                                                  |
| Data mapping        | Matched 233 CSV columns to questionnaire sections (`Kenya2015.csv`); two columns mapped by value pattern (lipids `b8`/`b17`, device ID `m13`) and flagged as inferred                                                  | I supplied the questionnaire and checked the output file                                          |
| Modelling           | Decided on the best ways to fill the missing values, which to keep and which to drop.<br />Chose the models that would best fit the data.<br />Guided on calibration, thresholds and SHAP.<br />Target definition, cleaning,  | Wrote the notebook, splitting the data into 70/15/15 split and build the models + ensemble.        |
| Deployment          | Wrote the FastAPI service, test page, tests, Dockerfile, READMERa                                                                                                                                                              | Ran the containers and tested the FastAPI services.                                                |
| Experiment tracking | Wrote the MLflow logging scripts, the registered-model wrapper, the notebook logging cell, the aggregate-only serving monitor and the MLflow docker service                                                                    | Ran `docker compose --profile mlflow up --build -d` and check the runs match his notebook output |

## How AI output was checked

* Notebook executed end to end; leakage proven three ways (label-shuffle canary, ablation, structural audit).
* Serving code reproduces the notebook's probabilities to 1e-9 on 60 held-out people, and the saved model reproduces the reported validation/test metrics.
* Automated tests (`pytest`): input validation, leakage-item rejection, minimal/full payloads, monotonicity sanity check, aggregate-only metrics endpoint, and that the MLflow-registered model scores identically to the API.
* MLflow metrics were compared against `model_card.json` (e.g. test ROC-AUC 0.6903) and the registered model's SHA-256 against the bundle.

## Known limitations of AI assistance

* Judgement calls (the 20-80% missing band, the 10:1 cost ratio) were AI-proposed defaults and are documented as assumptions.
