# BRFSS Diabetes Risk Baseline


End-to-end baseline for a **binary diabetes risk** label using the CDC **Behavioral Risk Factor Surveillance System (BRFSS)** 2023 landline–cellphone combined dataset (`LLCP2023.XPT`). The pipeline lives in a single script: `diabetes_brfss_baseline.py`.

## What this project does

- Loads the SAS XPORT file with `pandas.read_sas(..., format="xport")`.
- Builds a **binary cohort**: positive if `DIABETE4 == 1`, negative if `DIABETE4 == 3`; other codes and missing values are dropped.
- Cleans BRFSS sentinel missing codes for the selected feature set.
- **EDA**: tables and figures under `outputs_brfss_diabetes/eda/` and `outputs_brfss_diabetes/figures/`.
- **Models**:
  - **Logistic regression** with a small hyperparameter sweep over `C`.
  - **XGBoost** (optional): same idea over `max_depth` and `learning_rate` if `xgboost` is installed.
- **Model selection**: best LR / XGB candidate by **validation AUPRC** (then full test evaluation with saved ROC/PR/confusion plots).
- **Fairness-style reporting**: input label-prevalence checks, subgroup metrics, readable subgroup labels, metric-gap summaries, and bias figures under `outputs_brfss_diabetes/bias/`.
- **Explainability** (optional): permutation importance, one-hot **SHAP** summary plots, grouped-by-original-variable **SHAP** summary plots/importance bar plots/tables, and **LIME** local explanation HTML/PNG/CSV outputs when those libraries are available.
- **Model artifacts**: final selected sklearn pipelines under `outputs_brfss_diabetes/models/`.

All file paths and column lists are configured at the top of `diabetes_brfss_baseline.py` (e.g. `DATA_PATH`, `OUTPUT_DIR`, `FEATURE_COLS`).

## Data

1. Obtain **`LLCP2023.XPT`** from the CDC BRFSS site (2023 combined landline and cellphone data).
2. Place it in the project root (default `DATA_PATH = "LLCP2023.XPT"`) or update `DATA_PATH` in the script.

Large XPT files and the generated `outputs_brfss_diabetes/` directory are listed in `.gitignore` so they are not committed by mistake.

Optional: the same file may be available from the Hugging Face dataset [`YilingMa/DiabScreen`](https://huggingface.co/datasets/YilingMa/DiabScreen) (check the dataset card for terms and versioning). You are responsible for complying with **CDC redistribution and citation** requirements when sharing or publishing results.

## Requirements

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

| Package        | Role |
|----------------|------|
| `numpy`, `pandas`, `matplotlib`, `scikit-learn` | Required for the main pipeline |
| `xgboost`      | Optional second model family |
| `mlflow`       | Optional experiment tracking and artifacts |
| `shap`, `lime` | Optional explainability outputs |

If an optional dependency is missing, the script prints a short notice and continues with reduced functionality.

## Run

From the repository root:

```bash
python diabetes_brfss_baseline.py
```

Train / validation / test split is **70% / 15% / 15%**, stratified on the binary label (`RANDOM_STATE = 42` in code).

## Outputs

Everything is written under **`outputs_brfss_diabetes/`** (created automatically):

| Path | Content |
|------|---------|
| `eda/` | EDA tables and summaries, including simple and appendix-ready Table 1 outputs |
| `figures/` | ROC, PR, calibration, confusion matrices, etc. |
| `bias/` | Subgroup performance tables |
| `explainability/` | Coefficients, permutation importance, SHAP plots, LIME HTML |
| `models/` | Pickled final sklearn pipelines for selected models |
| `model_comparison_metrics.csv` | Final test metrics per model |
| `split_comparison_metrics.csv` | Train / validation / test metrics for selected final models |
| `hyperparameter_sweep_results.csv` | One row per sweep candidate |
| `*_test_predictions.csv` | Held-out test predictions for downstream analysis |
| `run_config.json` | Serialized run configuration (features, splits, flags) |
| `mlruns/` | Local MLflow tracking store (only populated when MLflow is used) |

## MLflow

Logging is **opt-in by installation**: if `import mlflow` succeeds, the script configures a **local file store** and logs one parent run plus nested child runs.

### Tracking URI and layout

- **Tracking URI**: `file:<absolute-path-to>/outputs_brfss_diabetes/mlruns`
- **Experiment name**: `brfss_diabetes_risk`
- **Parent run name**: `hyperparameter_sweep`

Each hyperparameter candidate is logged under a **nested** child run (`nested=True`) named after the model label (e.g. logistic regression with a given `C`, or XGBoost with depth / learning rate).

### What gets logged

- **Parent run**
  - **Parameters**: data path, label column, feature metadata, split sizes, prevalence, random seed, sweep config lists, flags for optional libs (`has_xgboost`, `has_shap`, `has_lime`), etc.
  - **Metrics**: flattened test metrics prefixed by model name (same convention as `log_metrics_to_mlflow` in code).
  - **Artifacts**: `model_comparison_metrics.csv`, `split_comparison_metrics.csv`, `hyperparameter_sweep_results.csv`, `run_config.json`, and the directories `eda/`, `figures/`, `bias/`, `explainability/`, and `models/` (uploaded as artifact folders).
- **Child runs**
  - **Parameters**: `model_family` plus the sweep parameters for that candidate.
  - **Metrics**: AUROC, AUPRC, F1, precision, recall, specificity, Brier score, threshold for that candidate.

If MLflow is **not** installed, the script prints that logging is skipped and still writes all CSV/plot outputs to disk.

### View runs in the MLflow UI

Install MLflow (already in `requirements.txt` as optional), then from the project root:

```bash
mlflow ui --backend-store-uri "$(pwd)/outputs_brfss_diabetes/mlruns"
```

Open the URL shown in the terminal (by default `http://127.0.0.1:5000`), select experiment **`brfss_diabetes_risk`**, and inspect the parent run and nested children.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── diabetes_brfss_baseline.py   # full pipeline
└── outputs_brfss_diabetes/        # generated (gitignored if configured)
```

## Disclaimer

This repository is for **research and education**. It is not medical advice. Use CDC BRFSS documentation for correct variable definitions, survey weights (this baseline does **not** apply survey weights), and any publication requirements.

## Citation

If you use BRFSS data, cite the CDC BRFSS and the survey year according to [CDC guidance](https://www.cdc.gov/brfss/). Cite this code repository separately if you reuse these scripts.
