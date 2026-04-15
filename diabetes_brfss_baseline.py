# diabetes_brfss_baseline.py

from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

warnings.filterwarnings("ignore")


# ============================================================
# Config
# ============================================================

RANDOM_STATE = 42
DATA_PATH = "LLCP2023.XPT"   
OUTPUT_DIR = "outputs_brfss_diabetes"
os.makedirs(OUTPUT_DIR, exist_ok=True)
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
BIAS_DIR = os.path.join(OUTPUT_DIR, "bias")
EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explainability")
MLFLOW_DIR = os.path.join(OUTPUT_DIR, "mlruns")
for output_path in [EDA_DIR, FIGURE_DIR, BIAS_DIR, EXPLAIN_DIR]:
    os.makedirs(output_path, exist_ok=True)
LR_SWEEP_CONFIGS = [
    {"C": 0.1},
    {"C": 1.0},
    {"C": 5.0},
]
XGB_SWEEP_CONFIGS = [
    {"max_depth": 3, "learning_rate": 0.05},
    {"max_depth": 5, "learning_rate": 0.05},
    {"max_depth": 5, "learning_rate": 0.10},
]


# ============================================================
# Column definitions
# ============================================================

LABEL_COL = "DIABETE4"

# Keep this first version compact and interpretable
FEATURE_COLS = [
    "SEXVAR", "_AGE80", "_RACEGR3", "_EDUCAG", "INCOME3",
    "GENHLTH", "PHYSHLTH", "MENTHLTH",
    "PRIMINS1", "PERSDOC3", "MEDCOST1", "CHECKUP1",
    "EXERANY2", "_SMOKER3", "ACEDRINK",
    "BPHIGH6", "BPMEDS1", "CHOLCHK3", "_BMI5"
]

# Diabetes-related leakage columns that should never be used as predictors
LEAKAGE_COLS = [
    "DIABETE4",
    "DIABAGE4",
    "PDIABTS1",
    "PREDIAB2",
    "DIABTYPE",
    "DIABEYE1",
    "DIABEDU1",
]

# Numeric vs categorical split for preprocessing
NUMERIC_COLS = ["_AGE80", "PHYSHLTH", "MENTHLTH", "ACEDRINK", "_BMI5"]
CATEGORICAL_COLS = [c for c in FEATURE_COLS if c not in NUMERIC_COLS]
SUBGROUP_COLS = ["SEXVAR", "_RACEGR3", "_EDUCAG", "INCOME3", "age_group"]


@dataclass
class MlflowRunContext:
    run_id: str | None = None
    enabled: bool = False


# ============================================================
# Utilities
# ============================================================

def read_brfss_xpt(path: str) -> pd.DataFrame:
    """Read BRFSS XPT file."""
    requested_path = Path(path).expanduser()
    resolved_path = requested_path

    if not resolved_path.exists():
        parent = resolved_path.parent if str(resolved_path.parent) != "" else Path(".")
        target_name = resolved_path.name.strip()

        for candidate in parent.iterdir():
            if candidate.is_file() and candidate.name.strip() == target_name:
                resolved_path = candidate
                break
        else:
            available = sorted(
                candidate.name for candidate in parent.iterdir() if candidate.is_file()
            )
            raise FileNotFoundError(
                f"Could not find BRFSS XPT file '{path}'. "
                f"Files in '{parent.resolve()}': {available}"
            )

    print(f"Reading data from: {resolved_path}")
    df = pd.read_sas(resolved_path, format="xport")
    print(f"Loaded shape: {df.shape}")
    return df


def decode_bytes_in_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Decode byte strings in object columns if present."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else x
            )
    return df


def replace_brfss_missing(series: pd.Series) -> pd.Series:
    """
    Replace common BRFSS special missing/refused codes with NaN.
    This is a generic helper; some variables may have additional special codes.
    """
    missing_codes = {
        7, 9,
        77, 99,
        777, 999,
        7777, 9999
    }
    return series.apply(lambda x: np.nan if x in missing_codes else x)


def clean_feature_values(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Apply BRFSS missing-code cleanup to selected feature columns."""
    df = df.copy()
    for col in feature_cols:
        df[col] = replace_brfss_missing(df[col])

    # Convert BMI from integer-coded BMI*100 to actual BMI
    if "_BMI5" in df.columns:
        df["_BMI5"] = df["_BMI5"] / 100.0

    return df


def build_binary_diabetes_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build binary cohort:
      positive = DIABETE4 == 1
      negative = DIABETE4 == 3
      exclude = 2, 4, 7, 9, NaN
    """
    keep_cols = [LABEL_COL] + FEATURE_COLS
    data = df[keep_cols].copy()

    data = data[data[LABEL_COL].isin([1.0, 3.0])].copy()
    data["y"] = (data[LABEL_COL] == 1.0).astype(int)
    data.drop(columns=[LABEL_COL], inplace=True)

    return data


def add_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add report-friendly derived variables used in EDA and subgroup analysis."""
    data = data.copy()
    age_bins = [18, 35, 50, 65, 81]
    age_labels = ["18-34", "35-49", "50-64", "65-80"]
    data["age_group"] = pd.cut(
        data["_AGE80"],
        bins=age_bins,
        labels=age_labels,
        right=False,
        include_lowest=True,
    )
    return data


def setup_mlflow() -> MlflowRunContext:
    """Configure a local MLflow tracking directory."""
    if not HAS_MLFLOW:
        print("\nMLflow not installed; skipping MLOps logging.")
        return MlflowRunContext()

    tracking_uri = f"file:{Path(MLFLOW_DIR).resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("brfss_diabetes_risk")
    run = mlflow.start_run(run_name="hyperparameter_sweep")
    return MlflowRunContext(run_id=run.info.run_id, enabled=True)


def log_params_to_mlflow(params: Dict[str, object]) -> None:
    if not HAS_MLFLOW:
        return
    safe_params = {}
    for key, value in params.items():
        if isinstance(value, (list, dict, tuple)):
            safe_params[key] = json.dumps(value)
        else:
            safe_params[key] = value
    mlflow.log_params(safe_params)


def log_metrics_to_mlflow(metrics_df: pd.DataFrame) -> None:
    if not HAS_MLFLOW:
        return
    for _, row in metrics_df.iterrows():
        model_name = row["model"]
        for col in metrics_df.columns:
            if col == "model":
                continue
            if pd.api.types.is_number(row[col]):
                mlflow.log_metric(f"{model_name}_{col}", float(row[col]))


def log_artifact_to_mlflow(path: str) -> None:
    if HAS_MLFLOW and os.path.exists(path):
        mlflow.log_artifact(path)


def log_directory_to_mlflow(path: str, artifact_path: str) -> None:
    if HAS_MLFLOW and os.path.exists(path):
        mlflow.log_artifacts(path, artifact_path=artifact_path)


def summarize_cohort(data: pd.DataFrame) -> None:
    """Print cohort summary."""
    print("\n=== Cohort Summary ===")
    print(f"Shape: {data.shape}")
    print("Outcome distribution:")
    print(data["y"].value_counts(dropna=False))
    print(f"Positive prevalence: {data['y'].mean():.4f}")


def missingness_report(data: pd.DataFrame) -> pd.DataFrame:
    """Return missingness report."""
    report = (
        data.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    return report


def create_table_one(data: pd.DataFrame) -> pd.DataFrame:
    """Create a simple Table 1 contrasting diabetes vs no diabetes."""
    rows = []
    grouped = data.groupby("y")

    for col in NUMERIC_COLS:
        rows.append({
            "variable": col,
            "type": "numeric",
            "overall": f"{data[col].mean():.2f} ({data[col].std():.2f})",
            "no_diabetes_y0": f"{grouped[col].mean().get(0, np.nan):.2f} ({grouped[col].std().get(0, np.nan):.2f})",
            "diabetes_y1": f"{grouped[col].mean().get(1, np.nan):.2f} ({grouped[col].std().get(1, np.nan):.2f})",
            "missing_rate": data[col].isna().mean(),
        })

    for col in CATEGORICAL_COLS + ["age_group"]:
        top_level = data[col].mode(dropna=True)
        top_value = top_level.iloc[0] if not top_level.empty else np.nan
        overall_pct = (data[col] == top_value).mean()
        y0_pct = (data.loc[data["y"] == 0, col] == top_value).mean()
        y1_pct = (data.loc[data["y"] == 1, col] == top_value).mean()
        rows.append({
            "variable": col,
            "type": f"categorical_top={top_value}",
            "overall": f"{overall_pct:.2%}",
            "no_diabetes_y0": f"{y0_pct:.2%}",
            "diabetes_y1": f"{y1_pct:.2%}",
            "missing_rate": data[col].isna().mean(),
        })

    return pd.DataFrame(rows)


def plot_missingness(miss: pd.DataFrame, save_path: str, top_n: int = 15) -> None:
    plot_df = miss.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["variable"], plot_df["missing_rate"])
    plt.xlabel("Missing rate")
    plt.title(f"Top {top_n} Missingness Rates")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_numeric_by_outcome(data: pd.DataFrame, feature: str, save_path: str) -> None:
    """Overlay histogram by outcome without extra plotting dependencies."""
    plt.figure(figsize=(7, 5))
    for outcome, label in [(0, "No diabetes"), (1, "Diabetes")]:
        values = data.loc[data["y"] == outcome, feature].dropna()
        plt.hist(values, bins=30, alpha=0.5, density=True, label=label)
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title(f"{feature} Distribution by Outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prevalence_by_group(data: pd.DataFrame, feature: str, save_path: str) -> None:
    plot_df = (
        data.groupby(feature, dropna=False)["y"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "prevalence", "count": "n"})
    )
    plot_df[feature] = plot_df[feature].astype(str)
    plt.figure(figsize=(8, 5))
    plt.bar(plot_df[feature], plot_df["prevalence"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Diabetes prevalence")
    plt.title(f"Diabetes Prevalence by {feature}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_eda_outputs(data: pd.DataFrame) -> None:
    """Save EDA artifacts used in the report."""
    table_one = create_table_one(data)
    table_one.to_csv(os.path.join(EDA_DIR, "table1_cohort_summary.csv"), index=False)

    miss = missingness_report(data)
    miss.to_csv(os.path.join(EDA_DIR, "missingness_report.csv"), index=False)

    plot_missingness(miss, os.path.join(EDA_DIR, "missingness_top15.png"))
    plot_numeric_by_outcome(data, "_AGE80", os.path.join(EDA_DIR, "age_distribution_by_outcome.png"))
    plot_numeric_by_outcome(data, "_BMI5", os.path.join(EDA_DIR, "bmi_distribution_by_outcome.png"))
    plot_prevalence_by_group(data, "age_group", os.path.join(EDA_DIR, "prevalence_by_age_group.png"))
    plot_prevalence_by_group(data, "_RACEGR3", os.path.join(EDA_DIR, "prevalence_by_race.png"))


def make_preprocessor() -> ColumnTransformer:
    """Preprocessing for numeric and categorical columns."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )
    return preprocessor


def specificity_from_confusion_matrix(cm: np.ndarray) -> float:
    """Compute specificity from a 2x2 confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    return tn / denom if denom > 0 else np.nan


def compute_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision_ppv": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_from_confusion_matrix(cm),
        "brier_score": brier_score_loss(y_true, y_prob),
        "threshold": threshold,
    }
    return metrics


def choose_threshold_by_f1(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Choose threshold on validation set by maximizing F1."""
    thresholds = np.linspace(0.05, 0.95, 19)
    scores = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        scores.append(f1_score(y_true, y_pred, zero_division=0))
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx])


def plot_roc(y_true: pd.Series, y_prob: np.ndarray, title: str, save_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr(y_true: pd.Series, y_prob: np.ndarray, title: str, save_path: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUPRC = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration(y_true: pd.Series, y_prob: np.ndarray, title: str, save_path: str) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed probability")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion(y_true: pd.Series, y_prob: np.ndarray, threshold: float, title: str, save_path: str) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def subgroup_metrics(
    df_eval: pd.DataFrame,
    subgroup_col: str,
    prob_col: str = "y_prob",
    label_col: str = "y",
    threshold: float = 0.5
) -> pd.DataFrame:
    """Compute subgroup metrics for bias assessment."""
    rows = []
    for group_value, group_df in df_eval.groupby(subgroup_col, dropna=False):
        if len(group_df) < 50:
            continue
        try:
            m = compute_metrics(group_df[label_col], group_df[prob_col], threshold=threshold)
            m["group"] = group_value
            m["n"] = len(group_df)
            rows.append(m)
        except Exception:
            continue
    return pd.DataFrame(rows)


def save_subgroup_outputs(
    eval_df: pd.DataFrame,
    model_name: str,
    threshold: float,
) -> None:
    for subgroup_col in SUBGROUP_COLS:
        subgroup_df = subgroup_metrics(
            eval_df,
            subgroup_col=subgroup_col,
            prob_col="y_prob",
            label_col="y",
            threshold=threshold,
        )
        subgroup_df.to_csv(
            os.path.join(BIAS_DIR, f"{model_name}_subgroup_{subgroup_col}.csv"),
            index=False,
        )


def save_logistic_coefficients(model: Pipeline, save_path: str) -> None:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(save_path, index=False)


def save_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> None:
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(
        os.path.join(EXPLAIN_DIR, f"{model_name}_permutation_importance.csv"),
        index=False,
    )


def save_shap_outputs(
    model: Pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    model_name: str,
) -> None:
    if not HAS_SHAP:
        return
    try:
        transformed_background = model.named_steps["preprocessor"].transform(X_background)
        transformed_explain = model.named_steps["preprocessor"].transform(X_explain)
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        predictor = model.named_steps["model"]

        if hasattr(transformed_background, "toarray"):
            transformed_background = transformed_background.toarray()
        if hasattr(transformed_explain, "toarray"):
            transformed_explain = transformed_explain.toarray()

        background_sample = transformed_background[: min(200, len(X_background))]
        explain_sample = transformed_explain[: min(500, len(X_explain))]

        if isinstance(predictor, LogisticRegression):
            explainer = shap.LinearExplainer(predictor, background_sample, feature_names=feature_names)
            shap_values = explainer(explain_sample)
        elif HAS_XGBOOST and isinstance(predictor, XGBClassifier):
            explainer = shap.TreeExplainer(predictor.get_booster())
            shap_values = explainer.shap_values(explain_sample)
        else:
            explainer = shap.Explainer(predictor.predict_proba, background_sample, feature_names=feature_names)
            shap_values = explainer(explain_sample)

        plt.figure()
        shap.summary_plot(shap_values, explain_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(EXPLAIN_DIR, f"{model_name}_shap_summary.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"Skipping SHAP for {model_name}: {exc}")


def save_lime_example(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
) -> None:
    if not HAS_LIME:
        return
    try:
        preprocessor = model.named_steps["preprocessor"]
        predictor = model.named_steps["model"]

        transformed_train = preprocessor.transform(X_train)
        transformed_test = preprocessor.transform(X_test.iloc[:1])

        if hasattr(transformed_train, "toarray"):
            transformed_train = transformed_train.toarray()
        if hasattr(transformed_test, "toarray"):
            transformed_test = transformed_test.toarray()

        feature_names = list(preprocessor.get_feature_names_out())
        explainer = LimeTabularExplainer(
            training_data=transformed_train,
            feature_names=feature_names,
            class_names=["no_diabetes", "diabetes"],
            mode="classification",
            random_state=RANDOM_STATE,
        )
        explanation = explainer.explain_instance(
            transformed_test[0],
            predictor.predict_proba,
            num_features=10,
        )
        explanation.save_to_file(os.path.join(EXPLAIN_DIR, f"{model_name}_lime_example.html"))
    except Exception as exc:
        print(f"Skipping LIME for {model_name}: {exc}")


# ============================================================
# Training
# ============================================================

def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, c_value: float = 1.0
) -> Pipeline:
    preprocessor = make_preprocessor()

    model = LogisticRegression(
        C=c_value,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 5,
    learning_rate: float = 0.05,
) -> Pipeline:
    if not HAS_XGBOOST:
        raise ImportError(
            "xgboost is not installed. Install with: pip install xgboost"
        )

    preprocessor = make_preprocessor()

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist"
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(
    model: Pipeline,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    save_artifacts: bool = True,
) -> Tuple[pd.DataFrame, float, np.ndarray]:
    """Evaluate on validation first to choose threshold, then final test."""
    valid_prob = model.predict_proba(X_valid)[:, 1]
    best_threshold = choose_threshold_by_f1(y_valid, valid_prob)

    test_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_prob, threshold=best_threshold)

    metrics_df = pd.DataFrame([{
        "model": model_name,
        **metrics
    }])

    if save_artifacts:
        plot_roc(y_test, test_prob, f"{model_name} ROC", os.path.join(FIGURE_DIR, f"{model_name}_roc.png"))
        plot_pr(y_test, test_prob, f"{model_name} PR", os.path.join(FIGURE_DIR, f"{model_name}_pr.png"))
        plot_calibration(y_test, test_prob, f"{model_name} Calibration", os.path.join(FIGURE_DIR, f"{model_name}_calibration.png"))
        plot_confusion(
            y_test,
            test_prob,
            best_threshold,
            f"{model_name} Confusion Matrix",
            os.path.join(FIGURE_DIR, f"{model_name}_confusion_matrix.png"),
        )

    return metrics_df, best_threshold, test_prob


def run_sweep_candidate(
    model_family: str,
    model_label: str,
    params: Dict[str, float],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Pipeline, pd.DataFrame, float, np.ndarray]:
    """Train one candidate configuration and optionally log a child MLflow run."""
    if model_family == "logistic_regression":
        model = train_logistic_regression(X_train, y_train, c_value=float(params["C"]))
    elif model_family == "xgboost":
        model = train_xgboost(
            X_train,
            y_train,
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
        )
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    metrics_df, threshold, test_prob = evaluate_model(
        model,
        X_valid,
        y_valid,
        X_test,
        y_test,
        model_label,
        save_artifacts=False,
    )
    metrics_df["model_family"] = model_family
    metrics_df["config"] = json.dumps(params, sort_keys=True)

    if HAS_MLFLOW:
        with mlflow.start_run(run_name=model_label, nested=True):
            log_params_to_mlflow({"model_family": model_family, **params})
            log_metrics_to_mlflow(metrics_df[["model", "auroc", "auprc", "f1", "precision_ppv", "recall_sensitivity", "specificity", "brier_score", "threshold"]])

    return model, metrics_df, threshold, test_prob


# ============================================================
# Main
# ============================================================

def main() -> None:
    mlflow_ctx = setup_mlflow()

    # 1) Read and decode
    df = read_brfss_xpt(DATA_PATH)
    df = decode_bytes_in_object_cols(df)

    # 2) Sanity checks
    print("\n=== Diabetes label distribution ===")
    print(df[LABEL_COL].value_counts(dropna=False).sort_index())

    # 3) Build binary cohort
    data = build_binary_diabetes_cohort(df)

    # 4) Clean BRFSS missing codes
    data = clean_feature_values(data, FEATURE_COLS)
    data = add_derived_features(data)

    # 5) Summarize
    summarize_cohort(data)

    # 6) EDA outputs
    generate_eda_outputs(data)

    miss = missingness_report(data)
    print("\nTop missingness:")
    print(miss.head(15))

    # 7) Split
    X = data[FEATURE_COLS].copy()
    y = data["y"].copy()

    # train / valid / test = 70 / 15 / 15
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full,
        test_size=0.17647,  # 0.17647 * 0.85 ≈ 0.15 total
        stratify=y_train_full,
        random_state=RANDOM_STATE
    )

    print("\n=== Split sizes ===")
    print(f"Train: {X_train.shape}, prevalence={y_train.mean():.4f}")
    print(f"Valid: {X_valid.shape}, prevalence={y_valid.mean():.4f}")
    print(f"Test : {X_test.shape}, prevalence={y_test.mean():.4f}")

    # 8) Small hyperparameter sweep
    sweep_rows = []
    eval_datasets = {}
    thresholds = {}

    print("\nRunning Logistic Regression sweep...")
    lr_candidates = []
    for params in LR_SWEEP_CONFIGS:
        model_label = f"logistic_regression_C{params['C']}"
        lr_model_candidate, lr_metrics_candidate, lr_thr_candidate, lr_prob_candidate = run_sweep_candidate(
            "logistic_regression",
            model_label,
            params,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
        )
        lr_candidates.append((lr_model_candidate, lr_metrics_candidate, lr_thr_candidate, lr_prob_candidate, params))
        sweep_rows.append(lr_metrics_candidate)

    best_lr_model, best_lr_metrics, lr_thr, lr_test_prob, best_lr_params = max(
        lr_candidates,
        key=lambda item: float(item[1]["auprc"].iloc[0]),
    )
    lr_model_name = "logistic_regression"
    lr_metrics, lr_thr, lr_test_prob = evaluate_model(
        best_lr_model,
        X_valid,
        y_valid,
        X_test,
        y_test,
        lr_model_name,
        save_artifacts=True,
    )
    lr_metrics["selected_params"] = json.dumps(best_lr_params, sort_keys=True)
    thresholds[lr_model_name] = lr_thr

    eval_lr = X_test.copy()
    eval_lr["age_group"] = data.loc[X_test.index, "age_group"].astype(str).values
    eval_lr["y"] = y_test.values
    eval_lr["y_prob"] = lr_test_prob
    eval_lr.to_csv(os.path.join(OUTPUT_DIR, "logistic_regression_test_predictions.csv"), index=False)
    eval_datasets[lr_model_name] = eval_lr

    all_metrics = [lr_metrics]

    if HAS_XGBOOST:
        print("\nRunning XGBoost sweep...")
        xgb_candidates = []
        for params in XGB_SWEEP_CONFIGS:
            model_label = f"xgboost_depth{params['max_depth']}_lr{params['learning_rate']}"
            xgb_model_candidate, xgb_metrics_candidate, xgb_thr_candidate, xgb_prob_candidate = run_sweep_candidate(
                "xgboost",
                model_label,
                params,
                X_train,
                y_train,
                X_valid,
                y_valid,
                X_test,
                y_test,
            )
            xgb_candidates.append((xgb_model_candidate, xgb_metrics_candidate, xgb_thr_candidate, xgb_prob_candidate, params))
            sweep_rows.append(xgb_metrics_candidate)

        xgb_model, _, _, _, best_xgb_params = max(
            xgb_candidates,
            key=lambda item: float(item[1]["auprc"].iloc[0]),
        )
        xgb_metrics, xgb_thr, xgb_test_prob = evaluate_model(
            xgb_model,
            X_valid,
            y_valid,
            X_test,
            y_test,
            "xgboost",
            save_artifacts=True,
        )
        xgb_metrics["selected_params"] = json.dumps(best_xgb_params, sort_keys=True)
        all_metrics.append(xgb_metrics)
        thresholds["xgboost"] = xgb_thr

        eval_xgb = X_test.copy()
        eval_xgb["age_group"] = data.loc[X_test.index, "age_group"].astype(str).values
        eval_xgb["y"] = y_test.values
        eval_xgb["y_prob"] = xgb_test_prob
        eval_xgb.to_csv(os.path.join(OUTPUT_DIR, "xgboost_test_predictions.csv"), index=False)
        eval_datasets["xgboost"] = eval_xgb
    else:
        print("\nXGBoost not installed; skipping XGBoost baseline.")

    sweep_results = pd.concat(sweep_rows, ignore_index=True)
    sweep_results.to_csv(os.path.join(OUTPUT_DIR, "hyperparameter_sweep_results.csv"), index=False)

    # 10) Save comparison table
    results = pd.concat(all_metrics, ignore_index=True)
    results.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_metrics.csv"), index=False)
    print("\n=== Final Test Metrics ===")
    print(results)

    # 11) Bias and explainability outputs
    save_subgroup_outputs(eval_lr, "logistic_regression", lr_thr)
    save_logistic_coefficients(
        best_lr_model,
        os.path.join(EXPLAIN_DIR, "logistic_regression_coefficients.csv"),
    )
    save_permutation_importance(best_lr_model, X_test, y_test, "logistic_regression")
    save_shap_outputs(best_lr_model, X_train, X_test, "logistic_regression")
    save_lime_example(best_lr_model, X_train, X_test, "logistic_regression")

    if HAS_XGBOOST:
        save_subgroup_outputs(eval_datasets["xgboost"], "xgboost", thresholds["xgboost"])
        save_permutation_importance(xgb_model, X_test, y_test, "xgboost")
        save_shap_outputs(xgb_model, X_train, X_test, "xgboost")
        save_lime_example(xgb_model, X_train, X_test, "xgboost")

    # Save config
    config = {
        "data_path": DATA_PATH,
        "label_col": LABEL_COL,
        "feature_cols": FEATURE_COLS,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "leakage_cols": LEAKAGE_COLS,
        "subgroup_cols": SUBGROUP_COLS,
        "thresholds": thresholds,
        "random_state": RANDOM_STATE,
        "has_xgboost": HAS_XGBOOST,
        "has_shap": HAS_SHAP,
        "has_lime": HAS_LIME,
        "has_mlflow": HAS_MLFLOW,
        "lr_sweep_configs": LR_SWEEP_CONFIGS,
        "xgb_sweep_configs": XGB_SWEEP_CONFIGS,
    }
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if mlflow_ctx.enabled:
        log_params_to_mlflow({
            "data_path": DATA_PATH,
            "label_col": LABEL_COL,
            "n_features": len(FEATURE_COLS),
            "numeric_cols": NUMERIC_COLS,
            "categorical_cols": CATEGORICAL_COLS,
            "random_state": RANDOM_STATE,
            "train_size": len(X_train),
            "valid_size": len(X_valid),
            "test_size": len(X_test),
            "positive_prevalence": float(y.mean()),
            "has_xgboost": HAS_XGBOOST,
            "has_shap": HAS_SHAP,
            "has_lime": HAS_LIME,
            "lr_sweep_configs": LR_SWEEP_CONFIGS,
            "xgb_sweep_configs": XGB_SWEEP_CONFIGS,
        })
        log_metrics_to_mlflow(results)
        log_artifact_to_mlflow(os.path.join(OUTPUT_DIR, "model_comparison_metrics.csv"))
        log_artifact_to_mlflow(os.path.join(OUTPUT_DIR, "hyperparameter_sweep_results.csv"))
        log_artifact_to_mlflow(os.path.join(OUTPUT_DIR, "run_config.json"))
        log_directory_to_mlflow(EDA_DIR, "eda")
        log_directory_to_mlflow(FIGURE_DIR, "figures")
        log_directory_to_mlflow(BIAS_DIR, "bias")
        log_directory_to_mlflow(EXPLAIN_DIR, "explainability")
        mlflow.end_run()

    print(f"\nDone. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
