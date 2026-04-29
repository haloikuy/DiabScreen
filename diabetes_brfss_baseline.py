# diabetes_brfss_baseline.py

from __future__ import annotations

import os
import json
import pickle
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
except Exception as exc:
    print(f"XGBoost unavailable; skipping XGBoost baseline. Reason: {exc}")
    XGBClassifier = None
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
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
MLFLOW_DIR = os.path.join(OUTPUT_DIR, "mlruns")
for output_path in [EDA_DIR, FIGURE_DIR, BIAS_DIR, EXPLAIN_DIR, MODEL_DIR]:
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

VALUE_LABELS = {
    "SEXVAR": {
        1.0: "Male",
        2.0: "Female",
    },
    "_RACEGR3": {
        1.0: "White only, non-Hispanic",
        2.0: "Black only, non-Hispanic",
        3.0: "Other race only, non-Hispanic",
        4.0: "Multiracial, non-Hispanic",
        5.0: "Hispanic",
    },
    "_EDUCAG": {
        1.0: "Did not graduate high school",
        2.0: "Graduated high school",
        3.0: "Attended college/technical school",
        4.0: "Graduated college/technical school",
    },
    "INCOME3": {
        1.0: "<$10k",
        2.0: "$10k-<$15k",
        3.0: "$15k-<$20k",
        4.0: "$20k-<$25k",
        5.0: "$25k-<$35k",
        6.0: "$35k-<$50k",
        7.0: "$50k-<$100k",
        8.0: "$100k-<$200k",
        9.0: "$200k+",
    },
}

VARIABLE_LABELS = {
    "SEXVAR": "Sex",
    "_AGE80": "Age",
    "_RACEGR3": "Race group",
    "_EDUCAG": "Education",
    "INCOME3": "Income",
    "GENHLTH": "General health",
    "PHYSHLTH": "Physical health days",
    "MENTHLTH": "Mental health days",
    "PRIMINS1": "Primary insurance",
    "PERSDOC3": "Personal doctor",
    "MEDCOST1": "Medical cost barrier",
    "CHECKUP1": "Recent checkup",
    "EXERANY2": "Exercise in past month",
    "_SMOKER3": "Smoking status",
    "ACEDRINK": "Alcohol drinks",
    "BPHIGH6": "High blood pressure",
    "BPMEDS1": "Blood pressure medication",
    "CHOLCHK3": "Cholesterol check",
    "_BMI5": "BMI",
    "age_group": "Age group",
}


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


def format_mean_sd(series: pd.Series, latex: bool = False) -> str:
    """Format a numeric variable as mean +/- SD."""
    values = series.dropna()
    if values.empty:
        return ""
    separator = r" $\pm$ " if latex else " +/- "
    return f"{values.mean():.1f}{separator}{values.std():.1f}"


def format_count_pct(series: pd.Series, value: object) -> str:
    """Format a categorical level as n (%), using the full group denominator."""
    denom = len(series)
    count = (series == value).sum() if not pd.isna(value) else series.isna().sum()
    pct = count / denom if denom > 0 else np.nan
    return f"{int(count):,} ({pct:.1%})" if pd.notna(pct) else f"{int(count):,} (-)"


def category_display_label(col: str, value: object) -> str:
    """Return a readable category label for Table 1."""
    if pd.isna(value):
        return "Missing"
    if col == "age_group":
        return str(value)
    return readable_group_label(col, value)


def category_sort_key(value: object) -> Tuple[int, str]:
    """Sort numeric-coded categories numerically and put missing last."""
    if pd.isna(value):
        return (1, "")
    try:
        return (0, f"{float(value):010.3f}")
    except (TypeError, ValueError):
        return (0, str(value))


def latex_escape(value: object) -> str:
    """Escape a small set of LaTeX special characters for table cells."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"$<$",
        ">": r"$>$",
    }
    return "".join(replacements.get(char, char) for char in text)


def create_full_table_one(data: pd.DataFrame) -> pd.DataFrame:
    """Create appendix-ready Table 1 with all selected feature levels by outcome."""
    rows = []
    no_diabetes_label = f"No diabetes (y=0, n={int((data['y'] == 0).sum()):,})"
    diabetes_label = f"Diabetes (y=1, n={int((data['y'] == 1).sum()):,})"
    overall_label = f"Overall (n={len(data):,})"

    for col in FEATURE_COLS + ["age_group"]:
        variable_label = VARIABLE_LABELS.get(col, col)
        if col in NUMERIC_COLS:
            rows.append({
                "variable": f"{variable_label}, mean +/- SD",
                "level": "",
                no_diabetes_label: format_mean_sd(data.loc[data["y"] == 0, col]),
                diabetes_label: format_mean_sd(data.loc[data["y"] == 1, col]),
                overall_label: format_mean_sd(data[col]),
            })
            continue

        rows.append({
            "variable": variable_label,
            "level": "",
            no_diabetes_label: "",
            diabetes_label: "",
            overall_label: "",
        })

        values = data[col]
        nonmissing_levels = sorted(
            values.dropna().unique(),
            key=category_sort_key,
        )
        levels = list(nonmissing_levels)
        if values.isna().any():
            levels.append(np.nan)

        for level in levels:
            rows.append({
                "variable": "",
                "level": category_display_label(col, level),
                no_diabetes_label: format_count_pct(data.loc[data["y"] == 0, col], level),
                diabetes_label: format_count_pct(data.loc[data["y"] == 1, col], level),
                overall_label: format_count_pct(data[col], level),
            })

    return pd.DataFrame(rows)


def save_full_table_one_latex(table_one: pd.DataFrame, save_path: str) -> None:
    """Write Table 1 as a booktabs-style LaTeX table for appendix use."""
    columns = list(table_one.columns)
    with open(save_path, "w") as f:
        f.write("% Requires \\usepackage{booktabs}\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Cohort characteristics stratified by diabetes status.}\n")
        f.write("\\label{tab:full_table1}\n")
        f.write("\\begin{tabular}{lllll}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(latex_escape(col) for col in columns) + r" \\" + "\n")
        f.write("\\midrule\n")
        for _, row in table_one.iterrows():
            is_section = row["level"] == "" and row.iloc[2:].eq("").all()
            if is_section:
                f.write("\\midrule\n")
                f.write(r"\textbf{" + latex_escape(row["variable"]) + r"} &  &  &  &  \\" + "\n")
                continue

            cells = [row[col] for col in columns]
            if row["level"] != "":
                cells[1] = r"\hspace{1em}" + latex_escape(row["level"])
                cells[0] = ""
                escaped_cells = [latex_escape(cells[0]), cells[1]] + [latex_escape(cell) for cell in cells[2:]]
            else:
                escaped_cells = [latex_escape(cell) for cell in cells]
            f.write(" & ".join(escaped_cells) + r" \\" + "\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


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

    full_table_one = create_full_table_one(data)
    full_table_one.to_csv(os.path.join(EDA_DIR, "full_table1_cohort_characteristics.csv"), index=False)
    save_full_table_one_latex(
        full_table_one,
        os.path.join(EDA_DIR, "full_table1_cohort_characteristics.tex"),
    )

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


def readable_group_label(subgroup_col: str, group_value: object) -> str:
    """Convert selected BRFSS coded subgroup values to report-friendly labels."""
    if pd.isna(group_value) or str(group_value).lower() == "nan":
        return "Missing"

    if subgroup_col == "age_group":
        return str(group_value)

    label_map = VALUE_LABELS.get(subgroup_col, {})
    try:
        numeric_value = float(group_value)
    except (TypeError, ValueError):
        return str(group_value)

    return label_map.get(numeric_value, str(group_value))


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
            m["group_label"] = readable_group_label(subgroup_col, group_value)
            m["n"] = len(group_df)
            rows.append(m)
        except Exception:
            continue
    return pd.DataFrame(rows)


def plot_subgroup_metric(
    subgroup_df: pd.DataFrame,
    subgroup_col: str,
    metric: str,
    title: str,
    save_path: str,
) -> None:
    """Save a subgroup metric bar plot for bias assessment figures."""
    if subgroup_df.empty or metric not in subgroup_df.columns:
        return

    plot_df = subgroup_df.sort_values(metric).copy()
    height = max(4.0, 0.45 * len(plot_df) + 1.5)

    plt.figure(figsize=(8, height))
    plt.barh(plot_df["group_label"].astype(str), plot_df[metric])
    plt.xlabel(metric)
    plt.ylabel(subgroup_col)
    plt.title(title)
    x_lower, x_upper = metric_axis_limits(plot_df[metric].tolist(), lower_zero=False, pad=0.04)
    plt.xlim(x_lower, min(1.05, x_upper + 0.02))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def ordered_subgroup_df(subgroup_df: pd.DataFrame, subgroup_col: str) -> pd.DataFrame:
    """Return subgroup rows in a report-friendly order."""
    if subgroup_df.empty:
        return subgroup_df
    subgroup_df = subgroup_df[subgroup_df["group_label"].astype(str) != "Missing"].copy()
    if subgroup_df.empty:
        return subgroup_df

    order_map = {}
    if subgroup_col == "age_group":
        order_map = {"18-34": 0, "35-49": 1, "50-64": 2, "65-80": 3}
    elif subgroup_col == "SEXVAR":
        order_map = {"Female": 0, "Male": 1}

    if not order_map:
        return subgroup_df.sort_values("group_label")

    return (
        subgroup_df.assign(_order=subgroup_df["group_label"].map(order_map).fillna(999))
        .sort_values(["_order", "group_label"])
        .drop(columns=["_order"])
    )


def metric_axis_limits(values: List[float], lower_zero: bool = False, pad: float = 0.06) -> Tuple[float, float]:
    """Choose readable y-axis limits for bounded performance metrics."""
    clean_values = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if clean_values.empty:
        return 0.0, 1.0

    min_value = float(clean_values.min())
    max_value = float(clean_values.max())
    value_range = max_value - min_value
    padding = max(pad, value_range * 0.25)

    lower = 0.0 if lower_zero else max(0.0, min_value - padding)
    upper = min(1.0, max_value + padding)
    if upper - lower < 0.12:
        midpoint = (upper + lower) / 2
        lower = 0.0 if lower_zero else max(0.0, midpoint - 0.06)
        upper = min(1.0, midpoint + 0.06)

    if upper <= lower:
        upper = min(1.0, lower + 0.1)
    return lower, upper


def save_input_bias_outputs(data: pd.DataFrame) -> None:
    """Assess label prevalence across input subgroups before model training."""
    age_df = (
        data.groupby("age_group", dropna=False)["y"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .rename(columns={"mean": "diabetes_prevalence", "count": "n", "sum": "n_diabetes"})
    )
    age_df["age_group_label"] = age_df["age_group"].astype(str)
    age_df = ordered_subgroup_df(
        age_df.rename(columns={"age_group_label": "group_label"}),
        "age_group",
    )
    age_df.to_csv(os.path.join(BIAS_DIR, "input_age_group_label_prevalence.csv"), index=False)

    plt.figure(figsize=(8.5, 5.2))
    ax = plt.gca()
    bars = ax.bar(age_df["group_label"], age_df["diabetes_prevalence"], color="#3ba3cf", edgecolor="#2c6f8f")
    for bar, prevalence, n in zip(bars, age_df["diabetes_prevalence"], age_df["n"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{prevalence:.2%}\n(n={int(n):,})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Observed diabetes prevalence")
    ax.set_xlabel("Age group")
    ax.set_title("Observed Diabetes Label Prevalence by Age Group", fontweight="bold")
    ax.set_ylim(0, min(1.0, max(age_df["diabetes_prevalence"].max() * 1.25, 0.05)))
    ax.grid(axis="y", alpha=0.25, linestyle="-")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BIAS_DIR, "input_age_group_label_prevalence.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_subgroup_outputs(
    eval_df: pd.DataFrame,
    model_name: str,
    threshold: float,
) -> None:
    summary_rows = []
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
        for metric in ["auroc", "auprc", "recall_sensitivity", "specificity", "precision_ppv"]:
            if subgroup_df.empty or metric not in subgroup_df.columns:
                continue
            summary_rows.append({
                "model": model_name,
                "subgroup": subgroup_col,
                "metric": metric,
                "min": subgroup_df[metric].min(),
                "max": subgroup_df[metric].max(),
                "range": subgroup_df[metric].max() - subgroup_df[metric].min(),
                "min_group": subgroup_df.loc[subgroup_df[metric].idxmin(), "group_label"],
                "max_group": subgroup_df.loc[subgroup_df[metric].idxmax(), "group_label"],
            })

        if subgroup_col in ["_RACEGR3", "INCOME3"]:
            plot_subgroup_metric(
                subgroup_df,
                subgroup_col,
                "auroc",
                f"{model_name}: AUROC by {subgroup_col}",
                os.path.join(BIAS_DIR, f"{model_name}_{subgroup_col}_auroc.png"),
            )
            plot_subgroup_metric(
                subgroup_df,
                subgroup_col,
                "recall_sensitivity",
                f"{model_name}: Sensitivity by {subgroup_col}",
                os.path.join(BIAS_DIR, f"{model_name}_{subgroup_col}_sensitivity.png"),
            )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(BIAS_DIR, f"{model_name}_subgroup_metric_ranges.csv"),
            index=False,
        )


def display_model_name(model_name: str) -> str:
    label_map = {
        "logistic_regression": "Logistic Regression",
        "xgboost": "XGBoost",
    }
    return label_map.get(model_name, model_name)


def combined_subgroup_tables(
    eval_datasets: Dict[str, pd.DataFrame],
    thresholds: Dict[str, float],
    subgroup_col: str,
) -> Dict[str, pd.DataFrame]:
    tables = {}
    for model_name, eval_df in eval_datasets.items():
        if model_name not in thresholds:
            continue
        tables[model_name] = subgroup_metrics(
            eval_df,
            subgroup_col=subgroup_col,
            prob_col="y_prob",
            label_col="y",
            threshold=thresholds[model_name],
        )
    return tables


def plot_combined_subgroup_performance_metrics(
    subgroup_tables: Dict[str, pd.DataFrame],
    subgroup_col: str,
    save_path: str,
) -> None:
    """Plot age/sex subgroup metrics for all available models in one figure."""
    metric_labels = [
        ("auroc", "AUROC"),
        ("recall_sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("precision_ppv", "PPV"),
        ("fpr", "FPR"),
        ("fnr", "FNR"),
    ]
    if not subgroup_tables:
        return

    readable_col = "Sex" if subgroup_col == "SEXVAR" else "Age group"
    palette = ["#3ba3cf", "#f47c51", "#6fbf73", "#b07cc6"]
    n_models = len(subgroup_tables)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(6.6 * n_models, 5.6),
        sharey=True,
        squeeze=False,
    )
    x = np.arange(len(metric_labels))
    all_values = []

    for ax, (model_name, subgroup_df) in zip(axes[0], subgroup_tables.items()):
        if subgroup_df.empty:
            continue
        plot_df = ordered_subgroup_df(subgroup_df, subgroup_col).copy()
        if plot_df.empty:
            continue
        plot_df["fpr"] = 1.0 - plot_df["specificity"]
        plot_df["fnr"] = 1.0 - plot_df["recall_sensitivity"]
        groups = plot_df["group_label"].astype(str).tolist()
        width = min(0.8 / max(len(groups), 1), 0.22)

        for i, (_, row) in enumerate(plot_df.iterrows()):
            values = [row[metric] for metric, _ in metric_labels]
            all_values.extend(values)
            offsets = x + (i - (len(groups) - 1) / 2) * width
            bars = ax.bar(
                offsets,
                values,
                width=width,
                label=str(row["group_label"]),
                color=palette[i % len(palette)],
                edgecolor="#666666",
                linewidth=0.6,
                alpha=0.95,
            )
            for bar, value in zip(bars, values):
                if pd.notna(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90 if len(groups) > 3 else 0,
                    )

        ax.set_title(display_model_name(model_name), fontweight="bold")
        ax.set_xlabel("Performance Metric")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metric_labels], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25, linestyle="-")
        ax.set_axisbelow(True)
        ax.legend(title=readable_col, fontsize=8, frameon=True)

    y_lower, y_upper = metric_axis_limits(all_values, lower_zero=False, pad=0.05)
    y_upper = min(1.12, y_upper + 0.04)
    for ax in axes[0]:
        ax.set_ylim(y_lower, y_upper)

    axes[0][0].set_ylabel("Score")
    fig.suptitle(f"{readable_col} Subgroup Performance Metrics by Model", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_subgroup_metric_line(
    subgroup_tables: Dict[str, pd.DataFrame],
    subgroup_col: str,
    metric: str,
    metric_label: str,
    save_path: str,
) -> None:
    """Plot AUROC or sensitivity by subgroup with all models on one axis."""
    if not subgroup_tables:
        return

    readable_col = "Sex" if subgroup_col == "SEXVAR" else "Age group"
    plt.figure(figsize=(8.5, 5.0))
    ax = plt.gca()
    colors = {
        "logistic_regression": "#3ba3cf",
        "xgboost": "#f47c51",
    }
    all_values = []
    for model_name, subgroup_df in subgroup_tables.items():
        if subgroup_df.empty or metric not in subgroup_df.columns:
            continue
        plot_df = ordered_subgroup_df(subgroup_df, subgroup_col)
        if plot_df.empty:
            continue
        all_values.extend(plot_df[metric].tolist())
        ax.plot(
            plot_df["group_label"].astype(str),
            plot_df[metric],
            marker="o",
            linewidth=2,
            color=colors.get(model_name),
            label=display_model_name(model_name),
        )
        for _, row in plot_df.iterrows():
            if pd.notna(row[metric]):
                ax.text(str(row["group_label"]), row[metric] + 0.008, f"{row[metric]:.2f}", ha="center", fontsize=8)

    ax.set_title(f"{metric_label} by {readable_col}", fontweight="bold")
    ax.set_xlabel(readable_col)
    ax.set_ylabel(metric_label)
    y_lower, y_upper = metric_axis_limits(all_values, lower_zero=False, pad=0.035)
    ax.set_ylim(y_lower, min(1.05, y_upper + 0.03))
    ax.grid(axis="y", alpha=0.25, linestyle="-")
    ax.set_axisbelow(True)
    ax.legend(title="Model", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_combined_subgroup_outputs(
    eval_datasets: Dict[str, pd.DataFrame],
    thresholds: Dict[str, float],
) -> None:
    """Save report-ready subgroup comparison figures with all models in each plot."""
    for subgroup_col in ["age_group", "SEXVAR"]:
        tables = combined_subgroup_tables(eval_datasets, thresholds, subgroup_col)
        if not tables:
            continue
        plot_combined_subgroup_performance_metrics(
            tables,
            subgroup_col,
            os.path.join(BIAS_DIR, f"combined_{subgroup_col}_performance_metrics.png"),
        )
        plot_combined_subgroup_metric_line(
            tables,
            subgroup_col,
            "auroc",
            "AUROC",
            os.path.join(BIAS_DIR, f"combined_{subgroup_col}_auroc_line.png"),
        )
        plot_combined_subgroup_metric_line(
            tables,
            subgroup_col,
            "recall_sensitivity",
            "Sensitivity",
            os.path.join(BIAS_DIR, f"combined_{subgroup_col}_sensitivity_line.png"),
        )


def save_model_artifact(model: Pipeline, model_name: str) -> str:
    """Persist the final sklearn pipeline for reproducibility and MLflow artifacts."""
    save_path = os.path.join(MODEL_DIR, f"{model_name}_pipeline.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    return save_path


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


def original_feature_name(transformed_feature_name: str) -> str:
    """Map transformed one-hot/scaled feature names back to the original BRFSS variable."""
    name = transformed_feature_name
    if "__" in name:
        name = name.split("__", 1)[1]

    if name in FEATURE_COLS:
        return name

    for col in sorted(CATEGORICAL_COLS, key=len, reverse=True):
        if name == col or name.startswith(f"{col}_"):
            return col

    return name


def normalize_shap_array(shap_values: object) -> np.ndarray:
    """Return a 2D SHAP array for the positive class when needed."""
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    elif hasattr(shap_values, "values"):
        shap_values = shap_values.values

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, -1]
    return shap_array


def grouped_shap_importance(
    shap_array: np.ndarray,
    transformed_feature_names: np.ndarray,
) -> pd.DataFrame:
    """Aggregate one-hot SHAP magnitudes back to original BRFSS variables."""
    original_names = [original_feature_name(name) for name in transformed_feature_names]
    rows = []
    for col in [feature for feature in FEATURE_COLS if feature in original_names]:
        idx = [i for i, original in enumerate(original_names) if original == col]
        rows.append({
            "feature": col,
            "mean_abs_shap": np.abs(shap_array[:, idx]).sum(axis=1).mean(),
            "n_transformed_columns": len(idx),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


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

        shap_array = normalize_shap_array(shap_values)
        mean_abs_shap = np.abs(shap_array).mean(axis=0)
        shap_importance_df = pd.DataFrame({
            "feature": feature_names,
            "original_feature": [original_feature_name(name) for name in feature_names],
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        shap_importance_df.to_csv(
            os.path.join(EXPLAIN_DIR, f"{model_name}_shap_top_features.csv"),
            index=False,
        )

        grouped_importance_df = grouped_shap_importance(shap_array, feature_names)
        grouped_importance_df.to_csv(
            os.path.join(EXPLAIN_DIR, f"{model_name}_shap_grouped_top_features.csv"),
            index=False,
        )

        plt.figure(figsize=(8, 6))
        plot_df = grouped_importance_df.head(15).iloc[::-1]
        plt.barh(plot_df["feature"], plot_df["mean_abs_shap"])
        plt.xlabel("Mean summed absolute SHAP value")
        plt.title(f"{model_name} grouped SHAP importance")
        plt.tight_layout()
        plt.savefig(
            os.path.join(EXPLAIN_DIR, f"{model_name}_shap_grouped_importance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

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
        pd.DataFrame(
            explanation.as_list(),
            columns=["feature_condition", "local_weight"],
        ).to_csv(
            os.path.join(EXPLAIN_DIR, f"{model_name}_lime_example_weights.csv"),
            index=False,
        )
        lime_fig = explanation.as_pyplot_figure()
        lime_fig.tight_layout()
        lime_fig.savefig(
            os.path.join(EXPLAIN_DIR, f"{model_name}_lime_example.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(lime_fig)
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


def evaluate_model_on_splits(
    model: Pipeline,
    split_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    threshold: float,
    model_name: str,
) -> pd.DataFrame:
    """Evaluate a fitted model on train/validation/test splits using one threshold."""
    rows = []
    for split_name, (X_split, y_split) in split_data.items():
        y_prob = model.predict_proba(X_split)[:, 1]
        rows.append({
            "model": model_name,
            "split": split_name,
            **compute_metrics(y_split, y_prob, threshold=threshold),
        })
    return pd.DataFrame(rows)


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

    valid_prob = model.predict_proba(X_valid)[:, 1]
    threshold = choose_threshold_by_f1(y_valid, valid_prob)
    valid_metrics = compute_metrics(y_valid, valid_prob, threshold=threshold)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_prob, threshold=threshold)

    metrics_df = pd.DataFrame([{
        "model": model_label,
        "model_family": model_family,
        "config": json.dumps(params, sort_keys=True),
        "threshold_selection_split": "validation",
        "validation_auroc": valid_metrics["auroc"],
        "validation_auprc": valid_metrics["auprc"],
        "validation_f1": valid_metrics["f1"],
        "validation_precision_ppv": valid_metrics["precision_ppv"],
        "validation_recall_sensitivity": valid_metrics["recall_sensitivity"],
        "validation_specificity": valid_metrics["specificity"],
        "validation_brier_score": valid_metrics["brier_score"],
        "auroc": test_metrics["auroc"],
        "auprc": test_metrics["auprc"],
        "f1": test_metrics["f1"],
        "precision_ppv": test_metrics["precision_ppv"],
        "recall_sensitivity": test_metrics["recall_sensitivity"],
        "specificity": test_metrics["specificity"],
        "brier_score": test_metrics["brier_score"],
        "threshold": threshold,
    }])

    if HAS_MLFLOW:
        with mlflow.start_run(run_name=model_label, nested=True):
            log_params_to_mlflow({"model_family": model_family, **params})
            log_metrics_to_mlflow(metrics_df[[
                "model",
                "validation_auroc",
                "validation_auprc",
                "validation_f1",
                "validation_precision_ppv",
                "validation_recall_sensitivity",
                "validation_specificity",
                "validation_brier_score",
                "auroc",
                "auprc",
                "f1",
                "precision_ppv",
                "recall_sensitivity",
                "specificity",
                "brier_score",
                "threshold",
            ]])

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
    save_input_bias_outputs(data)

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
    lr_model_path = save_model_artifact(best_lr_model, lr_model_name)

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
        xgb_model_path = save_model_artifact(xgb_model, "xgboost")

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

    split_data = {
        "train": (X_train, y_train),
        "validation": (X_valid, y_valid),
        "test": (X_test, y_test),
    }
    split_metric_tables = [
        evaluate_model_on_splits(best_lr_model, split_data, lr_thr, "logistic_regression")
    ]
    if HAS_XGBOOST:
        split_metric_tables.append(
            evaluate_model_on_splits(xgb_model, split_data, thresholds["xgboost"], "xgboost")
        )
    split_metrics = pd.concat(split_metric_tables, ignore_index=True)
    split_metrics.to_csv(os.path.join(OUTPUT_DIR, "split_comparison_metrics.csv"), index=False)

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

    save_combined_subgroup_outputs(eval_datasets, thresholds)

    # Save config
    config = {
        "data_path": DATA_PATH,
        "label_col": LABEL_COL,
        "feature_cols": FEATURE_COLS,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "leakage_cols": LEAKAGE_COLS,
        "subgroup_cols": SUBGROUP_COLS,
        "value_labels": VALUE_LABELS,
        "thresholds": thresholds,
        "model_artifacts": {
            "logistic_regression": lr_model_path,
            "xgboost": xgb_model_path if HAS_XGBOOST else None,
        },
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
        log_artifact_to_mlflow(os.path.join(OUTPUT_DIR, "split_comparison_metrics.csv"))
        log_artifact_to_mlflow(os.path.join(OUTPUT_DIR, "hyperparameter_sweep_results.csv"))
        log_artifact_to_mlflow(os.path.join(OUTPUT_DIR, "run_config.json"))
        log_directory_to_mlflow(EDA_DIR, "eda")
        log_directory_to_mlflow(FIGURE_DIR, "figures")
        log_directory_to_mlflow(BIAS_DIR, "bias")
        log_directory_to_mlflow(EXPLAIN_DIR, "explainability")
        log_directory_to_mlflow(MODEL_DIR, "models")
        mlflow.end_run()

    print(f"\nDone. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
