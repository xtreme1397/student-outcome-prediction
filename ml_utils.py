"""Shared helpers used by both the training script and the Streamlit app."""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COL = "Target"

# Integer-coded columns that are nominal (codes carry no order) -> one-hot encoded.
NOMINAL_COLS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM / tabs / repeated spaces that the raw UCI header contains."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).replace("﻿", "")).strip() for c in df.columns]
    return df


def read_any_csv(file) -> pd.DataFrame:
    """Read a CSV that may be ';' (raw UCI) or ',' separated."""
    df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
    return clean_columns(df)


def compute_metrics(y_true, y_pred, y_proba, labels) -> dict:
    """Six required metrics. Multi-class -> macro averaging, AUC is one-vs-rest.

    AUC is averaged over the classes present in y_true, so an uploaded file that
    lacks a class (e.g. no 'Enrolled' rows) still gets a valid AUC.
    """
    y_true = np.asarray(y_true)
    present = [c for c in labels if np.any(y_true == c)]
    if len(present) < 2:
        auc = float("nan")
    else:
        auc = float(np.mean([roc_auc_score(y_true == c, y_proba[:, c]) for c in present]))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
