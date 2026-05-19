"""Evaluation metrics and threshold tuning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
)


@dataclass
class EvalResult:
    threshold: float
    accuracy: float
    precision_fall: float
    recall_fall: float
    f1_fall: float
    f2_fall: float
    confusion: np.ndarray
    report: str


def predict_labels(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs.reshape(-1) >= threshold).astype(int)


def evaluate_at_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> EvalResult:
    y_pred = predict_labels(probs, threshold)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", zero_division=0
    )
    f2 = fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=["no_fall", "fall"], zero_division=0
    )
    return EvalResult(
        threshold=threshold,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_fall=float(prec),
        recall_fall=float(rec),
        f1_fall=float(f1),
        f2_fall=float(f2),
        confusion=cm,
        report=report,
    )


def find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: str = "f2_fall",
    thresholds: np.ndarray | None = None,
) -> EvalResult:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.01)
    best: EvalResult | None = None
    for t in thresholds:
        result = evaluate_at_threshold(y_true, probs, float(t))
        score = getattr(result, metric)
        if best is None or score > getattr(best, metric):
            best = result
    assert best is not None
    return best


def error_analysis_df(
    y_true: np.ndarray,
    probs: np.ndarray,
    filepaths: list[str],
    threshold: float,
    split_name: str,
) -> pd.DataFrame:
    y_pred = predict_labels(probs, threshold)
    rows = []
    for i, fp in enumerate(filepaths):
        true_l = "fall" if y_true[i] == 1 else "no_fall"
        pred_l = "fall" if y_pred[i] == 1 else "no_fall"
        err_type = "TP" if y_true[i] == 1 and y_pred[i] == 1 else ""
        if y_true[i] == 0 and y_pred[i] == 1:
            err_type = "FP"
        elif y_true[i] == 1 and y_pred[i] == 0:
            err_type = "FN"
        elif y_true[i] == 0 and y_pred[i] == 0:
            err_type = "TN"
        rows.append(
            {
                "File Name": Path(fp).name,
                "True": true_l,
                "Pred": pred_l,
                "Prob": float(probs.reshape(-1)[i]),
                "Type": err_type,
                "Set": split_name,
            }
        )
    return pd.DataFrame(rows)


def metrics_summary_row(model_name: str, split_name: str, result: EvalResult) -> dict:
    return {
        "Model": model_name,
        "Split": split_name,
        "Threshold": result.threshold,
        "Accuracy": result.accuracy,
        "Precision_fall": result.precision_fall,
        "Recall_fall": result.recall_fall,
        "F1_fall": result.f1_fall,
        "F2_fall": result.f2_fall,
    }
