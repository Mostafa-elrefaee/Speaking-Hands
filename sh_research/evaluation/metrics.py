"""Classification metrics computed identically for every architecture."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             precision_score, recall_score, top_k_accuracy_score)


def classification_report(y_true: np.ndarray, probs: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    y_pred = probs.argmax(1)
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_samples": int(len(y_true)),
    }
    if len(class_names) > 2 and probs.shape[1] == len(class_names):
        k = min(3, len(class_names) - 1)
        out[f"top{k}_accuracy"] = float(top_k_accuracy_score(y_true, probs, k=k, labels=labels))
    return out


def summarize_for_table(report: Dict[str, Any]) -> Dict[str, float]:
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    return {k: report[k] for k in keys if k in report}
