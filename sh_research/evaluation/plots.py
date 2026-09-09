"""Plots saved next to machine-readable results. Headless-safe (Agg backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_confusion(cm: Sequence[Sequence[int]], class_names: List[str], path: Path, title: str = "Confusion matrix") -> Path:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(class_names)),) * 2)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=90)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_history(history: Dict[str, List[float]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, ax, ttl in (("loss", axes[0], "Loss"), ("accuracy", axes[1], "Accuracy")):
        if key in history:
            ax.plot(history[key], label="train")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], label="val")
        ax.set_title(ttl); ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_comparison(rows: List[dict], metric: str, path: Path, err_key: str | None = None, label_key: str = "label") -> Path:
    labels = [r[label_key] for r in rows]
    vals = [r[metric] for r in rows]
    errs = [r.get(err_key, 0) for r in rows] if err_key else None
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(rows)), 4))
    ax.bar(labels, vals, yerr=errs, capsize=4)
    ax.set_ylabel(metric); ax.set_title(f"{metric} by configuration"); ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_accuracy_vs_latency(rows: List[dict], path: Path, acc_key: str = "test_accuracy_mean",
                             lat_key: str = "latency_p50_ms_mean", label_key: str = "label") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in rows:
        if acc_key in r and lat_key in r:
            ax.scatter(r[lat_key], r[acc_key], s=60)
            ax.annotate(r[label_key], (r[lat_key], r[acc_key]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Inference latency P50 (ms, batch=1)"); ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy / latency trade-off"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path
