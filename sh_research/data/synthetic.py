"""Synthetic dataset generator in the REPO's own CSV format -- ONLY for smoke
tests and CI. Class-dependent sinusoidal trajectories with noise; results on
this data say nothing about real sign recognition.

Writes exactly what extract_gesture_data.py --mode sequence writes:
    <dir>/sequence_data.csv              [label_id, T*F floats] per row
    <dir>/sequence_classifier_label.csv  one name per line (utf-8-sig)
    <dir>/signers.json                   {row_index: signer_id}   (optional)
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from landmark_utils import TOTAL_FEATURES


def make_synthetic(n_classes: int = 3, per_class: int = 40, T: int = 30, F: int = TOTAL_FEATURES,
                   n_signers: int = 4, noise: float = 0.3, seed: int = 0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, T)[:, None]
    base_freq = rng.uniform(0.5, 3.0, size=(n_classes, F))
    base_phase = rng.uniform(0, 2 * np.pi, size=(n_classes, F))
    X, y, signers = [], [], []
    for c in range(n_classes):
        for i in range(per_class):
            sig = np.sin(2 * np.pi * base_freq[c] * t + base_phase[c]) + noise * rng.randn(T, F)
            X.append(np.clip(sig, -1, 1).astype(np.float32))
            y.append(c)
            signers.append(f"S{i % n_signers}")
    return np.stack(X), np.array(y), [f"sign_{c}" for c in range(n_classes)], np.array(signers)


def write_speaking_hands_csv(out_dir: str | Path, **kw) -> Tuple[Path, Path, Path]:
    X, y, names, signers = make_synthetic(**kw)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_csv = out_dir / "sequence_data.csv"
    label_csv = out_dir / "sequence_classifier_label.csv"
    signers_json = out_dir / "signers.json"
    with open(data_csv, "w", newline="") as f:
        w = csv.writer(f)
        for seq, label in zip(X, y):
            w.writerow([int(label), *itertools.chain.from_iterable(seq.tolist())])
    with open(label_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        for n in names:
            w.writerow([n])
    signers_json.write_text(json.dumps({str(i): str(s) for i, s in enumerate(signers)}))
    return data_csv, label_csv, signers_json
