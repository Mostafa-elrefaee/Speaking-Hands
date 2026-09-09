#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dataset_utils.py

The ONE place that knows how Speaking Hands stores training data on disk.
train_keypoint_classifier.py, train_sequence_classifier.py, the research
package (sh_research.data) and extract_gesture_data.py all import from
here, so the CSV layout is parsed by exactly one implementation.

Formats (both produced by extract_gesture_data.py):

  static   one row per frame
           [label_id, <TOTAL_FEATURES floats>]
           -> model/keypoint_classifier/keypoint.csv

  sequence one row per seq_length-frame window
           [label_id, frame1's TOTAL_FEATURES floats, frame2's ..., frameN's ...]
           -> model/sequence_classifier/sequence_data.csv

Label files are one gesture name per line, row index == label id, written
with a UTF-8 BOM (utf-8-sig) by extract_gesture_data.py.
"""
import csv
import hashlib
import os

import numpy as np

from landmark_utils import TOTAL_FEATURES


def load_labels(label_csv_path, missing_ok=False):
    """Gesture names in id order. With missing_ok=True a missing file is []
    (extract_gesture_data.py creates the file on first use)."""
    if not os.path.exists(label_csv_path):
        if missing_ok:
            return []
        raise FileNotFoundError(f"Label file not found: {label_csv_path}")
    with open(label_csv_path, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f) if row]


def _load_raw(data_csv_path):
    raw = np.loadtxt(data_csv_path, delimiter=",", dtype="float32")
    if raw.ndim == 1:  # only one sample in the file
        raw = raw.reshape(1, -1)
    if raw.size == 0:
        raise ValueError(f"{data_csv_path} is empty")
    return raw


def load_static_csv(data_csv_path, num_features=TOTAL_FEATURES):
    """-> (x [N, num_features], y [N] int)"""
    raw = _load_raw(data_csv_path)
    y = raw[:, 0].astype(int)
    x = raw[:, 1:]
    if x.shape[1] != num_features:
        raise ValueError(
            f"Each row in {data_csv_path} has {x.shape[1]} feature values, "
            f"but num_features is {num_features}. These must match."
        )
    return x, y


def infer_seq_length(data_csv_path, num_features=TOTAL_FEATURES):
    """Sequence length implied by the CSV's column count (columns - 1) /
    num_features. Raises if the columns don't divide evenly."""
    with open(data_csv_path, encoding="utf-8") as f:
        first = f.readline()
    n_cols = len(first.strip().split(",")) if first.strip() else 0
    if n_cols <= 1 or (n_cols - 1) % num_features != 0:
        raise ValueError(
            f"{data_csv_path}: {n_cols} columns is not 1 + k * {num_features}; "
            f"this file was not written with the current feature format."
        )
    return (n_cols - 1) // num_features


def load_sequence_csv(data_csv_path, seq_length=None, num_features=TOTAL_FEATURES):
    """-> (x [N, seq_length, num_features], y [N] int).
    seq_length=None infers it from the file (the normal case)."""
    if seq_length is None:
        seq_length = infer_seq_length(data_csv_path, num_features)
    raw = _load_raw(data_csv_path)
    y = raw[:, 0].astype(int)
    x_flat = raw[:, 1:]
    expected_len = seq_length * num_features
    if x_flat.shape[1] != expected_len:
        raise ValueError(
            f"Each row has {x_flat.shape[1]} feature values, but "
            f"seq_length ({seq_length}) x num_features ({num_features}) "
            f"= {expected_len}. Pass the same --seq_length you used in "
            f"extract_gesture_data.py."
        )
    return x_flat.reshape(-1, seq_length, num_features), y


def check_label_ids(y, labels, data_csv_path, label_csv_path):
    """Shared sanity checks: ids in the data must exist in the label file;
    warn (return the names) for labels that have no samples."""
    num_classes = len(labels)
    present_ids = set(np.unique(y).tolist())
    expected_ids = set(range(num_classes))
    unknown_ids = present_ids - expected_ids
    if unknown_ids:
        raise SystemExit(
            f"Found label id(s) {sorted(unknown_ids)} in {data_csv_path} "
            f"that have no matching row in {label_csv_path} (which only "
            f"defines ids 0-{num_classes - 1}). Fix the label file (or "
            f"re-extract) before training."
        )
    missing_ids = expected_ids - present_ids
    return [labels[i] for i in sorted(missing_ids)]


def file_fingerprint(path, n_bytes=1 << 20):
    """Short md5 of a file (first n_bytes + size) -- a dataset version id
    recorded in experiment logs so runs on different data are told apart."""
    h = hashlib.md5()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    h.update(str(size).encode())
    return h.hexdigest()[:12]
