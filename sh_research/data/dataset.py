"""Dataset loading + deterministic preprocessing for research runs.

The ONLY on-disk format is the Speaking Hands one produced by
``extract_gesture_data.py --mode sequence`` and parsed by ``dataset_utils``
(the same loader the original training script uses):

    <data.path>       one row per window: [label_id, seq_len * 84 floats]
    <data.label_path> one gesture name per line (row index = label id)

Preprocessing order (identical for training and realtime inference):

    stored window [T0, 84]
      -> temporal subsampling (source_fps -> sampling_fps)
      -> length adaptation to sequence_length (tail / head / uniform)
      -> feature-group selection (feature_layout + feature_groups)
      -> optional normalisation with TRAIN statistics

``Preprocessor`` is serialised into every run's metadata.json, and app.py /
the realtime pipeline apply exactly what the model was trained with.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataset_utils import file_fingerprint, infer_seq_length, load_labels, load_sequence_csv
from landmark_utils import TOTAL_FEATURES

from ..config import DataConfig


# --------------------------------------------------------------------------- #
# Raw loading
# --------------------------------------------------------------------------- #
@dataclass
class RawDataset:
    sequences: List[np.ndarray]  # each [T_i, F]
    labels: np.ndarray  # [N] int
    class_names: List[str]
    signers: Optional[np.ndarray] = None  # [N] str or None
    source: str = ""
    label_source: str = ""
    fingerprint: str = ""  # content hash of the data CSV (dataset version id)
    stored_seq_length: int = 0

    def __len__(self):
        return len(self.sequences)

    def describe(self) -> Dict[str, object]:
        counts = np.bincount(self.labels, minlength=len(self.class_names)).tolist()
        return {"path": self.source, "label_path": self.label_source, "fingerprint": self.fingerprint,
                "n_samples": len(self), "stored_seq_length": self.stored_seq_length,
                "feature_dim": int(self.sequences[0].shape[1]) if self.sequences else 0,
                "class_names": list(self.class_names), "samples_per_class": counts,
                "has_signers": self.signers is not None}


def load_speaking_hands_csv(data_csv: str | Path, label_csv: str | Path, signers_path: str = "") -> RawDataset:
    data_csv, label_csv = Path(data_csv), Path(label_csv)
    if not data_csv.exists():
        raise FileNotFoundError(f"Sequence data CSV not found: {data_csv} (run extract_gesture_data.py --mode sequence)")
    class_names = load_labels(str(label_csv))
    seq_len = infer_seq_length(str(data_csv), TOTAL_FEATURES)
    X, y = load_sequence_csv(str(data_csv), seq_len, TOTAL_FEATURES)
    if len(class_names) == 0:
        raise ValueError(f"{label_csv} has no labels")
    bad = set(np.unique(y).tolist()) - set(range(len(class_names)))
    if bad:
        raise ValueError(f"label id(s) {sorted(bad)} in {data_csv} have no row in {label_csv} "
                         f"(run diagnose_label_mismatch.py)")
    signers = None
    if signers_path:
        m = json.loads(Path(signers_path).read_text())
        signers = np.array([str(m.get(str(i), "")) for i in range(len(y))])
        if not any(signers):
            signers = None
    return RawDataset([x.astype(np.float32) for x in X], y.astype(np.int64), class_names, signers,
                      str(data_csv), str(label_csv), file_fingerprint(str(data_csv)), seq_len)


def load_raw(cfg: DataConfig) -> RawDataset:
    return load_speaking_hands_csv(cfg.path, cfg.label_path, cfg.signers_path)


# --------------------------------------------------------------------------- #
# Preprocessing (shared by training and realtime)
# --------------------------------------------------------------------------- #
def subsample_fps(seq: np.ndarray, source_fps: float, sampling_fps: float) -> np.ndarray:
    if sampling_fps <= 0 or source_fps <= 0 or sampling_fps >= source_fps:
        return seq
    stride = source_fps / sampling_fps
    idx = np.unique(np.round(np.arange(0, len(seq), stride)).astype(int))
    idx = idx[idx < len(seq)]
    return seq[idx]


def adapt_length(seq: np.ndarray, target: int, mode: str) -> np.ndarray:
    T, F = seq.shape
    if T == target:
        return seq
    if mode == "uniform":
        idx = np.linspace(0, T - 1, target).round().astype(int)
        return seq[idx]
    pad = np.zeros((max(target - T, 0), F), dtype=seq.dtype)
    if mode == "tail":
        return np.concatenate([pad, seq[-target:]], axis=0) if T < target else seq[-target:]
    if mode == "head":
        return np.concatenate([seq[:target], pad], axis=0) if T < target else seq[:target]
    raise ValueError(f"Unknown resample mode: {mode}")


def feature_indices(layout: Dict[str, List[int]], groups: List[str], feat_dim: int) -> np.ndarray:
    if not groups:
        return np.arange(feat_dim)
    idx = []
    for g in groups:
        if g not in layout:
            raise KeyError(f"feature group {g!r} not in feature_layout {list(layout)}")
        a, b = layout[g]
        idx.extend(range(a, min(b, feat_dim)))
    return np.array(sorted(set(idx)), dtype=int)


@dataclass
class Preprocessor:
    """Deterministic, serialisable preprocessing. Fit on TRAIN split only."""

    sequence_length: int
    resample: str
    source_fps: float
    sampling_fps: float
    feature_layout: Dict[str, List[int]]
    feature_groups: List[str]
    normalize: str
    raw_feature_dim: int
    mean: Optional[List[float]] = None
    scale: Optional[List[float]] = None
    _idx: np.ndarray = field(default=None, repr=False)

    @classmethod
    def from_config(cls, cfg: DataConfig, raw_feature_dim: int = TOTAL_FEATURES) -> "Preprocessor":
        return cls(cfg.sequence_length, cfg.resample, cfg.source_fps, cfg.sampling_fps,
                   dict(cfg.feature_layout), list(cfg.feature_groups), cfg.normalize, raw_feature_dim)

    @classmethod
    def identity(cls, sequence_length: int, raw_feature_dim: int = TOTAL_FEATURES) -> "Preprocessor":
        """What the ORIGINAL pipeline does: no subsampling, no selection, no
        normalisation. Used for the legacy sequence_classifier.tflite."""
        return cls(sequence_length, "tail", 0.0, 0.0, {"all": [0, raw_feature_dim]}, [], "none", raw_feature_dim)

    @property
    def indices(self) -> np.ndarray:
        if self._idx is None:
            self._idx = feature_indices(self.feature_layout, self.feature_groups, self.raw_feature_dim)
        return self._idx

    @property
    def feature_dim(self) -> int:
        return int(len(self.indices))

    @property
    def output_shape(self) -> Tuple[int, int]:
        return (self.sequence_length, self.feature_dim)

    @property
    def is_identity(self) -> bool:
        return self.feature_dim == self.raw_feature_dim and self.mean is None

    # --- structure (no statistics) ---------------------------------------
    def shape_sequence(self, seq: np.ndarray) -> np.ndarray:
        seq = subsample_fps(seq, self.source_fps, self.sampling_fps)
        seq = adapt_length(seq, self.sequence_length, self.resample)
        return seq[:, self.indices]

    def select_features(self, frame: np.ndarray) -> np.ndarray:
        """Realtime path: feature selection for a single frame vector [F]."""
        return np.asarray(frame, dtype=np.float32)[self.indices]

    # --- statistics --------------------------------------------------------
    def fit(self, X_train: np.ndarray) -> "Preprocessor":
        flat = X_train.reshape(-1, X_train.shape[-1])
        if self.normalize == "standard":
            self.mean = flat.mean(0).tolist()
            self.scale = (flat.std(0) + 1e-6).tolist()
        elif self.normalize == "minmax":
            lo, hi = flat.min(0), flat.max(0)
            self.mean = lo.tolist()
            self.scale = (hi - lo + 1e-6).tolist()
        elif self.normalize == "none":
            self.mean = self.scale = None
        else:
            raise ValueError(f"Unknown normalize: {self.normalize}")
        return self

    def transform_array(self, X: np.ndarray) -> np.ndarray:
        """Apply normalisation to an already-shaped array [..., F']."""
        if self.mean is None:
            return np.asarray(X, dtype=np.float32)
        return ((X - np.asarray(self.mean, np.float32)) / np.asarray(self.scale, np.float32)).astype(np.float32)

    # --- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        return cls(**d)


def shape_all(raw: RawDataset, pre: Preprocessor) -> np.ndarray:
    return np.stack([pre.shape_sequence(s) for s in raw.sequences]).astype(np.float32)
