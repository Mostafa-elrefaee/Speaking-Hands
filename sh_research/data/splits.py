"""Deterministic train/val/test splits.

Two strategies:

* ``stratified`` — class-stratified random split (default).
* ``signer``     — signer-independent split: every sequence of a given signer
  lands in exactly one partition. This is the honest protocol for sign
  recognition papers (tests generalisation to unseen people). Requires
  signer ids in the dataset.

The split depends on ``split_seed`` only — NOT on the model seed — so that
multi-seed runs of different architectures see identical partitions.
Indices are saved to every experiment directory for traceability.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from ..config import DataConfig


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    strategy: str
    seed: int
    signers: Dict[str, List[str]] | None = None

    def to_dict(self) -> dict:
        d = {"strategy": self.strategy, "seed": self.seed,
             "train": self.train.tolist(), "val": self.val.tolist(), "test": self.test.tolist()}
        if self.signers:
            d["signers"] = self.signers
        return d


def _split(idx, y, frac, seed):
    """Stratified split, falling back to plain random when a class is too small."""
    try:
        return train_test_split(idx, test_size=frac, random_state=seed, stratify=y)
    except ValueError as e:  # least-populated class has < 2 members
        warnings.warn(f"Stratification impossible ({e}); using unstratified split for this partition")
        return train_test_split(idx, test_size=frac, random_state=seed)


def stratified_split(y: np.ndarray, val_frac: float, test_frac: float, seed: int) -> Split:
    idx = np.arange(len(y))
    train_idx, rest_idx = _split(idx, y, val_frac + test_frac, seed)
    rel_test = test_frac / (val_frac + test_frac)
    val_idx, test_idx = _split(rest_idx, y[rest_idx], rel_test, seed)
    return Split(np.sort(train_idx), np.sort(val_idx), np.sort(test_idx), "stratified", seed)


def signer_split(signers: np.ndarray, cfg: DataConfig) -> Split:
    uniq = sorted(str(s) for s in set(signers.tolist()))
    if len(uniq) < 3:
        raise ValueError(f"Signer split needs >=3 signers, found {len(uniq)}: {uniq}")
    val_s, test_s = list(cfg.val_signers), list(cfg.test_signers)
    if not val_s and not test_s:
        rng = np.random.RandomState(cfg.split_seed)
        order = [str(s) for s in rng.permutation(uniq)]
        n_test = max(1, int(round(len(uniq) * cfg.test_fraction)))
        n_val = max(1, int(round(len(uniq) * cfg.val_fraction)))
        test_s, val_s = order[:n_test], order[n_test:n_test + n_val]
    train_s = [s for s in uniq if s not in val_s and s not in test_s]
    if not train_s:
        raise ValueError("Signer split leaves no training signers")
    part = lambda group: np.where(np.isin(signers, group))[0]
    return Split(part(train_s), part(val_s), part(test_s), "signer", cfg.split_seed,
                 {"train": train_s, "val": val_s, "test": test_s})


def make_split(y: np.ndarray, cfg: DataConfig, signers: Optional[np.ndarray] = None) -> Split:
    if cfg.split_strategy == "stratified":
        return stratified_split(y, cfg.val_fraction, cfg.test_fraction, cfg.split_seed)
    if cfg.split_strategy == "signer":
        if signers is None:
            raise ValueError("split_strategy=signer but dataset has no signer ids")
        return signer_split(signers, cfg)
    raise ValueError(f"Unknown split_strategy: {cfg.split_strategy}")
