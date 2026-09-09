"""Model factory: one entry point, architecture chosen by config string.

    model = build_model(cfg.model, input_shape=(T, F), num_classes=C)

Adding a fifth architecture = one new file + one line in ``REGISTRY``.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

from tensorflow import keras

from ..config import ModelConfig
from .gru import build_gru
from .gru_lstm import build_gru_lstm
from .lstm import build_lstm
from .mlp import build_mlp

REGISTRY: Dict[str, Callable] = {
    "mlp": build_mlp,
    "gru": build_gru,
    "lstm": build_lstm,
    "gru_lstm": build_gru_lstm,
}

ARCHITECTURES = tuple(REGISTRY)


def build_model(cfg: ModelConfig, input_shape: Tuple[int, int], num_classes: int,
                batch_size: int | None = None) -> keras.Model:
    """Build an uncompiled Keras model for the requested architecture.

    ``input_shape`` is (sequence_length, feature_dim); the batch axis is
    implicit (or fixed to ``batch_size`` for the TFLite export clone).
    """
    arch = cfg.architecture.lower()
    if arch not in REGISTRY:
        raise ValueError(f"Unknown architecture {arch!r}. Choose from {ARCHITECTURES}")
    return REGISTRY[arch](cfg, tuple(input_shape), num_classes, batch_size)
