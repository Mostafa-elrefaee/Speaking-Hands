"""Configurable LSTM encoder + shared dense head."""
from __future__ import annotations

from tensorflow import keras

from ..config import ModelConfig
from .base import dense_head, encode_and_pool, sequence_input


def build_lstm(cfg: ModelConfig, input_shape, num_classes: int, batch_size=None) -> keras.Model:
    inp = sequence_input(input_shape, batch_size)
    x = encode_and_pool(inp, [("lstm", cfg.hidden_size, cfg.num_layers, "enc")], cfg)
    out = dense_head(x, cfg, num_classes)
    return keras.Model(inp, out, name="lstm")
