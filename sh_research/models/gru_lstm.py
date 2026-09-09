"""GRU + LSTM hybrid.

Topology (deliberately simple, so the comparison stays fair):

    Input [B, T, 84]
      -> GRU stack  (gru_num_layers x gru_hidden_size, return_sequences=True)
      -> LSTM stack (lstm_num_layers x lstm_hidden_size)
      -> temporal reduction (last state, or mean/max/mean_max pooling)
      -> shared dense head -> softmax

The GRU is a first-stage temporal feature extractor; the LSTM adds a second
stage with its own cell state. Hybrid-specific sizes left at 0 inherit
``hidden_size`` / ``num_layers``.
"""
from __future__ import annotations

from tensorflow import keras

from ..config import ModelConfig
from .base import dense_head, encode_and_pool, sequence_input


def build_gru_lstm(cfg: ModelConfig, input_shape, num_classes: int, batch_size=None) -> keras.Model:
    gru_h = cfg.gru_hidden_size or cfg.hidden_size
    lstm_h = cfg.lstm_hidden_size or cfg.hidden_size
    gru_n = cfg.gru_num_layers or cfg.num_layers
    lstm_n = cfg.lstm_num_layers or cfg.num_layers
    inp = sequence_input(input_shape, batch_size)
    x = encode_and_pool(inp, [("gru", gru_h, gru_n, "stage1"), ("lstm", lstm_h, lstm_n, "stage2")], cfg)
    out = dense_head(x, cfg, num_classes)
    return keras.Model(inp, out, name="gru_lstm")
