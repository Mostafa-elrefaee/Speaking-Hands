"""MLP baseline (non-temporal).

The temporal input [T, F] is flattened EXPLICITLY into one vector of size
T*F (frame order preserved: frame 0's 84 values first, frame T-1's last) and
passed through dense layers. There is no temporal modelling -- this is the
baseline the recurrent models are compared against.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from ..config import ModelConfig
from .base import dense_head, normalization_layer, sequence_input


def build_mlp(cfg: ModelConfig, input_shape, num_classes: int, batch_size=None) -> keras.Model:
    inp = sequence_input(input_shape, batch_size)
    x = layers.Flatten(name="flatten_T_x_F")(inp)  # [B, T, F] -> [B, T*F]
    for i, units in enumerate(cfg.mlp_hidden_layers):
        x = layers.Dense(units, name=f"mlp_dense_{i}")(x)
        norm = normalization_layer(cfg.normalization, f"mlp_norm_{i}")
        if norm is not None:
            x = norm(x)
        x = layers.Activation(cfg.activation, name=f"mlp_act_{i}")(x)
        if cfg.dropout > 0:
            x = layers.Dropout(cfg.dropout, name=f"mlp_drop_{i}")(x)
    out = dense_head(x, cfg, num_classes)
    return keras.Model(inp, out, name="mlp")
