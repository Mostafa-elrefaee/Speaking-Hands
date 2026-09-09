"""Shared building blocks so that all four architectures use identical heads.

Fair comparison depends on the *only* difference between models being the
temporal encoder. Pooling, normalisation, dropout and the dense classifier are
therefore built by the same functions for every architecture.

Every model takes the Speaking Hands sequence tensor [B, T, 84] (T =
data.sequence_length, 84 = landmark_utils.TOTAL_FEATURES) and returns
softmax probabilities [B, C] -- the same contract as the original
train_sequence_classifier.py model, so model/sequence_classifier/
sequence_classifier.py can run any of them from TFLite unchanged.

``pooling: last`` is implemented by the last recurrent layer returning only
its final state (return_sequences=False) rather than a Lambda slice: a Lambda
would block safe deserialisation in Keras 3 and is not needed.
"""
from __future__ import annotations

from typing import List, Optional

from tensorflow import keras
from tensorflow.keras import layers

from ..config import ModelConfig


def sequence_input(input_shape, batch_size: Optional[int] = None):
    """[B, T, F] input. batch_size=1 is used for the TFLite export clone
    (recurrent layers convert cleanly only with a fixed batch dimension)."""
    seq_len, feat_dim = input_shape
    return layers.Input(shape=(seq_len, feat_dim), batch_size=batch_size, name="sequence")


def normalization_layer(kind: str, name: str):
    if kind == "batch":
        return layers.BatchNormalization(name=name)
    if kind == "layer":
        return layers.LayerNormalization(name=name)
    if kind in ("none", "", None):
        return None
    raise ValueError(f"Unknown normalization: {kind}")


def temporal_pool(x, pooling: str, name: str = "pool"):
    """Reduce [B, T, H] -> [B, H'] for mean / max / mean_max. ('last' is
    handled by return_sequences=False in recurrent_stack.)"""
    if pooling == "mean":
        return layers.GlobalAveragePooling1D(name=f"{name}_mean")(x)
    if pooling == "max":
        return layers.GlobalMaxPooling1D(name=f"{name}_max")(x)
    if pooling == "mean_max":
        m = layers.GlobalAveragePooling1D(name=f"{name}_mean")(x)
        mx = layers.GlobalMaxPooling1D(name=f"{name}_max")(x)
        return layers.Concatenate(name=f"{name}_concat")([m, mx])
    raise ValueError(f"Unknown pooling: {pooling}")


def encode_and_pool(x, stacks, cfg: ModelConfig):
    """Run one or more recurrent stacks then reduce the temporal axis.

    stacks: list of (cell, hidden, n_layers, prefix). Every stack but the
    last returns sequences; the last returns sequences only when pooling
    needs the full temporal representation.
    """
    last_needs_seq = cfg.pooling != "last"
    for i, (cell, hidden, n_layers, prefix) in enumerate(stacks):
        final = i == len(stacks) - 1
        x = recurrent_stack(x, cell, hidden, n_layers, cfg,
                            return_sequences_last=(not final) or last_needs_seq, prefix=prefix)
    if last_needs_seq:
        x = temporal_pool(x, cfg.pooling)
    return x


def dense_head(x, cfg: ModelConfig, num_classes: int, sizes: List[int] | None = None):
    """Dense classifier: [Dense -> Norm -> Act -> Dropout]* -> softmax."""
    sizes = cfg.dense_layers if sizes is None else sizes
    for i, units in enumerate(sizes):
        x = layers.Dense(units, name=f"head_dense_{i}")(x)
        norm = normalization_layer(cfg.normalization, f"head_norm_{i}")
        if norm is not None:
            x = norm(x)
        x = layers.Activation(cfg.activation, name=f"head_act_{i}")(x)
        if cfg.head_dropout > 0:
            x = layers.Dropout(cfg.head_dropout, name=f"head_drop_{i}")(x)
    return layers.Dense(num_classes, activation="softmax", name="output")(x)


def recurrent_stack(x, cell: str, hidden: int, n_layers: int, cfg: ModelConfig,
                    return_sequences_last: bool, prefix: str):
    """Stack ``n_layers`` GRU/LSTM layers (optionally bidirectional)."""
    Cell = {"gru": layers.GRU, "lstm": layers.LSTM}[cell]
    for i in range(n_layers):
        last = i == n_layers - 1
        rnn = Cell(
            hidden,
            return_sequences=(not last) or return_sequences_last,
            dropout=cfg.dropout,
            recurrent_dropout=cfg.recurrent_dropout,
            name=f"{prefix}_{cell}_{i}",
        )
        if cfg.bidirectional:
            rnn = layers.Bidirectional(rnn, name=f"{prefix}_bi{cell}_{i}")
        x = rnn(x)
        norm = normalization_layer(cfg.normalization, f"{prefix}_norm_{i}")
        if norm is not None:
            x = norm(x)
    return x


def count_params(model: keras.Model) -> int:
    return int(model.count_params())
