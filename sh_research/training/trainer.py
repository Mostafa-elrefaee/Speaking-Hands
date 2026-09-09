"""Training pipeline shared by all architectures.

``train_model`` takes an *uncompiled* model from the factory plus the
TrainingConfig and returns the trained model, the Keras history and wall-clock
timings. Nothing here depends on the architecture.

Model persistence writes, next to each other:

* ``model.keras``   -- weights + graph (research checkpoint)
* ``model.tflite``  -- fixed-batch TFLite export, the format app.py runs
* ``labels.csv``    -- class names, one per line (utf-8-sig), the repo's label format
* ``metadata.json`` -- input shape, class names, preprocessing (feature layout,
                       selection, normalisation stats), sequence length, fps.

The TFLite export reuses the original repo's approach (train_sequence_
classifier.py): rebuild the same architecture with batch_size=1, copy the
weights, convert, then REFUSE to save unless the exported model's output size
matches the label list and its outputs match Keras numerically.
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..config import ModelConfig, TrainingConfig
from ..data.dataset import Preprocessor


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def build_optimizer(cfg: TrainingConfig) -> keras.optimizers.Optimizer:
    name = cfg.optimizer.lower()
    kw: Dict[str, Any] = {"learning_rate": cfg.learning_rate}
    if cfg.gradient_clip_norm > 0:
        kw["clipnorm"] = cfg.gradient_clip_norm
    if name == "adam":
        if cfg.weight_decay > 0:
            kw["weight_decay"] = cfg.weight_decay
        return keras.optimizers.Adam(**kw)
    if name == "adamw":
        return keras.optimizers.AdamW(weight_decay=cfg.weight_decay, **kw)
    if name == "sgd":
        if cfg.weight_decay > 0:
            kw["weight_decay"] = cfg.weight_decay
        return keras.optimizers.SGD(momentum=cfg.momentum, **kw)
    if name == "rmsprop":
        if cfg.weight_decay > 0:
            kw["weight_decay"] = cfg.weight_decay
        return keras.optimizers.RMSprop(**kw)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def build_scheduler_callbacks(cfg: TrainingConfig) -> List[keras.callbacks.Callback]:
    name = cfg.scheduler.lower()
    if name == "none":
        return []
    if name == "step":
        def step(epoch, lr):
            return cfg.learning_rate * (cfg.scheduler_gamma ** (epoch // cfg.scheduler_step_size))
        return [keras.callbacks.LearningRateScheduler(step, verbose=0)]
    if name == "cosine":
        def cosine(epoch, lr):
            frac = epoch / max(cfg.epochs - 1, 1)
            return cfg.scheduler_min_lr + 0.5 * (cfg.learning_rate - cfg.scheduler_min_lr) * (1 + math.cos(math.pi * frac))
        return [keras.callbacks.LearningRateScheduler(cosine, verbose=0)]
    if name == "plateau":
        return [keras.callbacks.ReduceLROnPlateau(monitor=cfg.monitor, factor=cfg.scheduler_gamma,
                                                  patience=cfg.scheduler_patience, min_lr=cfg.scheduler_min_lr, verbose=0)]
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def select_device(device: str) -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" or (device == "auto" and gpus):
        if not gpus:
            raise RuntimeError("device=gpu requested but no GPU visible to TensorFlow")
        return "/GPU:0"
    return "/CPU:0"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    model: keras.Model
    history: Dict[str, List[float]]
    train_time_s: float
    epochs_run: int
    best_epoch: int
    param_count: int


class _EpochTimer(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times: List[float] = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self._t0)


def train_model(model: keras.Model, cfg: TrainingConfig,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                checkpoint_path: Optional[Path] = None,
                extra_callbacks: Optional[List[keras.callbacks.Callback]] = None) -> TrainResult:
    if cfg.label_smoothing > 0:
        # Keras' sparse loss has no label smoothing; use the dense variant.
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=cfg.label_smoothing)
        num_classes = model.output_shape[-1]
        y_train_fit = keras.utils.to_categorical(y_train, num_classes)
        y_val_fit = keras.utils.to_categorical(y_val, num_classes)
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        y_train_fit, y_val_fit = y_train, y_val

    model.compile(optimizer=build_optimizer(cfg), loss=loss, metrics=["accuracy"])

    callbacks: List[keras.callbacks.Callback] = list(build_scheduler_callbacks(cfg))
    timer = _EpochTimer()
    callbacks.append(timer)
    if cfg.early_stopping_patience > 0:
        callbacks.append(keras.callbacks.EarlyStopping(monitor=cfg.monitor, patience=cfg.early_stopping_patience,
                                                       restore_best_weights=True, verbose=0))
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor=cfg.monitor,
                                                         save_best_only=True, verbose=0))
    callbacks.extend(extra_callbacks or [])

    t0 = time.perf_counter()
    with tf.device(select_device(cfg.device)):
        hist = model.fit(X_train, y_train_fit, validation_data=(X_val, y_val_fit),
                         epochs=cfg.epochs, batch_size=cfg.batch_size, callbacks=callbacks,
                         verbose=cfg.verbose, shuffle=True)
    train_time = time.perf_counter() - t0

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    history["epoch_time_s"] = timer.epoch_times
    monitor_vals = history.get(cfg.monitor, history.get("val_loss", []))
    if monitor_vals:
        best_epoch = int(np.argmin(monitor_vals)) if "loss" in cfg.monitor else int(np.argmax(monitor_vals))
    else:
        best_epoch = len(timer.epoch_times) - 1

    # If a checkpoint was written and early stopping did not already restore
    # the best weights, reload them for evaluation.
    if checkpoint_path is not None and checkpoint_path.exists() and cfg.early_stopping_patience == 0:
        model.load_weights(str(checkpoint_path))

    return TrainResult(model, history, train_time, len(timer.epoch_times), best_epoch, int(model.count_params()))


# --------------------------------------------------------------------------- #
# TFLite export (the deployment format app.py runs)
# --------------------------------------------------------------------------- #
def export_tflite(model: keras.Model, model_cfg: ModelConfig, input_shape: Tuple[int, int],
                  num_classes: int, check_samples: int = 8) -> Tuple[bytes, Dict[str, Any]]:
    """Rebuild with batch_size=1, copy weights, convert, self-check.

    Returns (tflite_bytes, info). Raises RuntimeError if the export does not
    match the label count or disagrees numerically with the Keras model --
    the same guard rails as train_sequence_classifier.py.
    """
    from ..models.factory import build_model

    model_classes = int(model.output_shape[-1])
    if model_classes != num_classes:
        raise RuntimeError(f"REFUSING TO SAVE: the model outputs {model_classes} classes but the label list "
                           f"has {num_classes}. The label file and the trained model disagree.")
    export_model = build_model(model_cfg, input_shape, num_classes, batch_size=1)
    export_model.set_weights(model.get_weights())
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    ops = "builtin"
    try:
        content = converter.convert()
    except Exception:  # pragma: no cover - depends on TF version/op support
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        content = converter.convert()
        ops = "select_tf_ops"

    interpreter = tf.lite.Interpreter(model_content=content)
    interpreter.allocate_tensors()
    in_d, out_d = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    exported_classes = int(out_d["shape"][-1])
    if exported_classes != num_classes:
        raise RuntimeError(f"REFUSING TO SAVE: exported model outputs {exported_classes} classes but "
                           f"the label list has {num_classes}.")
    if tuple(int(v) for v in in_d["shape"]) != (1, *input_shape):
        raise RuntimeError(f"REFUSING TO SAVE: exported input shape {tuple(in_d['shape'])} != {(1, *input_shape)}")
    x = np.random.rand(check_samples, *input_shape).astype(np.float32)
    ref = model(x, training=False).numpy()
    max_diff = 0.0
    for i in range(check_samples):
        interpreter.set_tensor(in_d["index"], x[i:i + 1])
        interpreter.invoke()
        max_diff = max(max_diff, float(np.abs(interpreter.get_tensor(out_d["index"]) - ref[i:i + 1]).max()))
    if max_diff > 1e-3:
        raise RuntimeError(f"REFUSING TO SAVE: TFLite output differs from Keras by {max_diff:.2e}")
    return content, {"ops": ops, "size_bytes": len(content), "max_abs_diff_vs_keras": max_diff,
                     "input_shape": [1, *input_shape], "num_classes": exported_classes}


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def write_labels_csv(path: Path, class_names: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        for name in class_names:
            w.writerow([name])


def save_model_bundle(model: keras.Model, out_dir: Path, class_names: List[str],
                      preprocessor: Preprocessor, extra: Optional[Dict[str, Any]] = None,
                      tflite_content: Optional[bytes] = None) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"model": str(out_dir / "model.keras"), "labels": str(out_dir / "labels.csv"),
             "metadata": str(out_dir / "metadata.json")}
    model.save(paths["model"])
    write_labels_csv(out_dir / "labels.csv", class_names)
    if tflite_content is not None:
        (out_dir / "model.tflite").write_bytes(tflite_content)
        paths["tflite"] = str(out_dir / "model.tflite")
    meta = {
        "input_shape": list(preprocessor.output_shape),
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "preprocessor": preprocessor.to_dict(),
        "architecture": model.name,
        "param_count": int(model.count_params()),
        "files": paths,
    }
    meta.update(extra or {})
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def load_model_bundle(model_dir: Path) -> Tuple[keras.Model, Dict[str, Any], Preprocessor]:
    model_dir = Path(model_dir)
    model = keras.models.load_model(str(model_dir / "model.keras"), compile=False)
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    pre = Preprocessor.from_dict(meta["preprocessor"])
    return model, meta, pre


def load_bundle_metadata(model_dir: Path) -> Tuple[Dict[str, Any], Preprocessor]:
    """metadata + preprocessor only (no TensorFlow/Keras model load)."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return meta, Preprocessor.from_dict(meta["preprocessor"])
