"""Inference latency measurement with monotonic timing and percentiles.

Three predictor styles are measured because they differ by an order of
magnitude on CPU:

* ``model.predict``     -- convenient but ~10-30 ms per-call overhead;
* compiled ``tf.function`` -- the Keras fast path;
* **TFLite interpreter** -- what app.py actually runs (model/sequence_classifier/
  sequence_classifier.py). This is the number that matters for the realtime
  accuracy/latency trade-off.

Percentiles are reported as-measured. Nothing is extrapolated.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np


def percentiles(samples: Sequence[float]) -> Dict[str, float]:
    if not len(samples):
        return {"n": 0}
    a = np.asarray(samples, dtype=np.float64)
    d = {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
         "max": float(a.max()), "p50": float(np.percentile(a, 50))}
    if a.size >= 20:
        d["p95"] = float(np.percentile(a, 95))
    if a.size >= 100:
        d["p99"] = float(np.percentile(a, 99))
    return d


def time_calls(fn: Callable[[], object], n: int, warmup: int = 10) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return percentiles(samples)


def make_fast_predictor(model) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a Keras model in a compiled tf.function for single-sample inference."""
    import tensorflow as tf

    @tf.function(reduce_retracing=True)
    def _call(x):
        return model(x, training=False)

    def predict(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x[None]
        return _call(tf.convert_to_tensor(x, tf.float32)).numpy()

    return predict


def make_tflite_predictor(model_content: Optional[bytes] = None, model_path: Optional[str] = None,
                          num_threads: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    """TFLite predictor with the same call convention as make_fast_predictor
    (batch of 1 only -- the exported models have a fixed batch dimension)."""
    import tensorflow as tf

    it = (tf.lite.Interpreter(model_content=model_content, num_threads=num_threads) if model_content is not None
          else tf.lite.Interpreter(model_path=model_path, num_threads=num_threads))
    it.allocate_tensors()
    ii = it.get_input_details()[0]["index"]
    oi = it.get_output_details()[0]["index"]

    def predict(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        it.set_tensor(ii, x)
        it.invoke()
        return it.get_tensor(oi)

    return predict


def benchmark_model_latency(model, input_shape, n: int = 200, warmup: int = 20,
                            tflite_content: Optional[bytes] = None) -> Dict[str, Dict[str, float]]:
    """Single-sample (batch=1) latency in milliseconds for each call style."""
    x = np.random.rand(1, *input_shape).astype(np.float32)
    fast = make_fast_predictor(model)
    out = {
        "tf_function_ms": time_calls(lambda: fast(x), n, warmup),
        "model_predict_ms": time_calls(lambda: model.predict(x, verbose=0), max(n // 4, 10), max(warmup // 4, 2)),
    }
    if tflite_content is not None:
        lite = make_tflite_predictor(model_content=tflite_content)
        out["tflite_ms"] = time_calls(lambda: lite(x), n, warmup)
    return out
