#!/usr/bin/env python3
"""Batch=1 inference latency of all four architectures at the repo's input
shape (30 frames x 84 features), measured three ways: Keras model.predict,
compiled tf.function, and the TFLite interpreter (what app.py runs).
Untrained weights -- latency does not depend on them.

  python scripts/benchmark_latency.py --n 300
  python scripts/benchmark_latency.py --seq-len 20 --set model.hidden_size=128
"""
import argparse
import json

from _common import add_config_args, build_config
from landmark_utils import TOTAL_FEATURES
from sh_research.evaluation import benchmark_model_latency
from sh_research.models import ARCHITECTURES, build_model
from sh_research.training import export_tflite


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--feat-dim", type=int, default=TOTAL_FEATURES)
    p.add_argument("--classes", type=int, default=2)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default=None, help="write JSON here")
    a = p.parse_args()
    cfg = build_config(a)
    shape = (a.seq_len, a.feat_dim)
    results = {}
    print(f"input shape (1, {a.seq_len}, {a.feat_dim}), {a.classes} classes, n={a.n}")
    print(f"{'arch':10s} {'params':>9s} {'tflite p50':>11s} {'tflite p95':>11s} {'tf.func p50':>12s} {'predict p50':>12s}")
    for arch in ARCHITECTURES:
        cfg.model.architecture = arch
        m = build_model(cfg.model, shape, a.classes)
        content, _ = export_tflite(m, cfg.model, shape, a.classes)
        lat = benchmark_model_latency(m, shape, n=a.n, tflite_content=content)
        results[arch] = {"param_count": int(m.count_params()), **lat}
        print(f"{arch:10s} {m.count_params():9d} {lat['tflite_ms']['p50']:11.3f} {lat['tflite_ms']['p95']:11.3f} "
              f"{lat['tf_function_ms']['p50']:12.3f} {lat['model_predict_ms']['p50']:12.3f}   (ms)")
    if a.out:
        from pathlib import Path
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"input_shape": [1, *shape], "n": a.n, "results": results}, indent=2))
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
