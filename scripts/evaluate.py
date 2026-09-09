#!/usr/bin/env python3
"""Re-evaluate a saved run on its own test split (or on another dataset) and re-measure latency.

  python scripts/evaluate.py experiments/<run_dir>
  python scripts/evaluate.py experiments/<run_dir> --data other/sequence_data.csv --label other/sequence_classifier_label.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np

from _common import *  # noqa: F401,F403
from sh_research.config import load_config
from sh_research.data import load_raw, shape_all
from sh_research.evaluation import benchmark_model_latency, classification_report, make_fast_predictor
from sh_research.training import load_model_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--data", default=None, help="evaluate on another sequence_data.csv (all rows)")
    p.add_argument("--label", default=None, help="label CSV for --data (default: the run's label path)")
    p.add_argument("--latency-samples", type=int, default=200)
    a = p.parse_args()
    run = Path(a.run_dir)
    cfg = load_config(run / "config.yaml")
    model, meta, pre = load_model_bundle(run)
    if a.data:
        cfg.data.path = a.data
    if a.label:
        cfg.data.label_path = a.label
    raw = load_raw(cfg.data)
    if raw.class_names != meta["class_names"]:
        raise SystemExit(f"label mismatch: model {meta['class_names']} vs data {raw.class_names}")
    X = pre.transform_array(shape_all(raw, pre))
    if a.data:
        idx = np.arange(len(raw))
    else:
        idx = np.array(json.loads((run / "split.json").read_text())["test"])
    predict = make_fast_predictor(model)
    probs = np.concatenate([predict(X[idx][i:i + 256]) for i in range(0, len(idx), 256)])
    rep = classification_report(raw.labels[idx], probs, meta["class_names"])
    tflite = (run / "model.tflite").read_bytes() if (run / "model.tflite").exists() else None
    lat = benchmark_model_latency(model, tuple(meta["input_shape"]), n=a.latency_samples, tflite_content=tflite)
    out = {"metrics": {k: v for k, v in rep.items() if k not in ("confusion_matrix", "per_class")},
           "per_class": rep["per_class"], "latency": lat}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
