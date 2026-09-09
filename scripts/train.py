#!/usr/bin/env python3
"""Train one architecture, optionally over several seeds, with full experiment logging.

  python scripts/train.py --config configs/gru.yaml
  python scripts/train.py --config configs/lstm.yaml --seeds 1 42 123 2026
  python scripts/train.py --config configs/gru.yaml --set model.hidden_size=128 --set data.sequence_length=20

Each run writes experiments/<stamp>_<name>_<arch>_s<seed>_<id>/ with model.tflite +
labels.csv (the format app.py runs): deploy with scripts/deploy_model.py <run_dir>
or test directly with  python app.py --seq_model <run_dir>.
"""
import argparse
import json

from _common import add_config_args, build_config
from sh_research.experiments import format_pm, run_multi_seed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="override cfg.seeds")
    p.add_argument("--arch", default=None, help="shortcut for --set model.architecture=...")
    a = p.parse_args()
    cfg = build_config(a)
    if a.arch:
        cfg.model.architecture = a.arch
    agg = run_multi_seed(cfg, a.seeds)
    print(json.dumps({k: v for k, v in agg.items() if not k.endswith("_values")}, indent=2))
    print(f"\n{cfg.model.architecture}: test acc {format_pm(agg, 'test_accuracy')} | "
          f"macro-F1 {format_pm(agg, 'test_f1_macro')} | TFLite p50 {format_pm(agg, 'tflite_p50_ms', 3)} ms | "
          f"params {int(agg['param_count_mean'])} | aggregate -> {agg['aggregate_path']}")


if __name__ == "__main__":
    main()
