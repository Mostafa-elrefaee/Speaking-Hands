#!/usr/bin/env python3
"""Run an ablation grid over a base config.

  python scripts/ablation.py --config configs/gru.yaml --grid configs/ablations/sequence_length.yaml --out experiments/abl_seq
"""
import argparse

from _common import add_config_args, build_config
from sh_research.experiments import run_ablation
from sh_research.experiments.ablation import load_grid


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--grid", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    aggs = run_ablation(build_config(a), load_grid(a.grid), a.out)
    for g in aggs:
        print(f"{' '.join(g['overrides']):60s} acc {g['test_accuracy_mean']:.4f} ± {g['test_accuracy_std']:.4f} "
              f"lat p50 {g['latency_p50_ms_mean']:.2f} ms")


if __name__ == "__main__":
    main()
