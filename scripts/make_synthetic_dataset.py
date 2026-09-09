#!/usr/bin/env python3
"""Generate a synthetic dataset in the repo's own CSV format (smoke tests ONLY,
never for real results).

  python scripts/make_synthetic_dataset.py data/synthetic_smoke --classes 3 --per-class 40
  python scripts/train.py --config configs/smoke.yaml
"""
import argparse

from _common import *  # noqa: F401,F403
from sh_research.data.synthetic import write_speaking_hands_csv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--signers", type=int, default=4)
    a = p.parse_args()
    data_csv, label_csv, signers = write_speaking_hands_csv(a.out_dir, n_classes=a.classes, per_class=a.per_class,
                                                            T=a.seq_len, n_signers=a.signers)
    print(f"wrote {data_csv}\n      {label_csv}\n      {signers}")


if __name__ == "__main__":
    main()
