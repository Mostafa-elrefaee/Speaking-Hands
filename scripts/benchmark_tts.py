#!/usr/bin/env python3
"""Prediction-to-speech benchmark on the real speech subsystem with a scripted
prediction stream (no camera / model needed).

  python scripts/benchmark_tts.py --out experiments/tts_bench --events 40             # pyttsx3 subprocess (audio!)
  python scripts/benchmark_tts.py --out experiments/tts_bench --tts mock --events 100 # queue/thread overhead only
  python scripts/benchmark_tts.py --out experiments/tts_bench --compare-prespawn      # standby process on vs off
  python scripts/benchmark_tts.py --out experiments/tts_bench --compare-prewarm
  python scripts/benchmark_tts.py --out experiments/tts_bench --sweep stabilizer.confidence_threshold=0.5,0.7,0.9
  python scripts/benchmark_tts.py --out experiments/tts_bench --set tts.mode=buffered_phrase --set tts.queue_size=2
"""
import argparse
from pathlib import Path

from _common import add_config_args, build_config
from sh_research.config import apply_overrides
from sh_research.tts.benchmark import run_tts_benchmark, summary_line


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--events", type=int, default=30)
    p.add_argument("--classes", default="Hi,help,thanks,yes,no", help="labels to cycle through")
    p.add_argument("--tts", choices=["pyttsx3_subprocess", "pyttsx3", "print", "mock"], default=None)
    p.add_argument("--hold", type=float, default=0.6, help="seconds each sign is held")
    p.add_argument("--gap", type=float, default=0.3, help="low-confidence seconds between signs")
    p.add_argument("--compare-prewarm", action="store_true")
    p.add_argument("--compare-prespawn", action="store_true")
    p.add_argument("--sweep", default=None, help="key=v1,v2,... run once per value")
    a = p.parse_args()
    cfg = build_config(a)
    if a.tts:
        cfg.tts.backend = a.tts
    classes = a.classes.split(",")
    if a.compare_prewarm:
        runs = [("prewarm=false", ["tts.prewarm=false"]), ("prewarm=true", ["tts.prewarm=true"])]
    elif a.compare_prespawn:
        runs = [("prespawn=false", ["tts.prespawn=false"]), ("prespawn=true", ["tts.prespawn=true"])]
    elif a.sweep:
        key, vals = a.sweep.split("=", 1)
        runs = [(f"{key}={v}", [f"{key}={v}"]) for v in vals.split(",")]
    else:
        runs = [("baseline", [])]
    for label, ov in runs:
        c = apply_overrides(cfg, ov)
        res = run_tts_benchmark(c, classes, Path(a.out) / label.replace("=", "-").replace(".", "_"), a.events,
                                hold_s=a.hold, gap_s=a.gap, label=label)
        print(summary_line(res))
    print(f"\nresults -> {a.out}/<run>/tts_benchmark.json + tts_benchmark_events.csv")


if __name__ == "__main__":
    main()
