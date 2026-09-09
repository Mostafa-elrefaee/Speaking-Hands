#!/usr/bin/env python3
"""Run the deterministic scheduler/buffer simulation scenarios (no hardware needed).

  python scripts/simulate_scheduler.py --target-fps 20 --duration 10
"""
import argparse

from _common import *  # noqa: F401,F403
from sh_research.realtime.simulate import standard_scenarios


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-fps", type=float, default=20.0)
    p.add_argument("--duration", type=float, default=10.0)
    a = p.parse_args()
    for r in standard_scenarios(a.target_fps, a.duration):
        print(r.summary())


if __name__ == "__main__":
    main()
