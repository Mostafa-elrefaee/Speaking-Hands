#!/usr/bin/env python3
"""Build a leaderboard (CSV, Markdown, plots) from finished runs.

  python scripts/compare.py experiments/ --out experiments/comparison
  python scripts/compare.py experiments/ --by architecture model.bidirectional
"""
import argparse

from _common import *  # noqa: F401,F403  (sets sys.path)
from sh_research.experiments import load_records, to_markdown, write_comparison


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="run dirs, results.json files, or a parent directory")
    p.add_argument("--out", default="experiments/comparison")
    p.add_argument("--by", nargs="*", default=["name", "architecture"],
                   help="grouping keys: name, architecture, model.<key>, data.<key>, training.<key>")
    p.add_argument("--realtime", nargs="*", default=None,
                   help="dirs/files with realtime_benchmark.json to join per architecture (full-pipeline latency)")
    a = p.parse_args()
    recs = load_records(a.paths)
    if not recs:
        raise SystemExit("No results.json found under the given paths")
    df = write_comparison(recs, a.out, a.by, a.realtime)
    print(to_markdown(df))
    print(f"\n{len(recs)} runs -> {a.out}/leaderboard.{{csv,md}} + plots")


if __name__ == "__main__":
    main()
