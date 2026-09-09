"""Shared CLI plumbing."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sh_research.config import ExperimentConfig, apply_overrides, load_config  # noqa: E402


def add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML config (defaults used if omitted)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override, e.g. --set model.hidden_size=128 (repeatable)")


def build_config(args) -> ExperimentConfig:
    cfg = load_config(args.config) if args.config else ExperimentConfig()
    return apply_overrides(cfg, args.overrides)
