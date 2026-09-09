"""Shared fixtures. Tests run from the repo root:  python -m pytest tests/ -v"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    """Synthetic dataset in the repo's own CSV format (sequence_data.csv +
    sequence_classifier_label.csv + signers.json), 30 x 84 like real data."""
    from sh_research.data.synthetic import write_speaking_hands_csv
    d = tmp_path_factory.mktemp("data")
    data_csv, label_csv, signers = write_speaking_hands_csv(d, n_classes=3, per_class=16, T=30, seed=0)
    return {"dir": d, "data": data_csv, "labels": label_csv, "signers": signers}


@pytest.fixture(scope="session")
def small_cfg(synthetic_csv):
    from sh_research.config import ExperimentConfig
    cfg = ExperimentConfig(name="test", seeds=[1], latency_samples=25)
    cfg.data.path = str(synthetic_csv["data"])
    cfg.data.label_path = str(synthetic_csv["labels"])
    cfg.data.signers_path = str(synthetic_csv["signers"])
    cfg.model.hidden_size = 8
    cfg.model.num_layers = 1
    cfg.model.dense_layers = [8]
    cfg.model.mlp_hidden_layers = [16]
    cfg.training.epochs = 2
    cfg.training.verbose = 0
    cfg.training.early_stopping_patience = 0
    cfg.tts.backend = "mock"
    return cfg
