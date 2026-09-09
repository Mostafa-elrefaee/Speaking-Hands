"""Model factory + shared training pipeline + experiment logging on synthetic
repo-format data. Every architecture is trained for 2 epochs and exported to
the TFLite format app.py runs."""
import json
from pathlib import Path

import numpy as np
import pytest

from sh_research.config import with_seed
from sh_research.evaluation import make_tflite_predictor
from sh_research.experiments import aggregate, load_records, run_experiment, run_multi_seed, write_comparison
from sh_research.models import ARCHITECTURES
from sh_research.training import load_model_bundle


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_train_each_architecture(small_cfg, tmp_path, arch):
    cfg = with_seed(small_cfg, 1)
    cfg.model.architecture = arch
    cfg.output_dir = str(tmp_path)
    rec = run_experiment(cfg)
    run = Path(rec.output_dir)
    for f in ["config.yaml", "split.json", "model.keras", "model.tflite", "labels.csv", "metadata.json",
              "history.json", "results.json", "confusion_matrix.png", "history.png"]:
        assert (run / f).exists(), f
    assert rec.architecture == arch and rec.epochs_run == 2 and rec.param_count > 0
    assert 0.0 <= rec.test_metrics["accuracy"] <= 1.0
    assert rec.inference_latency["tf_function_ms"]["n"] == 25 and rec.inference_latency["tflite_ms"]["n"] == 25
    assert rec.tflite["num_classes"] == 3 and rec.dataset["fingerprint"]
    model, meta, pre = load_model_bundle(run)
    assert tuple(meta["input_shape"]) == (30, 84) and pre.is_identity
    assert (run / "labels.csv").read_text(encoding="utf-8-sig").split() == ["sign_0", "sign_1", "sign_2"]
    d = json.loads((run / "results.json").read_text())
    assert d["model_hyperparameters"]["architecture"] == arch
    # the exported TFLite model loads through the repo's own SequenceClassifier wrapper
    from model.sequence_classifier.sequence_classifier import SequenceClassifier
    clf = SequenceClassifier(model_path=str(run / "model.tflite"), score_th=0.0)
    assert (clf.seq_length, clf.num_features) == (30, 84)
    x = np.random.rand(30, 84).astype("float32")
    probs = clf.predict_proba(list(x))
    assert probs.shape == (3,) and np.allclose(probs, model(x[None], training=False).numpy()[0], atol=1e-4)
    assert clf(list(x)) == int(probs.argmax())


def test_multi_seed_aggregate_and_compare(small_cfg, tmp_path):
    cfg = with_seed(small_cfg, 1)
    cfg.output_dir = str(tmp_path)
    agg = run_multi_seed(cfg, seeds=[1, 2])
    assert agg["n_runs"] == 2 and agg["seeds"] == [1, 2]
    assert "test_accuracy_mean" in agg and "test_accuracy_std" in agg and "tflite_p50_ms_mean" in agg
    assert Path(agg["aggregate_path"]).exists()
    recs = load_records([tmp_path])
    assert len(recs) == 2
    df = write_comparison(recs, tmp_path / "cmp")
    assert (tmp_path / "cmp" / "leaderboard.md").exists() and len(df) == 1


def test_scheduler_and_optimizer_variants(small_cfg, tmp_path):
    for opt, sch in [("adamw", "cosine"), ("sgd", "step"), ("rmsprop", "plateau")]:
        cfg = with_seed(small_cfg, 3)
        cfg.output_dir = str(tmp_path / f"{opt}_{sch}")
        cfg.training.optimizer, cfg.training.scheduler = opt, sch
        cfg.training.weight_decay = 1e-4
        cfg.training.label_smoothing = 0.1
        cfg.training.export_tflite = False
        rec = run_experiment(cfg)
        assert rec.epochs_run == 2 and rec.tflite == {}


def test_seed_reproducibility(small_cfg, tmp_path):
    cfg = with_seed(small_cfg, 11)
    cfg.training.export_tflite = False
    a = run_experiment(cfg, run_dir=tmp_path / "a")
    b = run_experiment(cfg, run_dir=tmp_path / "b")
    assert a.test_metrics["accuracy"] == pytest.approx(b.test_metrics["accuracy"])
    assert a.val_metrics["f1_macro"] == pytest.approx(b.val_metrics["f1_macro"])


def test_preprocessor_is_reproduced_at_inference(small_cfg, tmp_path):
    """normalize=standard + one-hand feature subset must be stored in metadata
    and applied identically at inference time."""
    cfg = with_seed(small_cfg, 5)
    cfg.data.normalize = "standard"
    cfg.data.feature_groups = ["right_hand"]
    cfg.model.architecture = "gru"
    rec = run_experiment(cfg, run_dir=tmp_path / "norm")
    model, meta, pre = load_model_bundle(tmp_path / "norm")
    assert tuple(meta["input_shape"]) == (30, 42) and pre.mean is not None and rec.feature_dim == 42
    lite = make_tflite_predictor(model_path=str(tmp_path / "norm" / "model.tflite"))
    raw = np.random.rand(30, 84).astype("float32")
    x = pre.transform_array(np.stack([pre.select_features(f) for f in raw]))
    assert np.allclose(lite(x)[0], model(x[None], training=False).numpy()[0], atol=1e-4)
