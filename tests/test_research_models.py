import numpy as np
import pytest

from sh_research.config import ModelConfig
from sh_research.models import ARCHITECTURES, build_model


@pytest.mark.parametrize("arch", ARCHITECTURES)
@pytest.mark.parametrize("pooling", ["last", "mean", "mean_max"])
def test_build_and_forward(arch, pooling):
    cfg = ModelConfig(architecture=arch, hidden_size=8, num_layers=2, pooling=pooling, dense_layers=[8], mlp_hidden_layers=[16])
    m = build_model(cfg, (12, 20), 5)
    y = m(np.random.rand(3, 12, 20).astype("float32"), training=False).numpy()
    assert y.shape == (3, 5)
    assert np.allclose(y.sum(1), 1.0, atol=1e-4)


@pytest.mark.parametrize("arch", ["gru", "lstm", "gru_lstm"])
def test_bidirectional_and_norm_change_params(arch):
    base = ModelConfig(architecture=arch, hidden_size=8, num_layers=1, dense_layers=[8])
    p0 = build_model(base, (10, 6), 3).count_params()
    bi = ModelConfig(architecture=arch, hidden_size=8, num_layers=1, dense_layers=[8], bidirectional=True)
    p1 = build_model(bi, (10, 6), 3).count_params()
    assert p1 > p0


def test_hybrid_specific_sizes():
    cfg = ModelConfig(architecture="gru_lstm", hidden_size=8, num_layers=1, gru_hidden_size=4, lstm_hidden_size=16,
                      gru_num_layers=2, lstm_num_layers=1, dense_layers=[8])
    m = build_model(cfg, (10, 6), 3)
    names = [l.name for l in m.layers]
    assert sum(n.startswith("stage1_gru") for n in names) == 2
    assert sum(n.startswith("stage2_lstm") for n in names) == 1


def test_mlp_flattens_explicitly():
    cfg = ModelConfig(architecture="mlp", mlp_hidden_layers=[8], dense_layers=[4])
    m = build_model(cfg, (7, 5), 2)
    flat = m.get_layer("flatten_T_x_F")
    assert flat.output.shape[-1] == 35


def test_unknown_architecture():
    with pytest.raises(ValueError):
        build_model(ModelConfig(architecture="transformer"), (10, 6), 3)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_tflite_export_matches_keras_on_repo_shape(arch):
    """Every architecture must export to the fixed-batch TFLite format app.py
    runs, at the real [1, 30, 84] shape, and agree with Keras numerically."""
    from sh_research.training import export_tflite
    from sh_research.evaluation import make_tflite_predictor
    cfg = ModelConfig(architecture=arch, hidden_size=8, num_layers=2, dense_layers=[8], mlp_hidden_layers=[16],
                      bidirectional=(arch == "gru"), pooling=("mean_max" if arch == "lstm" else "last"))
    m = build_model(cfg, (30, 84), 3)
    content, info = export_tflite(m, cfg, (30, 84), 3)
    assert info["num_classes"] == 3 and info["input_shape"] == [1, 30, 84] and info["max_abs_diff_vs_keras"] < 1e-3
    lite = make_tflite_predictor(model_content=content)
    x = np.random.rand(30, 84).astype("float32")
    assert np.allclose(lite(x)[0], m(x[None], training=False).numpy()[0], atol=1e-4)


def test_tflite_export_refuses_class_mismatch():
    from sh_research.training import export_tflite
    cfg = ModelConfig(architecture="gru", hidden_size=4, num_layers=1, dense_layers=[4])
    m = build_model(cfg, (10, 84), 2)
    with pytest.raises(RuntimeError, match="REFUSING"):
        export_tflite(m, cfg, (10, 84), 3)
