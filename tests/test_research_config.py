"""Configuration: YAML round trip, dot-key overrides, defaults = original app.py behaviour."""
import pytest

from sh_research.config import ExperimentConfig, apply_overrides, from_dict, load_config, save_config, to_dict


def test_roundtrip(tmp_path):
    cfg = ExperimentConfig()
    cfg.model.architecture = "gru_lstm"
    cfg.data.feature_groups = ["left_hand"]
    save_config(cfg, tmp_path / "c.yaml")
    back = load_config(tmp_path / "c.yaml")
    assert to_dict(back) == to_dict(cfg)


def test_overrides_typed():
    cfg = apply_overrides(ExperimentConfig(), ["model.hidden_size=128", "model.bidirectional=true",
                                               "training.learning_rate=3e-4", "data.feature_groups=[left_hand]",
                                               "realtime.target_fps=15", "tts.label_map={Hi: hello}"])
    assert cfg.model.hidden_size == 128 and cfg.model.bidirectional is True
    assert cfg.training.learning_rate == pytest.approx(3e-4)
    assert cfg.data.feature_groups == ["left_hand"] and cfg.realtime.target_fps == 15.0
    assert cfg.tts.label_map == {"Hi": "hello"}


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        apply_overrides(ExperimentConfig(), ["model.nonexistent=1"])
    with pytest.raises(KeyError):
        from_dict({"model": {"architecture": "gru", "bogus": 1}})


def test_overrides_do_not_mutate_original():
    base = ExperimentConfig()
    apply_overrides(base, ["model.hidden_size=999"])
    assert base.model.hidden_size == 64


def test_defaults_match_original_app():
    """The research defaults must reproduce app.py / gesture_output.py as they were."""
    cfg = ExperimentConfig()
    assert cfg.stabilizer.stable_count == 8 and cfg.stabilizer.confidence_threshold == 0.5
    assert cfg.stabilizer.averaging_window == 1 and cfg.stabilizer.voting_window == 0
    assert cfg.stabilizer.max_missing_frames == 5
    assert cfg.tts.backend == "pyttsx3_subprocess" and cfg.tts.duplicate_cooldown_ms == 0
    assert cfg.data.feature_layout == {"left_hand": [0, 42], "right_hand": [42, 84]}
    assert cfg.data.path.endswith("sequence_data.csv") and cfg.realtime.target_fps == 20.0


def test_shipped_configs_load():
    from pathlib import Path
    for p in sorted(Path("configs").glob("*.yaml")):
        cfg = load_config(p)
        assert cfg.model.architecture in ("mlp", "gru", "lstm", "gru_lstm"), p
