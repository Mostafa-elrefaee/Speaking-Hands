"""Dataset loading through the ONE authoritative loader (dataset_utils) and preprocessing."""
import numpy as np
import pytest

from dataset_utils import infer_seq_length, load_labels, load_sequence_csv
from sh_research.config import DataConfig
from sh_research.data import Preprocessor, load_raw, make_split, shape_all
from sh_research.data.dataset import adapt_length, subsample_fps


def test_adapt_length_modes():
    seq = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
    assert adapt_length(seq, 5, "uniform").shape == (5, 2)
    tail = adapt_length(seq, 12, "tail")
    assert tail.shape == (12, 2) and np.all(tail[:2] == 0) and tail[-1, 0] == seq[-1, 0]
    head = adapt_length(seq, 4, "head")
    assert head[0, 0] == 0 and head.shape == (4, 2)


def test_subsample_fps():
    seq = np.arange(30)[:, None].astype(np.float32)
    assert len(subsample_fps(seq, 30, 15)) == 15
    assert len(subsample_fps(seq, 30, 30)) == 30
    assert len(subsample_fps(seq, 30, 10)) == 10


def test_repo_csv_format(synthetic_csv):
    """The synthetic writer produces exactly what extract_gesture_data.py writes."""
    assert infer_seq_length(str(synthetic_csv["data"])) == 30
    X, y = load_sequence_csv(str(synthetic_csv["data"]))
    assert X.shape == (48, 30, 84) and y.max() == 2
    assert load_labels(str(synthetic_csv["labels"])) == ["sign_0", "sign_1", "sign_2"]


def test_load_raw_and_preprocess(synthetic_csv):
    cfg = DataConfig(path=str(synthetic_csv["data"]), label_path=str(synthetic_csv["labels"]),
                     signers_path=str(synthetic_csv["signers"]), sequence_length=20, feature_groups=["right_hand"],
                     normalize="minmax")
    raw = load_raw(cfg)
    assert len(raw) == 48 and raw.class_names == ["sign_0", "sign_1", "sign_2"] and raw.signers is not None
    assert raw.stored_seq_length == 30 and raw.fingerprint and raw.describe()["samples_per_class"] == [16, 16, 16]
    pre = Preprocessor.from_config(cfg, 84)
    X = shape_all(raw, pre)
    assert X.shape == (48, 20, 42)
    sp = make_split(raw.labels, cfg, raw.signers)
    pre.fit(X[sp.train])
    Xn = pre.transform_array(X)
    assert Xn[sp.train].min() >= -1e-5 and Xn[sp.train].max() <= 1 + 1e-5
    back = Preprocessor.from_dict(pre.to_dict())
    assert np.allclose(back.transform_array(X), Xn)
    assert pre.select_features(np.arange(84, dtype=np.float32)).shape == (42,)


def test_identity_preprocessor_matches_original_pipeline():
    pre = Preprocessor.identity(30)
    assert pre.is_identity and pre.output_shape == (30, 84)
    v = np.random.rand(84).astype(np.float32)
    assert np.array_equal(pre.select_features(v), v)


def test_real_repo_dataset_loads():
    """The committed sequence_data.csv (138 windows x 30 x 84, 2 classes)."""
    raw = load_raw(DataConfig())
    assert raw.sequences[0].shape == (30, 84) and len(raw.class_names) == 2 and len(raw) > 0


def test_signer_split_is_disjoint(synthetic_csv):
    cfg = DataConfig(path=str(synthetic_csv["data"]), label_path=str(synthetic_csv["labels"]),
                     signers_path=str(synthetic_csv["signers"]), split_strategy="signer",
                     val_fraction=0.25, test_fraction=0.25)
    raw = load_raw(cfg)
    sp = make_split(raw.labels, cfg, raw.signers)
    s = raw.signers
    assert not (set(s[sp.train]) & set(s[sp.val])) and not (set(s[sp.train]) & set(s[sp.test]))
    assert not (set(s[sp.val]) & set(s[sp.test]))


def test_split_independent_of_model_seed(synthetic_csv):
    cfg = DataConfig(path=str(synthetic_csv["data"]), label_path=str(synthetic_csv["labels"]), split_seed=7)
    raw = load_raw(cfg)
    a, b = make_split(raw.labels, cfg), make_split(raw.labels, cfg)
    assert np.array_equal(a.test, b.test)
    with pytest.raises(ValueError):
        make_split(raw.labels, DataConfig(split_strategy="signer"), None)
