"""Configuration for the research / realtime layer of Speaking Hands.

One typed ``ExperimentConfig`` describes everything the research code and the
realtime app can vary:

* loaded from YAML         -> ``load_config("configs/gru.yaml")``
* overridden with dot-keys -> ``apply_overrides(cfg, ["model.hidden_size=128"])``
* saved back to YAML       -> ``save_config(cfg, path)`` (every experiment
  directory stores the exact configuration it ran with)

The DEFAULTS below reproduce the behaviour of the original Speaking Hands
scripts (app.py flags, train_sequence_classifier.py, gesture_output.py), so
running with no config file behaves like the original project. app.py's
argparse flags map onto these fields (see app.py:build_config).
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from landmark_utils import PER_HAND_FEATURES, TOTAL_FEATURES

# The one-frame feature vector produced by landmark_utils.build_combined_vector:
#   [ Left hand: 42 values | Right hand: 42 values ] = 84
FEATURE_LAYOUT: Dict[str, List[int]] = {
    "left_hand": [0, PER_HAND_FEATURES],
    "right_hand": [PER_HAND_FEATURES, TOTAL_FEATURES],
}

SEQ_DATA_CSV = "model/sequence_classifier/sequence_data.csv"
SEQ_LABEL_CSV = "model/sequence_classifier/sequence_classifier_label.csv"
SEQ_MODEL_TFLITE = "model/sequence_classifier/sequence_classifier.tflite"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where the sequences come from and how they are shaped before the model."""

    # Speaking Hands CSV produced by `extract_gesture_data.py --mode sequence`:
    # one row per window = [label_id, seq_length * 84 floats], plus the label
    # file (one gesture name per line, row index = label id).
    path: str = SEQ_DATA_CSV
    label_path: str = SEQ_LABEL_CSV
    dataset_id: str = ""  # free text; a content fingerprint is recorded automatically

    # Frames per model input. The CSV's own window length is inferred from its
    # column count; if it differs from this value the window is adapted
    # (``resample``). 30 = the extract_gesture_data.py default.
    sequence_length: int = 30
    resample: str = "tail"  # uniform | tail | head  (only used when lengths differ)

    # Temporal subsampling. Windows are extracted at the recording video's
    # frame rate (extract_gesture_data.py does not store it; 30 is the usual
    # webcam rate and is an ASSUMPTION unless you pass --sample_fps at
    # extraction). sampling_fps < source_fps keeps every (source/sampling)-th
    # frame before length adaptation. Equal values = no subsampling.
    source_fps: float = 30.0
    sampling_fps: float = 30.0

    # Feature layout of one frame (landmark_utils). Subsets via feature_groups
    # allow one-hand ablations; [] = all 84 values.
    feature_layout: Dict[str, List[int]] = field(default_factory=lambda: dict(FEATURE_LAYOUT))
    feature_groups: List[str] = field(default_factory=list)

    # Extra per-feature normalisation fitted on the TRAIN split only. "none"
    # keeps the original pipeline (landmark_utils already wrist-centres and
    # max-abs scales each hand).
    normalize: str = "none"  # none | standard | minmax

    # Splits (train / val / test). The original script used a single 25 %
    # test split that doubled as validation; research runs hold out a
    # separate validation split for model selection so the test split is
    # never used for tuning.
    split_strategy: str = "stratified"  # stratified | signer
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    val_signers: List[str] = field(default_factory=list)
    test_signers: List[str] = field(default_factory=list)
    split_seed: int = 0  # independent of the model seed on purpose
    # Optional JSON {row_index: signer_id} for signer-independent splits
    # (extract_gesture_data.py does not record signers; leave empty otherwise).
    signers_path: str = ""


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    architecture: str = "lstm"  # mlp | gru | lstm | gru_lstm

    # Shared recurrent settings (gru, lstm)
    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.2
    recurrent_dropout: float = 0.0
    # Temporal reduction before the dense head:
    #   last (final state) | mean | max | mean_max (concat)
    pooling: str = "last"

    # Hybrid specific (gru_lstm). <=0 -> inherit hidden_size / num_layers.
    gru_hidden_size: int = 0
    lstm_hidden_size: int = 0
    gru_num_layers: int = 0
    lstm_num_layers: int = 0

    # MLP specific: [T, F] is flattened to [T*F] (=2520 for 30x84) then
    # passed through these dense layers.
    mlp_hidden_layers: List[int] = field(default_factory=lambda: [128, 64])

    # Dense head applied after pooling / flattening for every architecture.
    dense_layers: List[int] = field(default_factory=lambda: [32])
    activation: str = "relu"
    normalization: str = "none"  # none | batch | layer
    head_dropout: float = 0.0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    optimizer: str = "adam"  # adam | adamw | sgd | rmsprop
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9  # sgd only
    batch_size: int = 32
    epochs: int = 150
    label_smoothing: float = 0.0
    gradient_clip_norm: float = 0.0  # 0 = off

    scheduler: str = "none"  # none | step | cosine | plateau
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6

    early_stopping_patience: int = 20  # 0 = off (original script: 20 on val_loss)
    monitor: str = "val_loss"
    seed: int = 42
    deterministic_ops: bool = False
    device: str = "auto"  # auto | cpu | gpu
    verbose: int = 1
    export_tflite: bool = True  # also write model.tflite (what app.py runs)


# --------------------------------------------------------------------------- #
# Prediction stabilisation  (gesture_output.PredictionStabilizer)
# --------------------------------------------------------------------------- #
@dataclass
class StabilizerConfig:
    # Sequence-classifier confidence gate. Was app.py --seq_score_th (0.5).
    confidence_threshold: float = 0.5
    # Consecutive identical raw labels before a gesture is "current" and
    # spoken. Was app.py --stable_frames / STABLE_FRAMES (8).
    stable_count: int = 8
    # Optional probability smoothing BEFORE the gate (1 = off = original).
    averaging_window: int = 1
    # Optional majority vote over the last N argmaxes (0 = off = original).
    voting_window: int = 0
    # Optional extras on top of the change-detection that already guarantees
    # one utterance per held sign (all off = original behaviour).
    cooldown_s: float = 0.0  # min seconds between two stable events
    duplicate_suppression: bool = False  # block the same label re-firing ...
    repeat_after_s: float = 3.0  # ... within this many seconds
    # Sequence buffer is cleared after this many consecutive no-hand frames
    # (app.py MAX_MISSING_FRAMES).
    max_missing_frames: int = 5


# --------------------------------------------------------------------------- #
# Realtime
# --------------------------------------------------------------------------- #
@dataclass
class RealtimeConfig:
    # Camera (app.py --device/--width/--height/--backend override these)
    camera_index: int = 0
    camera_width: int = 960
    camera_height: int = 540
    camera_backend: str = "auto"  # auto | dshow | msmf | any  (camera_utils)

    # MediaPipe Hands (app.py flags override)
    max_num_hands: int = 2
    use_static_image_mode: bool = False
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1  # MediaPipe Hands: 0 = lite (faster), 1 = full (MediaPipe default)

    # ML processing rate. Camera rate is independent. 0 = process every
    # captured frame synchronously (the original app.py behaviour, also used
    # for video files and deterministic tests).
    target_fps: float = 20.0
    max_catchup_intervals: int = 2  # re-anchor instead of bursting when this far behind
    # Capture-thread -> processing buffer. 1 = latest-frame mailbox.
    max_queue_size: int = 1
    latest_frame_strategy: bool = True
    drop_stale_frames: bool = True
    stale_frame_max_age_s: float = 0.25
    # Clear the sequence window if two consecutive PROCESSED frames are more
    # than this many target intervals apart (a processing stall). This is a
    # time-based guard; hand-absence resets are max_missing_frames above.
    max_sequence_gap_intervals: float = 3.0

    # Profiling / display
    latency_window: int = 500
    profiling_enabled: bool = True
    debug_overlay_enabled: bool = True
    log_every_s: float = 0.0  # console metrics line (0 = off; app.py shows the overlay instead)
    benchmark: bool = False  # write realtime_benchmark.json (+ speech events CSV) at exit
    benchmark_dir: str = "experiments/realtime"
    show_window: bool = True


# --------------------------------------------------------------------------- #
# TTS
# --------------------------------------------------------------------------- #
@dataclass
class TTSConfig:
    enabled: bool = True
    # pyttsx3_subprocess = the existing Speaking Hands backend (one fresh
    # pyttsx3 process per word; immune to the runAndWait-twice hang).
    # pyttsx3 = in-process (fresh engine per word). print / mock = no audio.
    backend: str = "pyttsx3_subprocess"
    rate: int = 0  # words/min, 0 = engine default (original behaviour)
    volume: float = 0.0  # 0 = engine default
    voice: str = ""
    prewarm: bool = True  # probe the engine once at start (and pre-spawn the first process)
    prespawn: bool = True  # subprocess backend: keep one initialised process on standby

    # --- speech event manager (validated prediction -> speech event) ---
    mode: str = "immediate_word"  # immediate_word | buffered_phrase
    duplicate_cooldown_ms: float = 0.0  # same label not spoken again within this window (0 = off)
    require_prediction_change: bool = False  # never re-speak a label until another label was spoken
    minimum_stable_duration_ms: float = 0.0  # extra hold time before speaking (0 = stable_count only)
    phrase_max_words: int = 4  # buffered_phrase only
    phrase_timeout_ms: float = 1500.0  # buffered_phrase only: flush after this silence

    # --- text normalisation ---
    label_map: Dict[str, str] = field(default_factory=dict)  # e.g. {fk: "okay", Hi: "hello"}
    unknown_label_policy: str = "speak_label"  # speak_label | skip
    split_label_words: bool = False  # speak "flat_hand" as "flat hand" (original: label spoken as-is)
    capitalization: str = "none"  # none | lower | sentence
    punctuation: bool = False

    # --- bounded queue + stale policy ---
    queue_size: int = 2  # pending utterances (the original SpeechWorker allowed 16)
    overflow_policy: str = "drop_oldest"  # drop_oldest | drop_newest
    max_pending_age_ms: float = 1500.0  # never speak an event that waited longer (0 = off)
    interrupt_current_audio: bool = False  # cut the current word when a new one arrives
    duplicate_window_s: float = 0.0  # extra worker-level text dedupe (0 = off)
    mock_latency_s: float = 0.05  # mock backend only
    mock_init_s: float = 0.0  # mock backend only


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    name: str = "default"
    output_dir: str = "experiments"
    seeds: List[int] = field(default_factory=lambda: [1, 42, 123, 2026])
    latency_samples: int = 200
    notes: str = ""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stabilizer: StabilizerConfig = field(default_factory=StabilizerConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)


# --------------------------------------------------------------------------- #
# (De)serialisation helpers
# --------------------------------------------------------------------------- #
def to_dict(cfg: Any) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def _from_dict(cls, data: Dict[str, Any]):
    """Recursively build ``cls`` from a dict, rejecting unknown keys."""
    if data is None:
        return cls()
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        if isinstance(f.type, type) and is_dataclass(f.type):
            kwargs[name] = _from_dict(f.type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


_SECTION_TYPES = {
    "data": DataConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "stabilizer": StabilizerConfig,
    "realtime": RealtimeConfig,
    "tts": TTSConfig,
}


def from_dict(data: Dict[str, Any]) -> ExperimentConfig:
    data = dict(data or {})
    sections = {k: _from_dict(t, data.pop(k, None)) for k, t in _SECTION_TYPES.items()}
    top = _from_dict(ExperimentConfig, data)
    for k, v in sections.items():
        setattr(top, k, v)
    return top


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return from_dict(raw)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False, allow_unicode=True)


def _coerce(current: Any, raw: str) -> Any:
    """Parse a CLI string into the type of the existing value."""
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, (list, dict)):
        return yaml.safe_load(raw)
    return raw


def apply_overrides(cfg: ExperimentConfig, overrides: List[str]) -> ExperimentConfig:
    """Apply ``section.key=value`` overrides. Returns a new config object."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        target = cfg
        for p in parts[:-1]:
            target = getattr(target, p)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise KeyError(f"Unknown config key: {key}")
        setattr(target, leaf, _coerce(getattr(target, leaf), raw))
    return cfg


def with_seed(cfg: ExperimentConfig, seed: int) -> ExperimentConfig:
    cfg = copy.deepcopy(cfg)
    cfg.training.seed = seed
    return cfg
