# Speaking Hands — real-time sign-language recognition (hands → text → speech)

Smart-glasses AI pipeline: a camera watches the user's hands, MediaPipe Hands
turns each frame into landmarks, both hands are combined into **one 84-value
vector**, a static classifier (hand shape) and a sequence classifier (motion
over 30 frames) predict the sign, the prediction is stabilised, shown on
screen and **spoken** through pyttsx3.

This repository is the original Speaking Hands project (derived from
Kazuhito Takahashi's hand-gesture-recognition-using-mediapipe, see
[Credits](#credits-and-license)) **plus an integrated research layer**
(`sh_research/`, `configs/`, `scripts/`) that adds four interchangeable
sequence architectures, reproducible experiments, a fixed-rate (20 FPS)
real-time scheduler with per-stage latency profiling, and an asynchronous
TTS path with end-to-end (prediction → speech) latency measurement.
Everything that worked before still works, with the same commands.

---

## 1. Quick start

```bash
pip install -r requirements.txt

# run live (webcam), 20 FPS processing, spoken output
python app.py

# original behaviour (process every camera frame, no scheduler)
python app.py --target_fps 0

# without speech / without the profiling overlay
python app.py --no_tts --no_overlay

# end a session with a measured report + JSON (experiments/realtime/)
python app.py --benchmark --duration 60
```

Keys inside the window: `ESC` quit · `k` then `0-9` log a static sample to
`model/keypoint_classifier/keypoint.csv` · `p` toggle the profiling overlay.

`python -m pytest tests/` runs the full test suite (no camera or audio needed).

---

## 2. Pipeline

```
camera ─▶ capture thread ─▶ latest-frame buffer ─▶ 20 FPS scheduler ─┐   (sh_research.realtime.FrameSource)
                                                                    ▼
   flip + BGR→RGB ─▶ MediaPipe Hands (≤2 hands) ─▶ landmark_utils.build_combined_vector  → 84 values
                                                                    ▼
      rolling window (30 hand-present frames) ─▶ SequenceClassifier (TFLite, softmax)   ┐
      current frame                          ─▶ KeyPointClassifier (TFLite, argmax)     ┤ gesture_output.choose_gesture
                                                                                        ▼ (sequence wins if confident)
                 gesture_output.PredictionStabilizer (confidence gate + N-frame debounce) → ONE "current gesture"
                                                          │                                  (drawn on screen)
                                                          ▼
      SpeechEventManager (duplicate policy, label→text) ─▶ bounded queue ─▶ TTS worker thread ─▶ pyttsx3 subprocess
```

* `landmark_utils.py` is the single source of truth for the feature vector:
  `[left hand 42 | right hand 42]`, wrist-relative, max-abs scaled, a missing
  hand is zero-filled. Training data and live inference use the same function.
* `dataset_utils.py` is the single loader for the CSV datasets
  (`keypoint.csv`, `sequence_data.csv`, label files) used by both training
  scripts and the research package.
* `gesture_output.py`: `GestureStabilizer` (the original N-consecutive-frames
  debounce), `PredictionStabilizer` (adds optional probability smoothing /
  majority vote / confidence gate / cooldown around it) and `SpeechWorker`
  (compatibility API over `sh_research.tts.TTSWorker`).
* Models are decoupled from time: any TFLite model with input `[1, T, 84]` and
  a softmax output is loadable by `model/sequence_classifier/sequence_classifier.py`,
  whichever of the four research architectures produced it.

---

## 3. Data collection and training (original workflow, unchanged commands)

```bash
python record_video.py                                  # record clips (optional helper)
python extract_gesture_data.py --video videos/hi.mp4 --label Hi --mode sequence --reset   # first gesture only: --reset
python extract_gesture_data.py --video videos/help.mp4 --label help --mode sequence
python train_sequence_classifier.py                     # -> model/sequence_classifier/sequence_classifier.tflite
python extract_gesture_data.py --video videos/ok.mp4 --label ok --mode static
python train_keypoint_classifier.py                     # -> model/keypoint_classifier/keypoint_classifier.tflite
python diagnose_label_mismatch.py --model <.tflite> --label <label csv>   # for any IndexError on labels
```

New option: `extract_gesture_data.py --sample_fps 20` keeps sequence frames at
a fixed rate using the video's own timestamps, so a 30-frame training window
spans the same real time as the app's 20 FPS processing window (see §6).

---

## 4. Research layer: four architectures, one training pipeline

| architecture | `model.architecture` | encoder |
|---|---|---|
| MLP baseline | `mlp` | `[30, 84]` flattened to 2520 → dense layers (no temporal modelling) |
| GRU | `gru` | stacked (optionally bidirectional) GRU |
| LSTM | `lstm` | stacked (optionally bidirectional) LSTM — the original model family |
| GRU+LSTM | `gru_lstm` | GRU stage → LSTM stage |

All four share the same input tensor, dense head, loss, optimizer, callbacks,
splits and export code (`sh_research/models/base.py`, `sh_research/training/`),
so a comparison isolates the temporal encoder.

```bash
# train one architecture over several seeds (uses model/sequence_classifier/sequence_data.csv)
python scripts/train.py --config configs/gru.yaml --seeds 1 42 123 2026
python scripts/train.py --config configs/lstm.yaml --set model.hidden_size=128 --set data.sequence_length=20

# leaderboard + plots (mean ± std over seeds, Pareto flag on accuracy vs TFLite latency)
python scripts/compare.py experiments --out experiments/comparison

# ablation grids (configs/ablations/*.yaml: hidden size, layers/bidirectional, sequence length,
# one-hand vs two-hand features, optimizer/LR, regularisation, pooling)
python scripts/ablation.py --config configs/gru.yaml --grid configs/ablations/sequence_length.yaml

# evaluate a run on its held-out test split (or another CSV)
python scripts/evaluate.py experiments/<run_dir>

# batch=1 latency of all four architectures at [1, 30, 84] (Keras + TFLite)
python scripts/benchmark_latency.py --n 300
```

Every run directory (`experiments/<stamp>_<name>_<arch>_s<seed>_<id>/`) contains
`config.yaml`, `split.json`, `model.keras`, **`model.tflite`** (fixed-batch
export, self-checked against the label count and against Keras numerically),
**`labels.csv`** (the repo's label format), `metadata.json` (input shape,
classes, preprocessing), `history.json`, `results.json` (metrics, latency,
timing, environment, dataset fingerprint), `confusion_matrix.png`, `history.png`.

Use a run in the app directly, or deploy it as the default model:

```bash
python app.py --seq_model experiments/<run_dir>          # model.tflite + labels.csv + preprocessor from the run
python scripts/deploy_model.py experiments/<run_dir>     # -> model/sequence_classifier/*.tflite/*_label.csv (+ .bak)
```

`configs/default.yaml` lists every parameter with its meaning; defaults equal
the original project's behaviour. Any key can be overridden with
`--set section.key=value` on every script and on `app.py`.

**Data note.** The repository ships 138 windows of two classes (`Hi`, `help`).
That is enough to validate the pipeline end to end but not to rank
architectures: all four reach 100 % on the 21-sample test split. Collect more
signs (and more signers — `data.signers_path` enables signer-independent
splits) before drawing conclusions. `scripts/make_synthetic_dataset.py` +
`configs/smoke.yaml` provide a synthetic smoke-test dataset only.

---

## 5. Real-time behaviour: 20 FPS scheduler and profiling

`app.py` processes frames at a **fixed rate** (`--target_fps`, default 20 =
one processed frame every 50 ms) independent of the camera's own rate:

* a capture thread reads the camera continuously into a **latest-frame
  buffer** (`realtime.max_queue_size: 1`): a slow processor sees fewer,
  fresher frames instead of a growing backlog;
* a monotonic-clock scheduler waits for the next 50 ms deadline, re-anchoring
  (never bursting) when it falls behind; frames older than
  `stale_frame_max_age_s` are skipped; older buffered frames count as dropped;
* the sequence window is cleared after `max_missing_frames` hand-less frames
  (as before) or after a processing stall longer than
  `max_sequence_gap_intervals` intervals;
* `--target_fps 0` restores the original synchronous every-frame loop (used by
  the tests and for video files).

Every stage is timed with `time.monotonic()` around the actual call
(capture, buffer acquire, frame age, scheduler lateness, image preprocessing,
MediaPipe landmarks, feature vector, window update, static model, sequence
model, probability smoothing, stabilisation, speech-event creation, drawing)
and reported as mean / p50 / p95 / p99 / max on the overlay (`p` key), in the
console at exit, and as JSON with `--benchmark`:

```bash
python app.py --benchmark --duration 60          # experiments/realtime/<stamp>_realtime_benchmark.json
python scripts/benchmark_realtime.py --duration 20              # headless, synthetic landmarks, real TFLite models
python scripts/benchmark_realtime.py --extractor mediapipe      # headless, real MediaPipe Hands on generated frames
python scripts/simulate_scheduler.py                            # scheduler + buffer policy under camera jitter/bursts
```

The report states whether the target rate was **actually sustained** (achieved
rate ≥ 95 % of target and median tick cost below the interval), names the
largest bottleneck stage and its share of the loop, and records paper-ready
metadata (architecture, model config, sequence length, target vs actual FPS,
dataset fingerprint, seed, hardware, landmark config, thresholds, TTS config).
Nothing is estimated; every number is measured in that run.

---

## 6. Sequence length vs. frame rate

The sequence classifier consumes one sample per processed frame, so a
30-frame window spans `30 / target_fps` seconds at run time (1.5 s at 20 FPS)
while training windows span `30 / recording_fps` seconds (1.0 s for a 30 FPS
recording). The app prints a **MISMATCH** warning when these differ. To align
them either extract training data at the processing rate
(`extract_gesture_data.py --sample_fps 20`, then retrain) or set
`realtime.target_fps` to the recording rate. `data.sequence_length` and
`data.sampling_fps` in the research configs let you study shorter windows
(lower recognition delay) and subsampled training data; the trained model's
settings are stored in its `metadata.json` and applied by `app.py`.

---

## 7. Speech (TTS) path and latency measurement

Speech is produced from the **stabilised** gesture only (one utterance per
held sign), through:

1. `PredictionStabilizer` → `StableEvent` (T2, with the confirming frame's
   capture time T0 and prediction time T1);
2. `SpeechEventManager` (`tts.mode`: `immediate_word` or `buffered_phrase`;
   `duplicate_cooldown_ms`, `require_prediction_change`,
   `minimum_stable_duration_ms`; `label_map` for label → spoken text);
3. a **bounded queue** (`tts.queue_size`, `overflow_policy`) — the inference
   loop never waits for audio; events that waited longer than
   `max_pending_age_ms` are expired instead of spoken;
4. the TTS worker thread and a backend: `pyttsx3_subprocess` (default — the
   original one-process-per-word approach, now with measured audio-start
   timestamps and an optional pre-spawned standby process so interpreter
   start-up is off the critical path), `pyttsx3` (in-process), `print`, `mock`.

Timestamps T0 (capture) … T7 (playback end) are recorded per spoken word;
`app.py --benchmark` and `scripts/benchmark_tts.py` report prediction →
TTS-start, TTS generation, playback-start and total prediction → speech
latencies (p50 / p95 / p99), plus counts of duplicates suppressed, expired
and dropped events:

```bash
python scripts/benchmark_tts.py --out experiments/tts_bench --events 40                 # real pyttsx3 (audio!)
python scripts/benchmark_tts.py --out experiments/tts_bench --tts mock --events 100     # thread/queue overhead only
python scripts/benchmark_tts.py --out experiments/tts_bench --compare-prespawn
python scripts/benchmark_tts.py --out experiments/tts_bench --sweep stabilizer.confidence_threshold=0.5,0.7,0.9
```

---

## 8. Repository map

| path | role |
|---|---|
| `app.py` | live application (camera → prediction → screen + speech), CLI, benchmark writer |
| `landmark_utils.py` | 84-value combined two-hand feature vector (shared by extraction and inference) |
| `dataset_utils.py` | CSV dataset / label loaders shared by training scripts and research code |
| `gesture_output.py` | `choose_gesture`, `GestureStabilizer`, `PredictionStabilizer`, `SpeechWorker` |
| `camera_utils.py`, `camera_check.py` | camera backend selection (DirectShow/MSMF fallback), diagnostics |
| `extract_gesture_data.py`, `record_video.py` | dataset creation from videos |
| `train_keypoint_classifier.py`, `train_sequence_classifier.py` | original training scripts (TFLite export + self-check) |
| `diagnose_label_mismatch.py` | model/label class-count check |
| `model/keypoint_classifier/`, `model/sequence_classifier/` | TFLite wrappers, models, label files, datasets |
| `sh_research/config.py` | typed configuration (YAML + `--set` overrides), defaults = original behaviour |
| `sh_research/data/` | dataset loading (via `dataset_utils`), preprocessing, splits, synthetic data |
| `sh_research/models/` | MLP / GRU / LSTM / GRU+LSTM factory with a shared head |
| `sh_research/training/` | optimizers, schedulers, early stopping, checkpoints, TFLite export |
| `sh_research/evaluation/` | metrics, latency measurement (Keras and TFLite), plots |
| `sh_research/experiments/` | run/aggregate/compare/ablate, benchmark metadata |
| `sh_research/realtime/` | scheduler, frame buffer, capture thread, landmark extractor, `GesturePipeline`, profiling, overlay, simulator |
| `sh_research/tts/` | backends, worker, speech-event manager, text normalisation, TTS benchmark |
| `configs/` | shipped configurations (`default`, one per architecture, `realtime`, `smoke`, `ablations/`) |
| `scripts/` | train / evaluate / compare / ablation / deploy_model / benchmark_* / simulate_scheduler / make_synthetic_dataset |
| `tests/` | full test suite (`python -m pytest tests/`) |
| `docs/INTEGRATION_REPORT.md` | audit, integration decisions, measurements |
| `keypoint_classification*.ipynb`, `point_history_classification.ipynb` | notebooks from the original project (reference only) |

---

## 9. Requirements and known constraints

* Python 3.10+; `pip install -r requirements.txt`.
* **MediaPipe:** `app.py` uses the legacy `mediapipe.solutions.hands` API,
  which exists up to MediaPipe 0.10.21 and requires `protobuf<5`. TensorFlow
  ≥ 2.20 requires `protobuf ≥ 5.28`, so the known-good combination is
  **TensorFlow ≤ 2.19 + MediaPipe ≤ 0.10.21** (the test suite and benchmarks
  in this repository were also run under TF 2.19 / MediaPipe 0.10.21 /
  Keras 3.15). With a newer TensorFlow the research/training code still runs
  (verified under TF 2.21), but MediaPipe's legacy API does not import.
* **pyttsx3** needs a speech driver: SAPI5 (Windows), NSSpeechSynthesizer
  (macOS) or eSpeak (`sudo apt install espeak-ng`, Linux). Without one,
  `app.py` fails at start-up with an actionable message; use `--no_tts` or
  `--tts_backend print`.
* `tf.lite.Interpreter` prints a deprecation notice on TF ≥ 2.20 (LiteRT);
  it still works.

---

## Credits and license

Original hand-gesture recognition and training pipeline: Kazuhito Takahashi
(https://twitter.com/KzhtTkhs), English translation and improvements by Nikita
Kiselov (https://github.com/kinivi). Speaking Hands two-hand sign pipeline,
sequence classifier, speech output and the research/realtime layer: the
Speaking Hands team (https://github.com/Mostafa-elrefaee/Speaking-Hands).

Reference: [MediaPipe](https://mediapipe.dev/).

hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).
