# Integration report — `sh_research` into Speaking Hands

Date: 2026-09-09. Base repository: `Mostafa-elrefaee/Speaking-Hands` (main).
The existing project was treated as the source of truth; the research package
(`sh_research`, previously built without the real repository) was adapted to
it, never the other way round.

## A. Audit of the existing project (facts the integration was built on)

| aspect | reality in the repo |
|---|---|
| landmark extractor | MediaPipe **Hands** (legacy `mp.solutions.hands`), max 2 hands, `min_detection_confidence=0.7`, `min_tracking_confidence=0.5` |
| feature vector | `landmark_utils.build_combined_vector` → **84 values**: left hand 0–41, right hand 42–83; wrist-relative, max-abs scaled per hand; missing hand = zeros |
| static model | `model/keypoint_classifier/keypoint_classifier.tflite`: `[1,84] → [1,4]` (`fk`, `stop`, `ok`, `finish`); wrapper returns argmax |
| sequence model | `model/sequence_classifier/sequence_classifier.tflite`: `[1,30,84] → [1,2]` (`Hi`, `help`); 2-layer LSTM 64→32 + dense 32; wrapper returns argmax if p ≥ `score_th` (0.5) else −1 |
| dataset | `sequence_data.csv`: 138 rows `[label, 2520 floats]` (69 per class); `keypoint.csv`: 312 rows; label CSVs written with a UTF-8 BOM; no signer ids, no recording FPS stored |
| app loop | single thread: `waitKey(10)` → `cap.read()` → flip → Hands → vector → 30-deque (cleared after 5 hand-less frames) → static + sequence → `choose_gesture` (sequence wins) → `GestureStabilizer(8)` → `SpeechWorker` (thread + `Queue(16)`, one pyttsx3 **subprocess per word**, a deliberate workaround for the runAndWait-twice hang) |
| camera | `camera_utils.open_camera` (DirectShow → MSMF → default, warm-up, black-frame rejection) |
| training | `train_sequence_classifier.py`: Keras 3, single 25 % split doubling as validation, EarlyStopping(20), fixed-batch TFLite export with a class-count self-check (refuses to save on mismatch) |
| framework | Keras 3.15.1 confirmed from the shipped `.keras` checkpoints; TF ≥ 2.16 |
| pre-existing issues | `tests/test_app_integration.py` stale (referenced removed `app.TTSSpeaker`), root/`tests/` basename collision, `tts_speaker.py` dead duplicate of `gesture_output.py`, scikit-learn used but missing from `requirements.txt` |

Wrong assumptions in the standalone `sh_research` that had to go: 1662-d
MediaPipe Holistic features, `MP_Data`/npz datasets, its own Holistic
extractor and camera class, Keras/`tf.function` inference at run time (the
repo runs TFLite), its own stabilizer, a second TTS worker, a `Lambda` pooling
layer that breaks Keras-3 safe loading and TFLite conversion.

## B. Decisions (KEEP / MERGE / ADAPT / REMOVE)

| `sh_research` part | decision | result in the repo |
|---|---|---|
| config dataclasses | ADAPT | `sh_research/config.py`: 84-d layout, repo CSV paths, stabilizer/realtime/TTS defaults **equal the original app.py values**; `app.py --config/--set` on top of the classic flags |
| dataset loaders (npz, mp_dirs) | REMOVE → ADAPT | one loader: new `dataset_utils.py` (shared by both training scripts, `extract_gesture_data.py` and `sh_research.data`) |
| preprocessing (`Preprocessor`) | KEEP | serialised into `metadata.json`; `Preprocessor.identity()` = the original pipeline; applied by `app.py` when a run directory is given |
| splits | KEEP | stratified (default) / signer-independent (optional `signers_path`) |
| models MLP/GRU/LSTM/GRU+LSTM | KEEP (fix) | `Lambda` "last" pooling → `return_sequences=False`; `build_model(..., batch_size=1)` for export; same `[1,T,84] → softmax` contract as the original model |
| trainer | ADAPT | + `export_tflite` (rebuild fixed batch, copy weights, convert, **refuse** on class/shape/numeric mismatch), `labels.csv` in the repo's label format |
| latency measurement | ADAPT | + TFLite interpreter timing (what `app.py` runs) next to `tf.function`/`predict` |
| experiments (runner, aggregate, compare, ablation, metadata) | KEEP | + dataset fingerprint, TFLite info/latency; Pareto on TFLite latency |
| Holistic extractor, own camera | REMOVE | `HandsExtractor` wraps the app's `mp.solutions.hands` object + `build_combined_vector`; `CaptureThread` wraps the capture returned by `camera_utils.open_camera` |
| scheduler, frame buffer, metrics, simulator, overlay | KEEP | + stages `static_model_ms`, `sequence_model_ms`, `draw_ms`, `loop_ms`; `FrameSource` adds the synchronous mode (`target_fps 0` = original loop) |
| `pipeline.py` (Keras inference loop) | REMOVE → new | `GesturePipeline`: the app's per-frame logic (window, static + sequence TFLite, `choose_gesture`, stabiliser, speech event), used by `app.py` and the headless benchmark |
| stabilizer | MERGE | `gesture_output.PredictionStabilizer` **composes** the existing `GestureStabilizer`; defaults (stable_count 8, threshold 0.5, no smoothing) reproduce the original behaviour exactly; one stability mechanism in the project |
| TTS worker | MERGE | `gesture_output.SpeechWorker` = thin API shim over `sh_research.tts.TTSWorker` (same `say/pending/spoken/close`); classic constructor keeps the 16-deep queue |
| TTS backends | ADAPT | the original per-word pyttsx3 subprocess **is the default backend**, now emitting START/END markers (measured T6/T7) and optionally pre-spawning the next process; in-process backend keeps the fresh-engine-per-word rule; print/mock for tests |
| speech-event manager, text normalisation | KEEP (adapt) | labels spoken as-is by default (`split_label_words: false`, `capitalization: none`); parked events for `minimum_stable_duration_ms` now that the stabiliser emits once per held sign |
| `scripts/run_realtime.py` | REMOVE | `app.py` is the realtime entry point |
| docs of the standalone package | REMOVE | `README.md` rewritten for the integrated project; this report |

Repo files removed: `tts_speaker.py` + `tests/test_tts_speaker.py` (dead
duplicate of `gesture_output.py`, unused by `app.py`), `app.py.diff` (stale
patch artefact), stale `tests/test_app_integration.py` (replaced by the root
test, moved into `tests/`). All removals are visible in git history.

## C. Behaviour preserved / changed

Preserved: every original CLI flag and its meaning, console messages of
`load_sequence_classifier`, drawing, data-logging mode (`k` + digit),
`choose_gesture` arbitration, `GestureStabilizer` semantics, one pyttsx3
process per word, actionable start-up failure when pyttsx3 is missing,
`--target_fps 0` = the exact original synchronous loop (used by the tests).

Changed by default (each reversible by config):

* processing runs at 20 FPS through a capture thread + latest-frame buffer
  (`realtime.target_fps`, `--target_fps 0` to revert);
* the TTS queue holds 2 pending words and expires words older than 1.5 s
  (`tts.queue_size`, `tts.max_pending_age_ms`; the classic
  `SpeechWorker()` constructor still uses 16);
* a standby pyttsx3 process is pre-spawned (`tts.prespawn`);
* MediaPipe `model_complexity` is passed explicitly (value 1 = MediaPipe's
  default, so no change; 0 is available as a speed knob);
* the FPS readout shows the measured processing rate (and camera rate)
  instead of the old `CvFpsCalc` loop rate; a profiling overlay is drawn
  (`p` toggles, `--no_overlay` hides).

## D. Measurements (all measured; nothing extrapolated)

Sandbox: Linux x86-64, **1 CPU core**, no GPU. Two environments were used:
TF 2.21 / Keras 3.15.1 / numpy 2.4 (research code + tests) and a venv with
**TF 2.19 / Keras 3.15.1 / MediaPipe 0.10.21 / numpy 1.26 / protobuf 4.25**
(tests + MediaPipe benchmark). Raw JSON/CSV: `docs/measurements/`.

**Test suite** — `python -m pytest tests/`: **102 passed, 1 skipped** in both
environments (skipped: the real pyttsx3-subprocess audio test; no speech
driver in the sandbox). Baseline before integration: 20 pass / 1 skip (root
tests) + 15 pass / 2 fail (stale `tests/`).

**Model latency at `[1, 30, 84]`, batch 1, 200 samples** (`scripts/benchmark_latency.py`, TF 2.21):

| arch | params | TFLite p50 / p95 / p99 (ms) | tf.function p50 (ms) | `model.predict` p50 (ms) |
|---|---|---|---|---|
| mlp | 333 090 | 0.016 / 0.031 / 0.069 | 0.56 | 57.9 |
| gru | 55 906 | 0.264 / 0.315 / 0.370 | 2.56 | 58.1 |
| lstm | 73 314 | 0.285 / 0.350 / 0.370 | 1.99 | 58.3 |
| gru_lstm | 121 954 (2+2 layers) | 0.671 / 0.713 / 0.786 | 3.81 | 58.0 |

**Four architectures on the real dataset** (138 windows, `Hi`/`help`, split
96/21/21, 60 epochs, seeds 1 and 42, `docs/measurements/real_dataset_leaderboard.md`):
all four reach test accuracy 1.000 ± 0.000 and macro-F1 1.000; TFLite p50
0.020 (mlp) / 0.277 (gru) / 0.301 (lstm) / 0.278 ms (gru_lstm, 1+1 layers,
63 970 params); train time 9.6–12.9 s. **The dataset cannot rank the
architectures** — two well-separated classes and 21 test samples.

**Headless realtime, real deployed TFLite models, synthetic landmarks**
(`scripts/benchmark_realtime.py`, 15 s, camera 30 FPS → target 20 FPS, mock TTS):
achieved **20.02 FPS, sustained**; captured 451, processed 301, dropped 150
(the expected 30→20 discard); processing interval p50 50.05 ms, p95 50.43 ms;
scheduler lateness p50 0.24 ms; frame age p50 0.16 ms, p95 16.6 ms;
static model p50 0.10 ms; sequence model p95 0.62 ms; total processing p50
0.24 ms / p95 0.92 ms; capture→prediction p50 1.7 ms / p95 17.4 ms;
confirmed→audio p50 50.3 ms with a 50 ms mock synthesis (queue wait p50 0.11 ms).

**Headless realtime, real MediaPipe Hands** (TF 2.19 / MediaPipe 0.10.21,
960×540 generated noise frames — **no hands present**, so this is a lower
bound: with hands the landmark model runs after the palm detector): 20 s,
achieved **20.02 FPS, sustained**; `landmark_ms` p50 **12.65 ms**, p95 16.0 ms,
p99 19.1 ms, max 132 ms (first frames); image preprocessing p50 0.81 ms;
total p50 13.5 ms / p95 17.0 ms; capture→prediction p50 17.8 ms / p95 34.8 ms;
bottleneck: landmarks, 94 % of the loop. On this 1-core machine 20 FPS held
with ~37 ms of headroom per tick; real hands and slower CPUs will reduce it —
`app.py --benchmark` reports the actual numbers on the target machine.

**Scheduler simulation** (`scripts/simulate_scheduler.py`): 20.0 FPS held at
30/60/24 FPS cameras and under ±50 % jitter and bursts (interval p50/p95 =
50/50 ms, backlog max 1); camera-limited at 15 FPS; an 80 ms processing cost
degrades to 12.6 FPS with frame age p50 13 ms under the latest-frame policy
versus 147 ms with a 5-deep FIFO — the reason `max_queue_size` defaults to 1.

**TTS worker overhead** (`scripts/benchmark_tts.py --tts mock`, 30 events at
20 FPS): 30/30 spoken, 0 dropped/expired; queue wait p50 0.1 ms; confirmed →
audio p50 50.5 ms, p95 50.6 ms, max 50.7 ms with a 50 ms mock synthesis, i.e.
≈ 0.5 ms of pipeline overhead.

**Not measurable here:** real pyttsx3 audio (no SAPI5/eSpeak in the sandbox):
prewarm, process-spawn, synthesis and playback-start times must be measured
on the target machine with `scripts/benchmark_tts.py` (and
`--compare-prespawn`); live camera + real hands with `app.py --benchmark`.

## E. Remaining issues and recommendations

1. **Dataset size**: 138 windows / 2 classes / 1 signer. Architecture
   comparison, thresholds and stability settings need more signs and signers
   (record a `rest` class; use `data.signers_path` for signer-independent splits).
2. **Temporal mismatch**: training windows are 30 frames at the recording
   rate (30 FPS assumed = 1.0 s) while the app processes 20 FPS (1.5 s). The
   app prints a MISMATCH warning; re-extract with `--sample_fps 20` (new flag)
   or run `--target_fps 30` on machines that sustain it.
3. **Environment**: the legacy MediaPipe `solutions` API (≤ 0.10.21,
   `protobuf<5`) cannot coexist with TensorFlow ≥ 2.20 (`protobuf ≥ 5.28`);
   `requirements.txt` now pins the known-good set (TF < 2.20, MediaPipe ≤ 0.10.21,
   numpy < 2). Migrating to the MediaPipe Tasks `HandLandmarker` would lift this.
4. `tf.lite.Interpreter` is deprecated in favour of LiteRT (`ai_edge_litert`);
   it still works on TF 2.19–2.21.
5. MediaPipe latency with real hands and the TTS audio timings are still to be
   measured on the target hardware (commands in README §5 and §7).
6. `tts_speaker.py` was removed as a dead duplicate; restore from git history
   if any external script imported it.
