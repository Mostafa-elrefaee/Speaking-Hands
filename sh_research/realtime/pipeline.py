"""GesturePipeline: one processed frame, end to end, with per-stage timing.

This is the ML part of app.py's loop, factored out so the live app and the
headless benchmark run the SAME code:

    frame ──▶ [extractor] flip + BGR->RGB              (image_preprocess_ms)
              MediaPipe Hands                          (landmark_ms)
              build_combined_vector -> 84 values       (feature_ms)
          ──▶ sequence buffer (deque, hand-present frames only; reset after
              max_missing_frames no-hand frames OR a processing stall)  (window_ms)
          ──▶ KeyPointClassifier  (static, argmax)     (static_model_ms)
              SequenceClassifier  (probabilities)      (sequence_model_ms)
          ──▶ PredictionStabilizer.smooth()            (probability_ms)
              choose_gesture() + .commit()             (stabilize_ms)
          ──▶ SpeechEventManager -> TTSWorker.enqueue  (tts_event_ms, non-blocking)

Models know nothing about time: any TFLite model with input [1, T, 84] and a
softmax output plugs in through model/sequence_classifier/sequence_classifier.py,
whichever of the four architectures produced it.

Sequence window vs target rate: one processed sample per tick, so a full
window of ``seq_length`` frames spans ``seq_length / target_fps`` seconds
(30 / 20 = 1.5 s). Training windows were extracted at the recording video's
rate (data.source_fps); if that differs from target_fps the window covers a
different amount of real time than in training -- ``temporal`` in the report
says so explicitly (see README "Sequence length vs FPS").
"""
from __future__ import annotations

import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from gesture_output import PredictionStabilizer, StableEvent, choose_gesture

from ..config import ExperimentConfig
from ..data.dataset import Preprocessor
from ..tts.events import SpeechEvent, SpeechEventManager
from .metrics import PerfTracker


@dataclass
class FrameResult:
    image: np.ndarray                      # flipped BGR frame for drawing
    combined_vector: List[float]
    per_hand_info: list                    # [(side, landmark_list, brect)] per detected hand
    raw_gesture: Optional[str]
    source: Optional[str]                  # "seq" | "static" | None
    current: Optional[str]                 # stabilized gesture (single source of truth)
    changed: bool
    event: Optional[StableEvent] = None
    speech: Optional[SpeechEvent] = None
    seq_probs: Optional[np.ndarray] = None
    seq_confidence: Optional[float] = None
    static_label: Optional[str] = None
    stages: Dict[str, float] = field(default_factory=dict)


class GesturePipeline:
    def __init__(self, cfg: ExperimentConfig, extractor, keypoint_classifier, keypoint_labels: List[str],
                 sequence_classifier=None, sequence_labels: Optional[List[str]] = None,
                 stabilizer: Optional[PredictionStabilizer] = None, speech_events: Optional[SpeechEventManager] = None,
                 speaker=None, preprocessor: Optional[Preprocessor] = None, perf: Optional[PerfTracker] = None) -> None:
        self.cfg = cfg
        self.extractor = extractor
        self.keypoint_classifier = keypoint_classifier
        self.keypoint_labels = list(keypoint_labels)
        self.sequence_classifier = sequence_classifier
        self.sequence_labels = list(sequence_labels or [])
        self.stabilizer = stabilizer or PredictionStabilizer(cfg.stabilizer, self.sequence_labels or self.keypoint_labels)
        self.speech_events = speech_events or SpeechEventManager(cfg.tts)
        self.speaker = speaker
        self.perf = perf or PerfTracker(cfg.realtime.latency_window, cfg.realtime.target_fps, cfg.realtime.profiling_enabled)
        seq_len = int(sequence_classifier.seq_length) if sequence_classifier is not None else 1
        self.pre = preprocessor or Preprocessor.identity(seq_len)
        self.seq_buffer: Deque = deque(maxlen=seq_len)
        self.window_ts: Deque[float] = deque(maxlen=seq_len)
        self.missing_frames = 0
        self.events: List[StableEvent] = []
        self.last_event: Optional[StableEvent] = None
        self.temporal = self._temporal_info()

    # --- info -----------------------------------------------------------------------
    def _temporal_info(self) -> Dict[str, object]:
        rt = self.cfg.realtime
        seq_len = self.seq_buffer.maxlen
        span_rt = seq_len / rt.target_fps if rt.target_fps > 0 else None
        train_fps = self.pre.sampling_fps or self.pre.source_fps or self.cfg.data.sampling_fps
        span_train = seq_len / train_fps if train_fps else None
        mismatch = bool(span_rt and train_fps and abs(train_fps - rt.target_fps) > 1e-6)
        info = {"sequence_length": seq_len, "target_fps": rt.target_fps, "window_span_realtime_s": span_rt,
                "training_sampling_fps": train_fps, "window_span_training_s": span_train, "mismatch": mismatch}
        if mismatch and self.sequence_classifier is not None:
            warnings.warn(f"target_fps={rt.target_fps} differs from the training frame rate {train_fps}: a "
                          f"{seq_len}-frame window spans {span_rt:.2f} s at runtime vs {span_train:.2f} s in "
                          f"training. Extract with --sample_fps {rt.target_fps:g} or set realtime.target_fps to "
                          f"the recording rate to align them.")
        return info

    @property
    def current(self) -> Optional[str]:
        return self.stabilizer.current

    def close(self) -> None:
        close = getattr(self.extractor, "close", None)
        if close is not None:
            close()

    # --- one processed frame ------------------------------------------------------------
    def process(self, frame, captured_at: float, late_s: float = 0.0, acquire_ms: float = 0.0) -> FrameResult:
        rt, sc = self.cfg.realtime, self.cfg.stabilizer
        t0 = time.monotonic()
        st: Dict[str, float] = {"frame_age_ms": (t0 - captured_at) * 1e3, "schedule_late_ms": late_s * 1e3,
                                "acquire_ms": acquire_ms}

        # 4-6. flip/convert, MediaPipe Hands, 84-value combined vector
        image, (combined_vector, per_hand_info, timings) = self.extractor.extract(frame)
        st.update(timings)

        raw_gesture, source, static_label = None, None, None
        seq_probs, seq_conf, event, speech = None, None, None, None
        t_model_end = None

        if per_hand_info:  # at least one hand detected
            self.missing_frames = 0

            # 7. rolling window for the motion model (hand-present frames only,
            #    identical to how extract_gesture_data.py builds training
            #    windows; a missing SECOND hand is zero-filled in the vector).
            t = time.monotonic()
            interval = 1.0 / rt.target_fps if rt.target_fps > 0 else 0.0
            if (interval and self.window_ts
                    and (captured_at - self.window_ts[-1]) > rt.max_sequence_gap_intervals * interval):
                self.seq_buffer.clear()
                self.window_ts.clear()
                self.stabilizer.reset()
                self.perf.window_resets += 1
            feat = self.pre.select_features(combined_vector) if not self.pre.is_identity else combined_vector
            self.seq_buffer.append(feat)
            self.window_ts.append(captured_at)
            st["window_ms"] = (time.monotonic() - t) * 1e3

            # 8a. static prediction for the whole (one- or two-handed) gesture
            t = time.monotonic()
            hand_sign_id = self.keypoint_classifier(combined_vector)
            static_label = self.keypoint_labels[hand_sign_id]
            st["static_model_ms"] = (time.monotonic() - t) * 1e3

            # 8b. sequence prediction (probabilities; None until the buffer is full)
            t = time.monotonic()
            seq_label = None
            if self.sequence_classifier is not None:
                seq_probs = self.sequence_classifier.predict_proba(self._window())
            t_model_end = time.monotonic()
            st["sequence_model_ms"] = (t_model_end - t) * 1e3
            st["model_ms"] = st["static_model_ms"] + st["sequence_model_ms"]

            # 9. probability smoothing + confidence gate (was --seq_score_th)
            t = time.monotonic()
            if seq_probs is not None:
                seq_id, seq_conf = self.stabilizer.smooth(seq_probs)
                if seq_id is not None:
                    seq_label = self.sequence_labels[seq_id]
            st["probability_ms"] = (time.monotonic() - t) * 1e3

            raw_gesture, source = choose_gesture(seq_label, static_label)
        else:
            self.missing_frames += 1
            if self.missing_frames == sc.max_missing_frames:
                self.seq_buffer.clear()
                self.window_ts.clear()
                self.stabilizer.smoother.reset()

        # 10. stabilisation: the ONE stabilized gesture drives screen + speech.
        t = time.monotonic()
        conf = seq_conf if (source == "seq" and seq_conf is not None) else 1.0
        event = self.stabilizer.commit(raw_gesture, conf, t, frame_captured_at=captured_at,
                                       predicted_at=t_model_end, source=source)
        current = self.stabilizer.current
        st["stabilize_ms"] = (time.monotonic() - t) * 1e3

        # 11. speech event (duplicate policy + text normalisation) + non-blocking enqueue
        t = time.monotonic()
        if event is not None:
            self.events.append(event)
            self.last_event = event
            speech = self.speech_events.on_validated(event, t)
        if speech is None:  # parked min-stable-duration release / phrase timeout flush
            speech = self.speech_events.tick(t, current)
        if speech is not None and self.speaker is not None:
            self.speaker.enqueue(speech)
        st["tts_event_ms"] = (time.monotonic() - t) * 1e3

        st["total_ms"] = (time.monotonic() - t0) * 1e3
        st["end_to_end_ms"] = (time.monotonic() - captured_at) * 1e3
        self.perf.record_processing(st)
        return FrameResult(image, combined_vector, per_hand_info, raw_gesture, source, current,
                           event is not None, event, speech, seq_probs, seq_conf, static_label, st)

    def _window(self):
        if self.pre.mean is None:
            return self.seq_buffer
        if len(self.seq_buffer) < self.seq_buffer.maxlen:
            return self.seq_buffer
        return self.pre.transform_array(np.stack(self.seq_buffer))

    # --- reporting ------------------------------------------------------------------------
    def report(self) -> Dict[str, object]:
        rt = self.cfg.realtime
        rep: Dict[str, object] = {
            "target_fps": rt.target_fps,
            "buffer": {"max_queue_size": rt.max_queue_size, "latest_frame_strategy": rt.latest_frame_strategy,
                       "drop_stale_frames": rt.drop_stale_frames, "stale_frame_max_age_s": rt.stale_frame_max_age_s},
            "events_emitted": len(self.events),
            "stabilizer": {"events": self.stabilizer.events_emitted, "suppressed": self.stabilizer.suppressed},
            "temporal": self.temporal,
            "perf": self.perf.snapshot(),
            "speech_events": self.speech_events.stats(),
        }
        if self.speaker is not None and hasattr(self.speaker, "stats"):
            rep["tts"] = self.speaker.stats()
        return rep
