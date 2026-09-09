#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gesture_output.py

Post-processing that sits between the classifiers and the user in app.py:

    static prediction  ─┐
                        ├─> choose_gesture() ─> GestureStabilizer ─┬─> on-screen text
    sequence prediction ─┘        (priority)      (debounce)       └─> SpeechWorker (TTS)

The stabilizer's `current` value is the ONE source of truth for "what gesture
is being shown right now". Both the screen and the speech output read from
it; there is deliberately no second debounce for speech.

This module has no cv2 / mediapipe / tensorflow imports so it can be unit
tested with plain fake predictions (see tests/test_gesture_output.py).

Research extension (sh_research integration):
  * PredictionStabilizer wraps GestureStabilizer with optional probability
    smoothing / majority voting / confidence gate / cooldown, and turns each
    stabilization into a timestamped StableEvent for the speech-event manager
    and the latency benchmarks. With the default StabilizerConfig it behaves
    exactly like GestureStabilizer(stable_frames) + the old --seq_score_th gate.
  * SpeechWorker is now a thin compatibility wrapper over
    sh_research.tts.TTSWorker (bounded queue, stale-event expiry, measured
    T4..T7 timestamps). The pyttsx3-subprocess backend it used lives in
    sh_research/tts/backends.py unchanged in behaviour.
"""
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from sh_research.config import StabilizerConfig
from sh_research.tts.backends import (Pyttsx3Backend, Pyttsx3SubprocessBackend,  # noqa: F401 (re-exported)
                                      create_pyttsx3_engine)
from sh_research.tts.worker import TTSWorker


# ---------------------------------------------------------------------------
# Priority between the two classifiers
# ---------------------------------------------------------------------------

def choose_gesture(seq_label, static_label):
    """
    Pick this frame's raw gesture from the two classifiers' outputs.

    seq_label:    label string from the sequence classifier, or None if it
                  returned invalid_value (not enough history / low confidence)
                  or isn't loaded at all.
    static_label: label string from the static classifier, or None if no
                  hand was detected this frame.

    Returns (label, source) where source is "seq", "static" or None.

    The sequence classifier is authoritative when it speaks up, because:
      * its output is confidence-gated (score_th) while KeyPointClassifier
        always returns an argmax with no threshold, so a valid sequence
        result carries strictly more evidence than a static one;
      * it sees a superset of the information (the current frame plus the
        preceding window), so a motion sign that happens to pass through a
        known static hand-shape would otherwise be misread as that shape.
    The static classifier is the fallback for still signs and for the
    warm-up period before the buffer is full.
    """
    if seq_label is not None:
        return seq_label, "seq"
    if static_label is not None:
        return static_label, "static"
    return None, None


# ---------------------------------------------------------------------------
# Stability / debounce
# ---------------------------------------------------------------------------

class GestureStabilizer(object):
    """
    Promotes a raw per-frame prediction to the "current gesture" only after
    it has been identical for `stable_frames` consecutive frames.

    `None` is a valid raw value meaning "no gesture / no hand" and is
    debounced exactly like any label, so the on-screen text clears (and a
    repeated sign can be spoken again) only after the hands have genuinely
    been gone for a while, not on a single dropped frame.
    """

    def __init__(self, stable_frames):
        if stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        self.stable_frames = stable_frames
        self.current = None        # the stabilized value (single source of truth)
        self._candidate = None     # value seen on the most recent frame
        self._run_length = 0       # how many consecutive frames it has been seen

    def update(self, raw):
        """
        Feed this frame's raw prediction. Returns (current, changed) where
        `changed` is True only on the frame where `current` switches to a
        different value -- i.e. exactly one "stabilization event" per
        distinct gesture. Holding the same sign never re-triggers it.
        """
        if raw == self._candidate:
            self._run_length += 1
        else:
            self._candidate = raw
            self._run_length = 1

        changed = False
        if self._run_length >= self.stable_frames and raw != self.current:
            self.current = raw
            changed = True
        return self.current, changed


# ---------------------------------------------------------------------------
# Research stabilizer: probabilities -> gate -> GestureStabilizer -> StableEvent
# ---------------------------------------------------------------------------

@dataclass
class StableEvent:
    """One stabilization ("this gesture is now current") with the timestamps
    the latency measurements need."""
    label_index: int
    label: str
    confidence: float
    confirmed_at: float            # T2: monotonic time the decision was made
    frames_to_confirm: int         # consecutive frames the label was seen
    frame_captured_at: Optional[float] = None  # T0: capture time of the confirming frame
    predicted_at: Optional[float] = None       # T1: when the model produced the prediction
    stable_since: Optional[float] = None       # when this label's streak started
    source: Optional[str] = None               # "seq" / "static" (from choose_gesture)


class ProbabilitySmoother(object):
    """Optional pre-processing of a classifier's probability vectors:
    moving average over `averaging_window` frames and/or majority vote over
    the last `voting_window` argmaxes. Both off = pass-through."""

    def __init__(self, averaging_window=1, voting_window=0):
        self.averaging_window = max(1, int(averaging_window))
        self.voting_window = max(0, int(voting_window))
        self._probs: Deque[np.ndarray] = deque(maxlen=max(self.averaging_window, self.voting_window, 1))
        self._votes: Deque[int] = deque(maxlen=max(1, self.voting_window))

    def reset(self):
        self._probs.clear()
        self._votes.clear()

    def update(self, probs):
        """-> (label_index, confidence) after smoothing."""
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        self._probs.append(probs)
        if self.voting_window > 0:
            self._votes.append(int(probs.argmax()))
            label = Counter(self._votes).most_common(1)[0][0]
            recent = list(self._probs)[-self.voting_window:]
            conf = float(np.mean([p[label] for p in recent]))
            return label, conf
        recent = list(self._probs)[-self.averaging_window:]
        smoothed = np.mean(recent, axis=0)
        label = int(smoothed.argmax())
        return label, float(smoothed[label])


class PredictionStabilizer(object):
    """
    prediction -> [smoother] -> confidence gate -> GestureStabilizer -> StableEvent

    Two entry points share the same debounce so there is exactly ONE
    stability mechanism in the project:

      update(probs, ...)         one classifier's probability vector per frame
                                 (research benchmarks, single-model pipelines)
      commit(raw_label, ...)     an already-chosen label per frame -- app.py
                                 calls smooth() on the sequence probabilities,
                                 arbitrates with choose_gesture(), then commit()

    `current` is the stabilized gesture (or None), exactly as
    GestureStabilizer.current was, and drives both the screen and the speech.
    """

    def __init__(self, cfg: StabilizerConfig, class_names=None):
        self.cfg = cfg
        self.class_names = list(class_names) if class_names else []
        self.smoother = ProbabilitySmoother(cfg.averaging_window, cfg.voting_window)
        self.debounce = GestureStabilizer(cfg.stable_count)
        self._streak_start: float = 0.0
        self._last_emit_t: float = -1e9
        self._last_emit_label: Optional[str] = None
        self._last_emit_label_t: float = -1e9
        self.events_emitted = 0
        self.updates = 0
        self.suppressed = 0

    # -- state ----------------------------------------------------------------
    @property
    def current(self):
        return self.debounce.current

    def reset(self):
        """Forget the smoothing history and the running streak (used when the
        sequence window is cleared). `current` is kept, as before."""
        self.smoother.reset()
        self.debounce._candidate = None
        self.debounce._run_length = 0

    # -- step 1: probabilities -> gated label --------------------------------------
    def smooth(self, probs):
        """-> (label_index or None, confidence). None = below confidence_threshold."""
        label, conf = self.smoother.update(probs)
        if conf < self.cfg.confidence_threshold:
            return None, conf
        return label, conf

    # -- step 2: gated label -> stabilized event -----------------------------------
    def commit(self, raw_label, confidence=1.0, now=None, frame_captured_at=None,
               predicted_at=None, source=None):
        """Feed this frame's chosen raw label (a string, or None for "no
        gesture"). Returns a StableEvent on the frame `current` switches to a
        new non-None value, else None."""
        now = time.monotonic() if now is None else now
        self.updates += 1
        prev_candidate = self.debounce._candidate
        current, changed = self.debounce.update(raw_label)
        if raw_label != prev_candidate:
            self._streak_start = now
        if not changed or current is None:
            return None
        # optional extras (all off by default -> pure GestureStabilizer behaviour)
        if now - self._last_emit_t < self.cfg.cooldown_s:
            self.suppressed += 1
            return None
        if (self.cfg.duplicate_suppression and current == self._last_emit_label
                and now - self._last_emit_label_t < self.cfg.repeat_after_s):
            self.suppressed += 1
            return None
        self._last_emit_t = now
        self._last_emit_label, self._last_emit_label_t = current, now
        self.events_emitted += 1
        idx = self.class_names.index(current) if current in self.class_names else -1
        return StableEvent(idx, current, float(confidence), now, self.debounce._run_length,
                           frame_captured_at, predicted_at if predicted_at is not None else now,
                           self._streak_start, source)

    def update(self, probs, now=None, frame_captured_at=None, predicted_at=None):
        """Single-classifier path: probability vector -> StableEvent or None."""
        label, conf = self.smooth(probs)
        raw = self.class_names[label] if (label is not None and self.class_names) else \
            (str(label) if label is not None else None)
        return self.commit(raw, conf, now, frame_captured_at, predicted_at)

    @property
    def state(self):
        return {"current": self.current, "candidate": self.debounce._candidate,
                "streak": self.debounce._run_length}


# ---------------------------------------------------------------------------
# Text-to-speech on a background thread (compatibility wrapper)
# ---------------------------------------------------------------------------

class SpeechWorker(TTSWorker):
    """
    The original app.py speech API (say / pending / spoken / close) on top of
    sh_research.tts.TTSWorker.

    Backends:
      * default (engine_factory=None): one pyttsx3 subprocess per word --
        Pyttsx3SubprocessBackend, the original approach, now with measured
        audio-start timestamps and an optional pre-spawned standby process.
      * engine_factory given: in-process, a FRESH engine per word
        (Pyttsx3Backend). Used by the tests with a fake engine; also usable on
        Linux if you'd rather avoid process spawns.

    `maxsize` maps to the bounded queue (the old default was 16; the research
    default is TTSConfig.queue_size = 2 so speech can never lag behind signs).
    """

    def __init__(self, engine_factory=None, maxsize=None, cfg=None):
        from sh_research.config import TTSConfig
        if cfg is None:  # classic constructor: the original 16-deep queue
            cfg = TTSConfig(queue_size=16 if maxsize is None else int(maxsize))
        elif maxsize is not None:
            cfg.queue_size = int(maxsize)
        if engine_factory is not None:
            backend = Pyttsx3Backend(cfg.rate, cfg.volume, cfg.voice, engine_factory=engine_factory)
        elif cfg.backend == "pyttsx3_subprocess":
            backend = Pyttsx3SubprocessBackend(cfg.rate, cfg.volume, cfg.voice, prespawn=cfg.prespawn)
        else:
            backend = None  # make_backend(cfg.backend)
        super().__init__(cfg, backend)

    def start(self, init_timeout=15.0):
        """Start the thread and wait for the one-time engine probe, so a
        missing/broken TTS backend fails loudly here rather than silently
        inside the thread later."""
        return super().start(wait_ready=True, init_timeout=init_timeout)

    def say(self, text):
        if not text:
            return
        self.speak(text)

    @property
    def spoken(self):
        return self.spoken_texts
