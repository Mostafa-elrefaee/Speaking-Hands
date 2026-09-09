"""Speech-event manager: turns *validated predictions* into *speech events*.

    model -> prediction -> [gesture_output.PredictionStabilizer]      -> StableEvent (T2)
          -> [SpeechEventManager]                                     -> SpeechEvent (T3)
          -> bounded queue -> TTSWorker (T4..T6)

The manager owns exactly one duplicate-suppression mechanism (configurable):

* ``duplicate_cooldown_ms``     — the same label is not spoken again inside the window
                                  (the default and usually sufficient control);
* ``require_prediction_change`` — the same label is never re-spoken until a different
                                  label has been spoken (stricter, for held signs);
* ``minimum_stable_duration_ms``— the label must have been winning for this long before
                                  it is spoken (adds latency; off by default).

and the speech mode: ``immediate_word`` (one event per validated sign) or
``buffered_phrase`` (words are accumulated and flushed as one event after
``phrase_max_words`` or ``phrase_timeout_ms`` of silence; ``tick()`` drives the
timeout). Text normalisation happens here so the queue only ever carries
speakable strings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from ..config import TTSConfig
from .text import join_phrase, label_to_text, word_for_label

if TYPE_CHECKING:  # gesture_output imports this package; avoid the cycle
    from gesture_output import StableEvent


@dataclass
class SpeechEvent:
    text: str
    labels: List[str]
    captured_ts: Optional[float]  # T0 frame capture (of the last confirming frame)
    predicted_ts: Optional[float]  # T1 model output
    validated_ts: float  # T2 stabilizer confirmation
    created_ts: float  # T3 speech event creation
    confidence: float


class SpeechEventManager:
    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._last_label: Optional[str] = None
        self._last_label_ts: float = -1e9
        self._last_spoken_label: Optional[str] = None
        self._phrase: List["StableEvent"] = []
        self._phrase_last_ts: float = -1e9
        self._pending: Optional["StableEvent"] = None  # waiting for minimum_stable_duration_ms
        self.suppressed_duplicates = 0
        self.suppressed_unstable = 0
        self.suppressed_empty = 0
        self.events_created = 0

    # --- policy ------------------------------------------------------------------
    def _held_long_enough(self, ev: "StableEvent", now: float) -> bool:
        if self.cfg.minimum_stable_duration_ms <= 0 or ev.stable_since is None:
            return True
        return (now - ev.stable_since) * 1e3 >= self.cfg.minimum_stable_duration_ms

    def _allowed(self, ev: "StableEvent", now: float) -> bool:
        if ev.label == self._last_label:
            if self.cfg.require_prediction_change and self._last_spoken_label == ev.label:
                self.suppressed_duplicates += 1
                return False
            if (now - self._last_label_ts) * 1e3 < self.cfg.duplicate_cooldown_ms:
                self.suppressed_duplicates += 1
                return False
        return True

    def _make(self, evs: List["StableEvent"], text: Optional[str], now: float) -> Optional[SpeechEvent]:
        if not text:
            self.suppressed_empty += 1
            return None
        last = evs[-1]
        self.events_created += 1
        self._last_spoken_label = last.label
        return SpeechEvent(text, [e.label for e in evs], last.frame_captured_at, last.predicted_at,
                           last.confirmed_at, now, last.confidence)

    # --- API ---------------------------------------------------------------------------
    def on_validated(self, ev: "StableEvent", now: Optional[float] = None) -> Optional[SpeechEvent]:
        """Feed a stabilized prediction (T2). Returns a SpeechEvent (T3) or None.

        The stabilizer emits ONE event per held sign (when it becomes
        current). With minimum_stable_duration_ms the event is parked and
        released by tick() once the sign has been held long enough -- or
        dropped if the sign changes first."""
        now = time.monotonic() if now is None else now
        self._pending = None
        if not self._held_long_enough(ev, now):
            self._pending = ev
            return None
        return self._accept(ev, now)

    def _accept(self, ev: "StableEvent", now: float) -> Optional[SpeechEvent]:
        if not self._allowed(ev, now):
            return None
        self._last_label, self._last_label_ts = ev.label, now
        if self.cfg.mode == "buffered_phrase":
            self._phrase_last_ts = now
            if self._phrase and self._phrase[-1].label == ev.label:
                return None  # a held sign is one word, not one word per validated frame
            self._phrase.append(ev)
            if len(self._phrase) >= self.cfg.phrase_max_words:
                return self.flush(now)
            return None
        return self._make([ev], label_to_text(ev.label, self.cfg), now)

    def tick(self, now: Optional[float] = None, current: Optional[str] = None) -> Optional[SpeechEvent]:
        """Call once per processed frame with the stabilizer's current gesture.
        Releases a parked event (minimum_stable_duration_ms) while the same
        sign is still held, and in buffered_phrase mode flushes the phrase
        after phrase_timeout_ms of silence (measured from the release of the
        last sign, not from its onset)."""
        now = time.monotonic() if now is None else now
        if self._pending is not None:
            ev = self._pending
            if current is not None and current != ev.label:
                self._pending = None
                self.suppressed_unstable += 1
            elif self._held_long_enough(ev, now):
                self._pending = None
                return self._accept(ev, now)
        if self._phrase:
            if current is not None and current == self._phrase[-1].label:
                self._phrase_last_ts = now  # sign still held: silence has not started
            elif (now - self._phrase_last_ts) * 1e3 >= self.cfg.phrase_timeout_ms:
                return self.flush(now)
        return None

    def flush(self, now: Optional[float] = None) -> Optional[SpeechEvent]:
        now = time.monotonic() if now is None else now
        if not self._phrase:
            return None
        evs, self._phrase = self._phrase, []
        words = [word_for_label(e.label, self.cfg) for e in evs]
        return self._make(evs, join_phrase([w for w in words if w], self.cfg), now)

    def stats(self) -> dict:
        return {"mode": self.cfg.mode, "events_created": self.events_created,
                "suppressed_duplicates": self.suppressed_duplicates,
                "suppressed_unstable": self.suppressed_unstable, "suppressed_empty": self.suppressed_empty,
                "duplicate_cooldown_ms": self.cfg.duplicate_cooldown_ms,
                "require_prediction_change": self.cfg.require_prediction_change,
                "minimum_stable_duration_ms": self.cfg.minimum_stable_duration_ms}
