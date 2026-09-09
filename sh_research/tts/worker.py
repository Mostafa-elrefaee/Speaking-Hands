"""Asynchronous TTS worker.

    SpeechEvent (T3) -> enqueue() [non-blocking, µs]
                     -> bounded queue (queue_size, overflow_policy)
                     -> worker thread: T4 dequeue -> expiry check (max_pending_age_ms)
                                        -> backend.speak(): T5 synthesis done, T6 playback start, T7 end
                     -> SpeechLatencyRecord with T0..T7 and all deltas

* The inference loop never waits: ``enqueue`` returns immediately even while
  audio is generating or playing.
* Stale policy: overflow drops the oldest (or newest) pending event; at dequeue
  an event older than ``max_pending_age_ms`` is expired instead of spoken, so
  the system never voices a prediction seconds after the user stopped signing.
* ``interrupt_current_audio`` calls ``backend.stop()`` on arrival of a new
  event only when the backend declares ``supports_interrupt``.
* ``prewarm``: the backend is initialised once at ``start()``; the time is
  recorded so prewarm-vs-cold latency can be compared (scripts/benchmark_tts.py).
  A prewarm failure is exposed as ``init_error`` so app.py can fail loudly at
  start-up with an actionable message (the original SpeechWorker behaviour);
  runtime failures are recorded and never raised into the inference loop.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from ..config import TTSConfig
from ..evaluation.latency import percentiles
from .backends import TTSBackend, make_backend
from .events import SpeechEvent


@dataclass
class SpeechLatencyRecord:
    text: str
    t0_captured: Optional[float]
    t1_predicted: Optional[float]
    t2_validated: float
    t3_event_created: float
    t3b_enqueued: float
    t4_worker_start: float
    t5_synthesis_done: Optional[float]
    t6_playback_start: Optional[float]
    t7_playback_end: Optional[float]

    @staticmethod
    def _ms(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if a is None or b is None else (b - a) * 1e3

    @property
    def total_prediction_to_speech_ms(self) -> Optional[float]:
        return self._ms(self.t2_validated, self.t6_playback_start)

    @property
    def tts_generation_ms(self) -> Optional[float]:
        return self._ms(self.t4_worker_start, self.t5_synthesis_done)

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "t0_captured": self.t0_captured, "t1_predicted": self.t1_predicted, "t2_validated": self.t2_validated,
            "t3_event_created": self.t3_event_created, "t4_worker_start": self.t4_worker_start,
            "t5_synthesis_done": self.t5_synthesis_done, "t6_playback_start": self.t6_playback_start,
            "t7_playback_end": self.t7_playback_end,
            "t2_t1_validation_ms": self._ms(self.t1_predicted, self.t2_validated),
            "t3_t2_event_ms": self._ms(self.t2_validated, self.t3_event_created),
            "t4_t3_queue_wait_ms": self._ms(self.t3_event_created, self.t4_worker_start),
            "t5_t4_synthesis_ms": self._ms(self.t4_worker_start, self.t5_synthesis_done),
            "t6_t5_audio_startup_ms": self._ms(self.t5_synthesis_done, self.t6_playback_start),
            "t6_t2_confirmed_to_audio_ms": self._ms(self.t2_validated, self.t6_playback_start),
            "t1_t0_capture_to_model_ms": self._ms(self.t0_captured, self.t1_predicted),
            "t2_t0_capture_to_stable_ms": self._ms(self.t0_captured, self.t2_validated),
            "t4_t0_capture_to_tts_start_ms": self._ms(self.t0_captured, self.t4_worker_start),
            "t6_t0_capture_to_audio_ms": self._ms(self.t0_captured, self.t6_playback_start),
            "playback_ms": self._ms(self.t6_playback_start, self.t7_playback_end),
        }


DELTA_KEYS = ("t2_t1_validation_ms", "t3_t2_event_ms", "t4_t3_queue_wait_ms", "t5_t4_synthesis_ms",
              "t6_t5_audio_startup_ms", "t6_t2_confirmed_to_audio_ms", "t1_t0_capture_to_model_ms",
              "t2_t0_capture_to_stable_ms", "t4_t0_capture_to_tts_start_ms", "t6_t0_capture_to_audio_ms", "playback_ms")


class TTSWorker:
    def __init__(self, cfg: TTSConfig, backend: Optional[TTSBackend] = None, history: int = 1000) -> None:
        self.cfg = cfg
        self.backend = backend or make_backend(cfg.backend, rate=cfg.rate, volume=cfg.volume, voice=cfg.voice,
                                               prespawn=cfg.prespawn, mock_latency_s=cfg.mock_latency_s,
                                               mock_init_s=cfg.mock_init_s)
        self._q: "queue.Queue[Optional[SpeechEvent]]" = queue.Queue(maxsize=max(1, cfg.queue_size))
        self._thread = threading.Thread(target=self._run, name="tts-worker", daemon=True)
        self._stop_evt = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self.init_error: Optional[BaseException] = None
        self.records: Deque[SpeechLatencyRecord] = deque(maxlen=history)
        self.spoken_texts: List[str] = []  # everything successfully spoken, in order
        self.dropped_overflow = 0
        self.expired = 0
        self.suppressed_duplicates = 0
        self.interrupts = 0
        self.requested = 0
        self.spoken_count = 0
        self.failed = 0
        self.errors: List[str] = []
        self.prewarm_ms: Optional[float] = None
        self._last_text: Optional[str] = None
        self._last_text_ts: float = -1e9

    # --- lifecycle -----------------------------------------------------------
    def start(self, wait_ready: bool = True, init_timeout: float = 15.0) -> "TTSWorker":
        """Start the thread. With wait_ready the one-time backend prewarm is
        awaited and an init failure is re-raised here with its message, so a
        missing / broken TTS backend fails loudly at start-up."""
        self._thread.start()
        if wait_ready:
            if not self._ready.wait(init_timeout):
                raise RuntimeError(f"Speech engine did not initialise within {init_timeout}s")
            if self.init_error is not None:
                raise self.init_error
        return self

    def stop(self, wait: bool = True, timeout: float = 3.0) -> None:
        """Never raises; never blocks longer than ``timeout``."""
        self._stop_evt.set()
        try:
            self.backend.stop()  # kill an in-flight utterance if the backend can
        except Exception:
            pass
        try:
            self._q.put_nowait(None)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass
        if wait and self._thread.is_alive():
            self._thread.join(timeout)
        try:
            self.backend.close()
        except Exception:
            pass

    close = stop  # original SpeechWorker name

    @property
    def is_speaking(self) -> bool:
        return bool(getattr(self.backend, "is_busy", False))

    @property
    def pending(self) -> int:
        return self._q.qsize()

    @property
    def available(self) -> bool:
        return self._thread.is_alive() and not self._stop_evt.is_set() and self.init_error is None

    # --- enqueue (called from the inference loop; never blocks) ----------------
    def enqueue(self, event: SpeechEvent) -> bool:
        """Queue a speech event. Returns True if accepted, False if suppressed/dropped."""
        if not self.cfg.enabled or not event.text:
            return False
        now = time.monotonic()
        self.requested += 1
        if self.cfg.duplicate_window_s > 0:
            with self._lock:
                if event.text == self._last_text and now - self._last_text_ts < self.cfg.duplicate_window_s:
                    self.suppressed_duplicates += 1
                    return False
                self._last_text, self._last_text_ts = event.text, now
        event._enqueued_ts = now  # type: ignore[attr-defined]
        if (self.cfg.interrupt_current_audio and getattr(self.backend, "supports_interrupt", False)
                and getattr(self.backend, "is_busy", False)):
            self.backend.stop()
            self.interrupts += 1
        try:
            self._q.put_nowait(event)
            return True
        except queue.Full:
            if self.cfg.overflow_policy == "drop_newest":
                self.dropped_overflow += 1
                return False
            dropped = None
            try:  # drop_oldest: evict one pending item and retry once
                dropped = self._q.get_nowait()
                self.dropped_overflow += 1
            except queue.Empty:
                pass
            if dropped is not None:
                print(f"[tts] speech backlog full, dropped oldest unspoken word: {dropped.text!r}")
            try:
                self._q.put_nowait(event)
                return True
            except queue.Full:
                self.dropped_overflow += 1
                return False

    def speak(self, text: str, confirmed_ts: Optional[float] = None, captured_ts: Optional[float] = None,
              predicted_ts: Optional[float] = None) -> bool:
        """Convenience: build a SpeechEvent from plain text and enqueue it."""
        now = time.monotonic()
        t2 = confirmed_ts if confirmed_ts is not None else now
        return self.enqueue(SpeechEvent(text, [text], captured_ts, predicted_ts, t2, now, 1.0))

    # --- background loop ------------------------------------------------------
    def _run(self) -> None:
        try:
            if self.cfg.prewarm and hasattr(self.backend, "prewarm"):
                self.prewarm_ms = float(self.backend.prewarm())
        except Exception as e:
            self.init_error = e
            self.errors.append(f"prewarm {type(e).__name__}: {e}")
            self._ready.set()
            return
        self._ready.set()
        while not self._stop_evt.is_set():
            try:
                ev = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if ev is None:
                break
            t4 = time.monotonic()
            if self.cfg.max_pending_age_ms > 0 and (t4 - ev.created_ts) * 1e3 > self.cfg.max_pending_age_ms:
                self.expired += 1  # stale: never speak an old prediction
                continue
            try:
                t5, t6, t7 = self.backend.speak(ev.text)
                self.spoken_count += 1
                self.spoken_texts.append(ev.text)
                self.records.append(SpeechLatencyRecord(ev.text, ev.captured_ts, ev.predicted_ts, ev.validated_ts,
                                                        ev.created_ts, getattr(ev, "_enqueued_ts", ev.created_ts),
                                                        t4, t5, t6, t7))
            except Exception as e:  # keep the worker alive; inference is unaffected
                self.failed += 1
                self.errors.append(f"{type(e).__name__}: {e}")
                if not self._stop_evt.is_set():
                    print(f"[tts] failed to speak {ev.text!r}: {e}")

    # --- reporting ----------------------------------------------------------------
    def stats(self) -> Dict[str, object]:
        rows = [r.to_dict() for r in self.records]
        out: Dict[str, object] = {
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
            "prewarm": self.cfg.prewarm, "prewarm_ms": self.prewarm_ms,
            "queue_size": self.cfg.queue_size, "overflow_policy": self.cfg.overflow_policy,
            "max_pending_age_ms": self.cfg.max_pending_age_ms,
            "interrupt_current_audio": self.cfg.interrupt_current_audio,
            "requested": self.requested, "spoken": self.spoken_count, "failed": self.failed,
            "suppressed_duplicates": self.suppressed_duplicates, "dropped_overflow": self.dropped_overflow,
            "expired": self.expired, "interrupts": self.interrupts, "errors": self.errors[-5:],
        }
        spawn = getattr(self.backend, "spawn_ms", None)
        if spawn:
            out["process_spawn_ms"] = percentiles(spawn)
            out["no_start_marker"] = getattr(self.backend, "no_start_marker", 0)
        for k in DELTA_KEYS:
            out[k] = percentiles([r[k] for r in rows if r[k] is not None])
        out["prediction_to_tts_start_ms"] = percentiles([(r["t4_worker_start"] - r["t2_validated"]) * 1e3 for r in rows])
        out["tts_generation_ms"] = out["t5_t4_synthesis_ms"]
        out["audio_startup_ms"] = out["t6_t5_audio_startup_ms"]
        out["total_prediction_to_speech_ms"] = out["t6_t2_confirmed_to_audio_ms"]
        out["capture_to_speech_ms"] = out["t6_t0_capture_to_audio_ms"]
        out["queue_wait_ms"] = out["t4_t3_queue_wait_ms"]
        return out

    def records_as_rows(self) -> List[Dict[str, object]]:
        return [r.to_dict() for r in self.records]
