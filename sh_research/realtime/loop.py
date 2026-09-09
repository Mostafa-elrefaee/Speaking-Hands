"""FrameSource: where the app's loop gets its next frame from.

Two modes, selected by ``realtime.target_fps``:

* ``target_fps > 0``  -- SCHEDULED (default, 20 FPS):
      capture thread -> FrameBuffer(max_queue_size) -> RateScheduler -> next()
  ``next()`` sleeps until the next 50 ms deadline (monotonic clock, fixed
  interval with re-anchoring so drift never accumulates), then takes the
  NEWEST buffered frame (older ones are counted as dropped) and discards it
  if it is older than ``stale_frame_max_age_s``. Camera rate and processing
  rate are therefore independent and both are measured.

* ``target_fps == 0`` -- SYNCHRONOUS (the original app.py behaviour):
  ``next()`` calls ``cap.read()`` directly, so every captured frame is
  processed in order. Used for video files and deterministic tests.

The same object serves app.py and the headless benchmark, so the timing
policy exists in exactly one place.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import RealtimeConfig
from .camera import CaptureThread, _BaseSource
from .metrics import PerfTracker
from .scheduler import FrameBuffer, RateScheduler, TimedFrame


@dataclass
class NextFrame:
    frame: np.ndarray
    captured_at: float
    seq: int
    late_s: float = 0.0
    acquire_ms: float = 0.0


class FrameSource:
    def __init__(self, cfg: RealtimeConfig, perf: PerfTracker, cap=None, thread: Optional[_BaseSource] = None) -> None:
        """Pass either an opened ``cap`` (cv2.VideoCapture-like: read()/release())
        or a ready-made capture ``thread`` (e.g. SyntheticCamera)."""
        self.cfg = cfg
        self.perf = perf
        self.cap = cap
        self.scheduled = cfg.target_fps > 0
        self.buffer: Optional[FrameBuffer] = None
        self.scheduler: Optional[RateScheduler] = None
        self.thread: Optional[_BaseSource] = thread
        self._sync_seq = 0
        self._ended = False
        if self.scheduled:
            self.buffer = FrameBuffer(cfg.max_queue_size)
            self.scheduler = RateScheduler(cfg.target_fps, cfg.max_catchup_intervals)
            if self.thread is None:
                if cap is None:
                    raise ValueError("FrameSource needs a cap or a capture thread")
                self.thread = CaptureThread(cap, self.buffer, perf)
            else:
                self.thread.slot = self.buffer
                self.thread.perf = perf
        elif thread is not None:
            raise ValueError("a capture thread requires target_fps > 0")

    def start(self) -> "FrameSource":
        if self.thread is not None and not self.thread.is_alive():
            self.thread.start()
        if self.scheduler is not None:
            self.scheduler.reset()
        return self

    @property
    def ended(self) -> bool:
        if self._ended:
            return True
        if self.thread is not None:
            return self.thread.ended and (self.buffer is None or self.buffer.size() == 0)
        return False

    def next(self) -> Optional[NextFrame]:
        """Next frame to process, or None (no fresh frame yet / stale / ended)."""
        if not self.scheduled:
            t0 = time.monotonic()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._ended = True
                return None
            t = time.monotonic()
            self._sync_seq += 1
            self.perf.record_capture(t, (t - t0) * 1e3)
            return NextFrame(frame, t, self._sync_seq, 0.0, 0.0)

        if self.thread is not None and self.thread.error is not None:
            raise self.thread.error
        late = self.scheduler.wait_until_due()  # stage 3: frame scheduling
        ta = time.monotonic()
        item: Optional[TimedFrame] = self.buffer.take(timeout=self.scheduler.interval,
                                                      newest=self.cfg.latest_frame_strategy)
        acquire_ms = (time.monotonic() - ta) * 1e3
        self.perf.queue_size = self.buffer.size()
        self.perf.dropped_frames = self.buffer.dropped
        if item is None:
            return None  # camera slower than target rate: nothing new to process
        if self.cfg.drop_stale_frames and (time.monotonic() - item.captured_at) > self.cfg.stale_frame_max_age_s:
            self.perf.skipped_stale += 1
            return None
        return NextFrame(item.frame, item.captured_at, item.seq, late, acquire_ms)

    def close(self) -> None:
        if self.thread is not None:
            self.thread.stop()
            self.thread.join(timeout=2.0)
        if self.buffer is not None:
            self.perf.dropped_frames = self.buffer.dropped
