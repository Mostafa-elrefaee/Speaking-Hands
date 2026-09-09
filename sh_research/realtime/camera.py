"""Frame sources that run in their own thread and push into a FrameBuffer.

``CaptureThread`` wraps an ALREADY-OPENED capture (from camera_utils.
open_camera(), which keeps the DirectShow/MSMF fallback, warm-up and
black-frame checks) -- it does not open cameras itself. The capture thread
runs at whatever rate the camera delivers and never waits for the processing
loop; because the buffer is bounded, a slow processor simply sees fewer,
fresher frames rather than a growing backlog.

``SyntheticCamera`` generates frames at a fixed rate for tests/benchmarks.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

from .metrics import PerfTracker
from .scheduler import FrameBuffer


class _BaseSource(threading.Thread):
    def __init__(self, slot: FrameBuffer, perf: Optional[PerfTracker] = None) -> None:
        super().__init__(daemon=True)
        self.slot, self.perf = slot, perf
        self._stop_evt = threading.Event()
        self.frames = 0
        self.ended = False  # the source has no more frames (video file / test script)
        self.error: Optional[BaseException] = None

    def stop(self) -> None:
        self._stop_evt.set()

    def _publish(self, frame: np.ndarray, capture_ms: Optional[float] = None) -> None:
        t = time.monotonic()
        self.slot.put(frame, t)
        self.frames += 1
        if self.perf is not None:
            self.perf.record_capture(t, capture_ms)


class CaptureThread(_BaseSource):
    """cap.read() in a loop -> FrameBuffer. ``max_consecutive_fails`` read
    failures in a row mean the stream ended (video file) or the camera went
    away; the loop then sets ``ended`` and exits."""

    def __init__(self, cap, slot: FrameBuffer, perf: Optional[PerfTracker] = None,
                 max_consecutive_fails: int = 30) -> None:
        super().__init__(slot, perf)
        self.cap = cap
        self.max_fails = max_consecutive_fails

    def run(self) -> None:
        fails = 0
        try:
            while not self._stop_evt.is_set():
                t0 = time.monotonic()
                ok, frame = self.cap.read()  # stage 1: camera capture
                if not ok or frame is None:
                    fails += 1
                    if fails >= self.max_fails:
                        break
                    time.sleep(0.005)
                    continue
                fails = 0
                self._publish(frame, (time.monotonic() - t0) * 1e3)
        except BaseException as e:  # surface the error to the main thread
            self.error = e
        finally:
            self.ended = True


class SyntheticCamera(_BaseSource):
    """Generates frames at a fixed rate (default 30 FPS) for tests/benchmarks."""

    def __init__(self, slot: FrameBuffer, fps: float = 30.0, shape=(480, 640, 3),
                 perf: Optional[PerfTracker] = None, max_frames: Optional[int] = None,
                 frame_fn: Optional[Callable[[int], np.ndarray]] = None) -> None:
        super().__init__(slot, perf)
        self.interval = 1.0 / fps
        self.shape = shape
        self.max_frames = max_frames
        self.frame_fn = frame_fn

    def run(self) -> None:
        next_t = time.monotonic()
        i = 0
        try:
            while not self._stop_evt.is_set() and (self.max_frames is None or i < self.max_frames):
                t0 = time.monotonic()
                frame = self.frame_fn(i) if self.frame_fn else np.zeros(self.shape, dtype=np.uint8)
                self._publish(frame, (time.monotonic() - t0) * 1e3)
                i += 1
                next_t += self.interval
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()
        finally:
            self.ended = True
